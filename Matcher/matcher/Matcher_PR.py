import os
from os import path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import numpy as np
import cv2
import ot
from scipy.optimize import linear_sum_assignment

from segment_anything import sam_model_registry
from segment_anything import SamAutomaticMaskGenerator
from dinov2.models import vision_transformer as vits
import dinov2.utils.utils as dinov2_utils
from dinov2.data.transforms import MaybeToTensor, make_normalize_transform

from matcher.k_means import kmeans_pp

import random


def _mask_image_uint8(image_hwc_uint8: np.ndarray, mask_hw: np.ndarray, bg_value: int = 0) -> np.ndarray:
    """Apply binary mask (HxW) to raw RGB image (HxWx3, uint8), filling background with bg_value."""
    m = (mask_hw > 0).astype(np.uint8)
    if m.sum() == 0:
        return image_hwc_uint8.copy()
    out = image_hwc_uint8.copy()
    out[~m.astype(bool)] = bg_value
    return out


class Matcher:
    def __init__(
        self,
        encoder,
        eta,
        gamma,
        generator=None,
        input_size=518,
        num_centers=8,
        use_box=False,
        use_points_or_centers=True,
        sample_range=(4, 6),
        max_sample_iterations=30,
        alpha=1.,
        beta=0.,
        exp=0.,
        score_filter_cfg=None,
        num_merging_mask=10,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        # IL-selection options
        il_metric='cos',     # 'cos' or 'kl'
        il_alpha=0.25,       # weight for SAM score
        il_beta=0.10,        # weight for margin
    ):
        # models
        self.encoder = encoder
        self.generator = generator
        self.rps = None

        if not isinstance(input_size, tuple):
            input_size = (input_size, input_size)
        self.input_size = input_size

        # transforms for image encoder
        self.encoder_transform = transforms.Compose([
            MaybeToTensor(),
            make_normalize_transform(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        self.tar_img = None
        self.tar_img_np = None

        self.ref_imgs = None
        self.ref_masks_pool = None
        self.nshot = None

        self.encoder_img_size = None
        self.encoder_feat_size = None

        self.num_centers = num_centers
        self.use_box = use_box
        self.use_points_or_centers = use_points_or_centers
        self.sample_range = sample_range
        self.max_sample_iterations = max_sample_iterations

        self.alpha, self.beta, self.exp = alpha, beta, exp
        assert score_filter_cfg is not None
        self.score_filter_cfg = score_filter_cfg
        self.num_merging_mask = num_merging_mask

        self.device = device
        self.eta = 0.0 if eta is None else float(eta)
        self.gamma = 0.0 if gamma is None else float(gamma)

        # grad map: kept in SAM feature space (1, C_sam, H_sam, W_sam)
        self.grad_map = None

        # IL-selection params & buffers
        self.il_metric = str(il_metric).lower()
        self.il_alpha = float(il_alpha)
        self.il_beta  = float(il_beta)

        self.ref_feats = None         # (N_dino, C_dino)
        self.tar_feat = None          # (N_dino, C_dino)

        # Raw support data for base score computation
        self.sup_img_np = None        # HxWx3 uint8
        self.sup_mask_raw = None      # HxW 0/1

        # Support embedding (vector/distribution)
        self.support_vec = None       # (1, C_dino) L2-normalized
        self.support_dist = None      # (1, C_dino) softmax distribution

    # -------------------------
    # Public API
    # -------------------------
    def set_reference(self, imgs, masks):

        def reference_masks_verification(masks):
            if masks.sum() == 0:
                _, _, sh, sw = masks.shape
                masks[..., (sh // 2 - 7):(sh // 2 + 7), (sw // 2 - 7):(sw // 2 + 7)] = 1
            return masks

        # Store raw support data (first shot; compatible with bsz=1, nshot=1)
        masks_raw_copy = masks.detach().clone()
        mflat = masks_raw_copy.reshape(-1, masks_raw_copy.shape[-2], masks_raw_copy.shape[-1])
        self.sup_mask_raw = (mflat[0].detach().cpu().numpy() > 0).astype(np.uint8)  # HxW

        imgs = imgs.flatten(0, 1)  # (bs*nshot, 3, H, W)
        img_size = imgs.shape[-1]
        assert img_size == self.input_size[-1]
        feat_size = img_size // self.encoder.patch_size

        # Save first support image as numpy uint8 HxWx3
        self.sup_img_np = (imgs[0].detach().cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)

        # DINO token setup
        self.encoder_img_size = img_size
        self.encoder_feat_size = feat_size

        # Process reference masks to patch grid
        masks = reference_masks_verification(masks)
        masks = masks.permute(1, 0, 2, 3)  # ns, 1, h, w
        ref_masks_pool = F.avg_pool2d(masks, (self.encoder.patch_size, self.encoder.patch_size))
        nshot = ref_masks_pool.shape[0]
        th = getattr(self.generator.predictor.model, 'mask_threshold', 0.0) \
             if hasattr(self.generator, 'predictor') and hasattr(self.generator.predictor, 'model') else 0.0
        ref_masks_pool = (ref_masks_pool > th).float()
        ref_masks_pool = ref_masks_pool.reshape(-1)  # nshot*N

        self.ref_imgs = imgs
        self.ref_masks_pool = ref_masks_pool
        self.nshot = nshot

        # Build support vector from pixel-masked re-encoding
        self._build_support_vector_with_raw_mask()

    def set_target(self, img):
        img_h, img_w = img.shape[-2:]
        assert img_h == self.input_size[0] and img_w == self.input_size[1]

        img_np = img.mul(255).byte()
        img_np = img_np.squeeze(0).permute(1, 2, 0).cpu().numpy()

        self.tar_img = img
        self.tar_img_np = img_np

    def set_rps(self):
        if self.rps is None:
            assert self.encoder_feat_size is not None
            self.rps = RobustPromptSampler(
                encoder_feat_size=self.encoder_feat_size,
                sample_range=self.sample_range,
                max_iterations=self.max_sample_iterations
            )

    @torch.no_grad()
    def extract_img_feats(self):
        ref_imgs = torch.cat([self.encoder_transform(rimg)[None, ...] for rimg in self.ref_imgs], dim=0)
        tar_img  = torch.cat([self.encoder_transform(timg)[None, ...] for timg in self.tar_img], dim=0)

        ref_feats = self.encoder.forward_features(ref_imgs.to(self.device))["x_prenorm"][:, 1:]
        tar_feat  = self.encoder.forward_features(tar_img.to(self.device))["x_prenorm"][:, 1:]

        ref_feats = ref_feats.reshape(-1, self.encoder.embed_dim)  # ns*N, C_dino
        tar_feat  = tar_feat.reshape(-1, self.encoder.embed_dim)   # N_dino, C_dino

        ref_feats = F.normalize(ref_feats, dim=1, p=2)
        tar_feat  = F.normalize(tar_feat,  dim=1, p=2)

        return ref_feats, tar_feat

    @torch.no_grad()
    def _encode_masked_image_vector_dino(self, image_uint8: np.ndarray, mask_hw: np.ndarray):
        """
        Apply mask to raw RGB image (HxWx3, uint8), encode with DINO,
        and return mean patch token as 1xC L2-normalized vector.
        """
        masked = _mask_image_uint8(image_uint8, mask_hw, bg_value=0)          # HxWx3 uint8
        img_t  = torch.from_numpy(masked).permute(2, 0, 1).float() / 255.0    # 3xHxW
        enc_in = self.encoder_transform(img_t)[None, ...].to(self.device)     # 1x3xHxW
        feat   = self.encoder.forward_features(enc_in)["x_prenorm"][:, 1:]    # 1xNxC
        v      = feat.mean(dim=1)                                              # 1xC
        v_norm = F.normalize(v, dim=1, p=2)
        p_dist = torch.softmax(v, dim=1)
        return v_norm, p_dist

    def _build_support_vector_with_raw_mask(self):
        """Compute support_vec/support_dist from (support_image ⊙ support_mask)."""
        if self.sup_img_np is None or self.sup_mask_raw is None:
            self.support_vec = None
            self.support_dist = None
            return
        v_norm, p_dist = self._encode_masked_image_vector_dino(self.sup_img_np, self.sup_mask_raw)
        self.support_vec  = v_norm   # 1xC
        self.support_dist = p_dist   # 1xC

    def _map_sam_grad_to_dino(self, grad_sam):
        """
        Map SAM feature-space grad (1, C_sam, H_sam, W_sam) to DINO token space (N_dino, C_dino).
        Heuristic: spatial resize to (H_dino,W_dino), then pad/trim channels to C_dino and flatten.
        """
        if grad_sam is None:
            return None
        g = grad_sam
        if g.dim() != 4:
            return None
        _, C_sam, H_sam, W_sam = g.shape
        Hd = Wd = self.encoder_feat_size
        Cd = self.encoder.embed_dim

        # spatial resize
        g_sp = F.interpolate(g, size=(Hd, Wd), mode='bilinear', align_corners=False)  # 1, C_sam, Hd, Wd
        g_flat = g_sp.view(1, C_sam, -1).permute(0, 2, 1).squeeze(0)                  # (N_dino, C_sam)

        # channel adapt
        if C_sam < Cd:
            pad = torch.zeros(g_flat.size(0), Cd - C_sam, device=g_flat.device, dtype=g_flat.dtype)
            g_adapt = torch.cat([g_flat, pad], dim=1)
        else:
            g_adapt = g_flat[:, :Cd]

        return g_adapt  # (N_dino, C_dino)

    def _compute_base_and_margin(self, merged_mask_np, tar_tokens_used):
        """
        base:   cosine similarity between (support_image ⊙ support_mask) and
                (query_image ⊙ merged_mask) encoded with DINO.
        margin: fixed to 0.0 (patch-wise variant disabled).
        """
        q_vec, q_dist = self._encode_masked_image_vector_dino(self.tar_img_np, merged_mask_np.astype(np.uint8))
        base = torch.sum(self.support_vec * q_vec, dim=1).item()
        margin = 0.0
        return float(base), float(margin)

    def predict(self, nested, idx, benchmark, fold, args):
        """
        Returns (pred_mask_tensor[B,1,H,W], meta_dict)
        meta: {'base', 'margin', 'sam_score', 'sel_score', 'iter'}
        """
        # Prepare features
        if nested == 0:
            self.ref_feats, self.tar_feat = self.extract_img_feats()
            self.grad_map = None
            tar_tokens_used = self.tar_feat
        else:
            # Map SAM-space grad to DINO token space and inject
            delta = self._map_sam_grad_to_dino(self.grad_map)
            if delta is not None:
                tar_tokens_used = self.tar_feat + delta
            else:
                tar_tokens_used = self.tar_feat

        # Patch matching -> points/box
        all_points, box, S, C, reduced_points_num = self.patch_level_matching(ref_feats=self.ref_feats, tar_feat=tar_tokens_used)
        points = self.clustering(all_points) if not self.use_points_or_centers else all_points

        self.set_rps()

        # Mask generation (SAM grad update happens internally)
        pred_masks, meta_mask = self.mask_generation(self.tar_img_np, points, box, all_points, self.ref_masks_pool, C)

        # IL-selection meta (base, margin, sam_score -> sel_score)
        merged_np = meta_mask.get('merged_mask_np', None)
        sam_score = float(meta_mask.get('sam_score', 0.0))
        if merged_np is not None:
            base, margin = self._compute_base_and_margin(merged_np.astype(np.uint8), tar_tokens_used)
        else:
            base, margin = 0.0, 0.0
        sel_score = base + self.il_alpha * sam_score + self.il_beta * margin

        meta = dict(base=base, margin=margin, sam_score=sam_score, sel_score=sel_score, iter=nested)
        return pred_masks, meta

    # -------------------------
    # Internal utilities
    # -------------------------
    def patch_level_matching(self, ref_feats, tar_feat):
        # forward matching
        S = ref_feats @ tar_feat.t()  # ns*N, N
        C = (1 - S) / 2  # distance

        S_forward = S[self.ref_masks_pool.flatten().bool()]
        indices_forward = linear_sum_assignment(S_forward.detach().cpu(), maximize=True)
        indices_forward = [torch.as_tensor(index, dtype=torch.int64, device=self.device) for index in indices_forward]
        sim_scores_f = S_forward[indices_forward[0], indices_forward[1]]
        indices_mask = self.ref_masks_pool.flatten().nonzero()[:, 0]

        # reverse matching
        S_reverse = S.t()[indices_forward[1]]
        indices_reverse = linear_sum_assignment(S_reverse.detach().cpu(), maximize=True)
        indices_reverse = [torch.as_tensor(index, dtype=torch.int64, device=self.device) for index in indices_reverse]
        retain_ind = torch.isin(indices_reverse[1], indices_mask)
        if not (retain_ind == False).all().item():
            indices_forward = [indices_forward[0][retain_ind], indices_forward[1][retain_ind]]
            sim_scores_f = sim_scores_f[retain_ind]

        reduced_points_num = len(sim_scores_f) // 2 if len(sim_scores_f) > 40 else len(sim_scores_f)
        sim_sorted, sim_idx_sorted = torch.sort(sim_scores_f, descending=True)
        sim_filter = sim_idx_sorted[:reduced_points_num]
        points_matched_inds = indices_forward[1][sim_filter]

        points_matched_inds_set = torch.tensor(list(set(points_matched_inds.detach().cpu().tolist())))
        points_matched_inds_set_w = points_matched_inds_set % (self.encoder_feat_size)
        points_matched_inds_set_h = points_matched_inds_set // (self.encoder_feat_size)
        idxs_mask_set_x = (points_matched_inds_set_w * self.encoder.patch_size + self.encoder.patch_size // 2).tolist()
        idxs_mask_set_y = (points_matched_inds_set_h * self.encoder.patch_size + self.encoder.patch_size // 2).tolist()

        ponits_matched = []
        for x, y in zip(idxs_mask_set_x, idxs_mask_set_y):
            if int(x) < self.input_size[1] and int(y) < self.input_size[0]:
                ponits_matched.append([int(x), int(y)])
        ponits = np.array(ponits_matched)

        if self.use_box and len(ponits) > 0:
            box = np.array([
                max(ponits[:, 0].min(), 0),
                max(ponits[:, 1].min(), 0),
                min(ponits[:, 0].max(), self.input_size[1] - 1),
                min(ponits[:, 1].max(), self.input_size[0] - 1),
            ])
        else:
            box = None
        return ponits, box, S, C, reduced_points_num

    def clustering(self, points):
        num_centers = min(self.num_centers, len(points))
        flag = True
        while flag:
            centers, cluster_assignment = kmeans_pp(points, num_centers)
            ids, fre = torch.unique(torch.as_tensor(cluster_assignment), return_counts=True)
            if ids.shape[0] == num_centers:
                flag = False
            else:
                print('Kmeans++ failed, re-run')
        centers = np.array(centers).astype(np.int64)
        return centers

    def mask_generation(self, tar_img_np, points, box, all_ponits, ref_masks_pool, C):
        # RPS sampling -> SAM generation
        samples_list, label_list = self.rps.sample_points(points)
        tar_masks_ori = self.generator.generate(
            tar_img_np,
            select_point_coords=samples_list,
            select_point_labels=label_list,
            select_box=[box] if self.use_box else None,
        )

        # Collect SAM candidate masks/logits/scores
        cand_pred_masks = []
        cand_logits = []
        cand_sam_scores = []  # predicted_iou or stability_score fallback

        for qmask in tar_masks_ori:
            seg = qmask['segmentation']  # HxW
            cand_pred_masks.append(torch.from_numpy(seg).float()[None, None, ...].to(self.device))

            if 'logits' in qmask and isinstance(qmask['logits'], torch.Tensor):
                lg = qmask['logits'].to(self.device)
            elif 'logits' in qmask:
                lg = torch.from_numpy(qmask['logits']).to(self.device)
            else:
                lg = torch.from_numpy(seg.astype(np.float32)).to(self.device)
            cand_logits.append(lg[None, None, ...] if lg.dim() == 2 else lg)

            s = float(qmask.get('predicted_iou', qmask.get('stability_score', 0.0)))
            cand_sam_scores.append(s)

        pred_masks = torch.cat(cand_pred_masks, dim=0).detach().cpu().numpy() > 0
        pred_logits = torch.cat(cand_logits, dim=0)  # (N,1,h',w') or (N,h',w')

        # Compute EMD/purity/coverage metrics
        purity = torch.zeros(pred_masks.shape[0])
        coverage = torch.zeros(pred_masks.shape[0])
        emd = torch.zeros(pred_masks.shape[0])

        for i in range(len(pred_masks)):
            purity_, coverage_, emd_, sample_, label_, mask_ = \
                self.rps.get_mask_scores(
                    points=points,
                    masks=pred_masks[i],
                    all_points=all_ponits,
                    emd_cost=C,
                    ref_masks_pool=ref_masks_pool
                )
            assert np.all(mask_ == pred_masks[i])
            purity[i] = purity_
            coverage[i] = coverage_
            emd[i] = emd_

        pred_masks_np = pred_masks.squeeze(1)                      # (N,H,W)
        metric_preds = {"purity": purity, "coverage": coverage, "emd": emd}
        scores = self.alpha * emd + self.beta * purity * (coverage ** self.exp)

        def check_pred_mask(x):
            x = np.asarray(x)
            if x.ndim < 3:
                x = x[None, ...]
            return x

        def check_pred_logits(x):
            if x.dim() < 3:
                x = x.unsqueeze(0)
            return x

        pred_masks_np = check_pred_mask(pred_masks_np)

        # Metric-based filtering
        for metric in ["coverage", "emd", "purity"]:
            th = float(self.score_filter_cfg.get(metric, 0.0))
            if th > 0:
                th = min(th, float(metric_preds[metric].max()))
                idx = torch.where(metric_preds[metric] >= th)[0]
                if idx.numel() == 0:
                    continue
                scores = scores[idx]
                pred_masks_np = check_pred_mask(pred_masks_np[idx])
                cand_sam_scores = list(np.array(cand_sam_scores)[idx.cpu().numpy()])
                for k in metric_preds.keys():
                    metric_preds[k] = metric_preds[k][idx]
                pred_logits = pred_logits[idx]

        # Score-based selection and merging
        chosen_indices = None
        if self.score_filter_cfg.get("score_filter", False):
            distances = 1 - scores
            distances, rank = torch.sort(distances, descending=False)
            distances_norm = (distances - distances.min()) / (distances.max() + 1e-6)
            filer_dis = (distances < float(self.score_filter_cfg.get("score", 0.33)))
            if filer_dis.numel() > 0:
                filer_dis[..., 0] = True
            filer_dis = filer_dis & (distances_norm < float(self.score_filter_cfg.get("score_norm", 0.1)))

            chosen_indices = rank[filer_dis][: self.num_merging_mask]
            masks = pred_masks_np[chosen_indices]
            masks = check_pred_mask(masks)
            logits = pred_logits[chosen_indices]
            logits = check_pred_logits(logits)
            masks = (masks.sum(0) > 0)[None, ...]  # (1,H,W)
        else:
            topk = int(min(self.num_merging_mask, scores.numel()))
            topk_idx = scores.topk(topk).indices
            topk_masks = pred_masks_np[topk_idx]
            topk_masks = check_pred_mask(topk_masks)

            if float(self.score_filter_cfg.get("topk_scores_threshold", 0.0)) > 0:
                topk_scores = scores[topk_idx].cpu().numpy()
                topk_scores = topk_scores / (topk_scores.max() + 1e-6)
                keep = (topk_scores > float(self.score_filter_cfg["topk_scores_threshold"]))
                topk_idx = topk_idx[keep]
                topk_masks = check_pred_mask(topk_masks[keep])

            chosen_indices = topk_idx
            if topk_masks.size == 0:
                topk_masks = check_pred_mask(pred_masks_np[topk_idx])
            masks = (topk_masks.sum(0) > 0)[None, ...]  # (1,H,W)
            logits = pred_logits[topk_idx]
            logits = check_pred_logits(logits)

        # Aggregate SAM score over chosen candidates
        if chosen_indices is not None and len(cand_sam_scores) > 0:
            chosen_idx_list = chosen_indices.detach().cpu().numpy().tolist() if torch.is_tensor(chosen_indices) else list(chosen_indices)
            chosen_idx_list = [int(i) for i in chosen_idx_list if 0 <= int(i) < len(cand_sam_scores)]
            if len(chosen_idx_list) > 0:
                sam_score = float(np.mean([cand_sam_scores[i] for i in chosen_idx_list]))
            else:
                sam_score = float(np.mean(cand_sam_scores))
        else:
            sam_score = float(np.mean(cand_sam_scores)) if len(cand_sam_scores) > 0 else 0.0

        # Gradient flow update in SAM feature space
        gradient_feature = torch.autograd.grad(
            outputs=logits, inputs=self.generator.predictor.features,
            grad_outputs=torch.ones_like(logits),
            create_graph=True, retain_graph=False
        )[0]

        s = torch.ones_like(self.generator.predictor.features.detach())
        eta = self.eta
        gamma = self.gamma
        noise_std = np.sqrt(max(0.0, gamma))
        v = s.data * torch.clamp(gradient_feature.data, min=-0.1, max=0.1).detach()
        step = eta * v + np.sqrt(2 * eta) * noise_std * torch.randn_like(self.generator.predictor.features)

        if self.grad_map is None:
            self.grad_map = step.detach()
        else:
            self.grad_map = (self.grad_map + step).detach()

        # Reset SAM predictor internal state
        if hasattr(self.generator, 'predictor') and hasattr(self.generator.predictor, 'reset_image'):
            self.generator.predictor.reset_image()

        out = torch.tensor(masks, device=self.device, dtype=torch.float32).detach()  # (1,H,W)

        meta = {
            'sam_score': sam_score,
            'merged_mask_np': masks.squeeze(0).astype(np.uint8),  # HxW 0/1
        }
        return out, meta

    def clear(self):
        # Reset predictor state
        if hasattr(self.generator, 'predictor') and hasattr(self.generator.predictor, 'reset_predictor'):
            try:
                self.generator.predictor.reset_predictor()
            except Exception:
                pass

        # Clear member state
        self.tar_img = None
        self.tar_img_np = None

        self.ref_imgs = None
        self.ref_masks_pool = None
        self.nshot = None

        self.encoder_img_size = None
        self.encoder_feat_size = None

        self.grad_map = None
        self.ref_feats = None
        self.tar_feat = None

        self.sup_img_np = None
        self.sup_mask_raw = None
        self.support_vec = None
        self.support_dist = None

        torch.cuda.empty_cache()


class RobustPromptSampler:

    def __init__(self, encoder_feat_size, sample_range, max_iterations):
        self.encoder_feat_size = encoder_feat_size
        self.sample_range = sample_range
        self.max_iterations = max_iterations

    def get_mask_scores(self, points, masks, all_points, emd_cost, ref_masks_pool):

        def is_in_mask(point, mask):
            # input: point: n*2, mask: h*w
            # output: n*1
            h, w = mask.shape
            point = point.astype(np.int64)
            point = point[:, ::-1]  # y,x
            point = np.clip(point, 0, [h - 1, w - 1])
            return mask[point[:, 0], point[:, 1]]

        ori_masks = masks
        masks_small = cv2.resize(
            masks[0].astype(np.float32),
            (self.encoder_feat_size, self.encoder_feat_size),
            interpolation=cv2.INTER_AREA
        )
        thres = 0 if masks_small.max() > 0 else (masks_small.max() - 1e-6)
        masks_small = masks_small > thres

        # EMD score
        emd_cost_pool = emd_cost[ref_masks_pool.flatten().bool(), :][:, masks_small.flatten()]
        emd = ot.emd2(
            a=[1. / emd_cost_pool.shape[0] for _ in range(emd_cost_pool.shape[0])],
            b=[1. / emd_cost_pool.shape[1] for _ in range(emd_cost_pool.shape[1])],
            M=emd_cost_pool.detach().cpu().numpy()
        )
        emd_score = 1 - emd

        labels = np.ones((points.shape[0],))

        # Purity / coverage
        assert all_points is not None
        points_in_mask = is_in_mask(all_points, ori_masks[0])
        points_in_mask = all_points[points_in_mask]

        mask_area = max(float(masks_small.sum()), 1.0)
        purity = points_in_mask.shape[0] / mask_area
        coverage = points_in_mask.shape[0] / all_points.shape[0]
        purity = torch.tensor([purity]) + 1e-6
        coverage = torch.tensor([coverage]) + 1e-6
        return purity, coverage, emd_score, points, labels, ori_masks

    def combinations(self, n, k):
        if k > n:
            return []
        if k == 0:
            return [[]]
        if k == n:
            return [[i for i in range(n)]]
        res = []
        for i in range(n):
            for j in self.combinations(i, k - 1):
                res.append(j + [i])
        return res

    def sample_points(self, points):
        # return list of array
        sample_list = []
        label_list = []
        for i in range(min(self.sample_range[0], len(points)), min(self.sample_range[1], len(points)) + 1):
            if len(points) > 8:
                index = [random.sample(range(len(points)), i) for _ in range(self.max_iterations)]
                sample = np.take(points, index, axis=0)  # (max_iterations * i) * 2
            else:
                index = self.combinations(len(points), i)
                sample = np.take(points, index, axis=0)  # i * n * 2

            label = np.ones((sample.shape[0], i))
            sample_list.append(sample)
            label_list.append(label)
        return sample_list, label_list


def build_matcher_oss(args):

    # DINOv2 image encoder
    dinov2_kwargs = dict(
        img_size=518,
        patch_size=14,
        init_values=1e-5,
        ffn_layer='mlp',
        block_chunks=0,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
    )
    dinov2 = vits.__dict__[args.dinov2_size](**dinov2_kwargs)

    dinov2_utils.load_pretrained_weights(dinov2, args.dinov2_weights, "teacher")
    dinov2.eval()
    dinov2.to(device=args.device)
    for param in dinov2.parameters():
        param.requires_grad_(False)

    # SAM
    sam = sam_model_registry[args.sam_size](checkpoint=args.sam_weights)
    sam.to(device=args.device)
    generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=args.points_per_side,
        points_per_batch=64,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        stability_score_offset=1.0,
        sel_stability_score_thresh=args.sel_stability_score_thresh,
        sel_pred_iou_thresh=args.iou_filter,
        box_nms_thresh=args.box_nms_thresh,
        sel_output_layer=args.output_layer,
        output_layer=args.dense_multimask_output,
        dense_pred=args.use_dense_mask,
        multimask_output=args.dense_multimask_output > 0,
        sel_multimask_output=args.multimask_output > 0,
    )

    score_filter_cfg = {
        "emd": args.emd_filter,
        "purity": args.purity_filter,
        "coverage": args.coverage_filter,
        "score_filter": args.use_score_filter,
        "score": args.deep_score_filter,
        "score_norm": args.deep_score_norm_filter,
        "topk_scores_threshold": args.topk_scores_threshold
    }

    # Pass IL-selection hyperparameters
    il_metric = getattr(args, 'il_metric', getattr(args, 'il-metric', 'cos'))
    il_alpha  = getattr(args, 'score_alpha', 0.25)
    il_beta   = getattr(args, 'score_beta', 0.10)

    return Matcher(
        encoder=dinov2,
        generator=generator,
        num_centers=args.num_centers,
        use_box=args.use_box,
        use_points_or_centers=args.use_points_or_centers,
        sample_range=args.sample_range,
        max_sample_iterations=args.max_sample_iterations,
        alpha=args.alpha,
        beta=args.beta,
        exp=args.exp,
        score_filter_cfg=score_filter_cfg,
        num_merging_mask=args.num_merging_mask,
        device=args.device,
        eta=args.eta,
        gamma=args.gamma,
        il_metric=il_metric,
        il_alpha=il_alpha,
        il_beta=il_beta,
    )
