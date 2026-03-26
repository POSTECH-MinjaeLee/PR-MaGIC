import os
import argparse
import warnings
from datetime import datetime
import random
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F

from per_segment_anything import sam_model_registry, SamPredictor
from evaluation import Evaluator
from datasets.dataset import FSSDataset


# =========================================
# Reproducibility
# =========================================
def fix_randseed(seed):
    if seed is None:
        seed = int(random.random() * 1e5)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# =========================================
# Arguments
# =========================================
def get_arguments():
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument('--datapath', type=str, default='../data')
    parser.add_argument('--benchmark', type=str, default='coco', choices=['fss', 'coco', 'pascal', 'lvis', 'paco_part', 'pascal_part', 'dis'])
    parser.add_argument('--bsz', type=int, default=1)
    parser.add_argument('--nworker', type=int, default=0)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--nshot', type=int, default=1)
    parser.add_argument('--img-size', type=int, default=518)
    parser.add_argument('--use_original_imgsize', action='store_true')
    parser.add_argument('--points_num', type=int, default=0)

    # SAM
    parser.add_argument('--ckpt', type=str, default='../weights/sam_vit_h_4b8939.pth')
    parser.add_argument('--sam_type', type=str, default='vit_h')
    parser.add_argument('--seed', type=int, default=42)

    # Refinement / PR-MaGIC
    parser.add_argument('--nested', type=int, default=6)
    parser.add_argument('--eta', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=1e-1)

    # Selection metric
    parser.add_argument('--f', type=str, default='KL', choices=['KL', 'kl', 'cos', 'COS'])

    # Device & misc
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--tracking', type=int, default=1)

    # Mask-weights tuner
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--train_epoch', type=int, default=1000)
    parser.add_argument('--log_epoch', type=int, default=200)
    parser.add_argument('--ref_idx', type=str, default='1')

    # IL selection score weights (single)
    parser.add_argument('--score_alpha', type=float, default=0.25)  # SAM score weight
    parser.add_argument('--score_beta',  type=float, default=0.10)  # margin weight

    # Multi-sweep (comma-separated)
    parser.add_argument('--alpha_list', type=str, default="")
    parser.add_argument('--beta_list',  type=str, default="")

    # I/O
    parser.add_argument('--log-root', type=str, default='',
                        help='Output directory (overrides auto-generated path if set)')
    parser.add_argument('--outdir', type=str, default='persam_f')

    args = parser.parse_args()
    if args.log_root:
        args.outdir = args.log_root
    else:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.outdir = os.path.join('./save', args.benchmark, f"{ts}-eta{args.eta}-gamma{args.gamma}")
    os.makedirs(args.outdir, exist_ok=True)
    return args


# =========================================
# Small utils
# =========================================
class Mask_Weights(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(2, 1, requires_grad=True) / 3)


def point_selection(mask_sim, device, num_extra, topk=1):
    w, h = mask_sim.shape
    top1_val, top1_idx = mask_sim.view(-1).max(0)
    top1_x = top1_idx // h
    top1_y = top1_idx % h
    top1_xy = torch.tensor([[top1_y.item(), top1_x.item()]]).to(device)

    quantiles = [(0.99999, num_extra), (0.9999, num_extra)]
    selected_indices = set([top1_idx.item()])
    selected_xy = []

    for q, num_points in quantiles:
        threshold = torch.quantile(mask_sim, q)
        mask = mask_sim >= threshold
        for idx in list(selected_indices):  # safe: prevent modification during iteration
            x = idx // h
            y = idx % h
            mask[x, y] = False
        candidates = torch.nonzero(mask, as_tuple=False)
        if candidates.shape[0] == 0:
            continue
        num_to_select = min(num_points, candidates.shape[0])
        sampled_indices = torch.randperm(candidates.shape[0])[:num_to_select]
        sampled_points = candidates[sampled_indices]
        for point in sampled_points:
            y, x = point.tolist()
            all_idx = x * h + y
            if all_idx not in selected_indices:
                selected_xy.append([y, x])
                selected_indices.add(all_idx)

    if selected_xy:
        selected_xy = torch.tensor(selected_xy, device=device)
        all_xy = torch.cat((top1_xy, selected_xy), dim=0)
    else:
        all_xy = top1_xy

    all_labels = np.ones(all_xy.shape[0], dtype=int)
    all_xy = all_xy.cpu().numpy()
    return all_xy, all_labels


def calculate_dice_loss(inputs, targets, num_masks=1):
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


def calculate_sigmoid_focal_loss(inputs, targets, num_masks=1, alpha: float = 0.25, gamma: float = 2):
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean(1).sum() / num_masks


# =========================================
# Feature-based similarity map (baseline)
# =========================================
@torch.no_grad()
def _compute_sim_from_features(predictor, target_feat, cur_feat_bchw):
    feat = cur_feat_bchw.squeeze(0)  # C x h x w
    C, h, w = feat.shape
    feat = feat / (feat.norm(dim=0, keepdim=True) + 1e-8)
    feat = feat.reshape(C, h * w)
    sim = target_feat @ feat  # 1 x (h*w)
    sim = sim.reshape(1, 1, h, w)
    sim = F.interpolate(sim, scale_factor=4, mode="bilinear")
    sim = predictor.model.postprocess_masks(
        sim, predictor.input_size, predictor.original_size
    ).squeeze()
    return sim


# =========================================
# Pixel-masked re-encoding utilities
# =========================================
def _mask_image_uint8(image_hwc_uint8: np.ndarray, mask_hw: np.ndarray, bg_value: int = 0) -> np.ndarray:
    """image: HxWx3 uint8 in [0,255], mask: HxW (0/1 or 0..255)."""
    m = (mask_hw > 0).astype(np.uint8)
    if m.sum() == 0:
        return image_hwc_uint8.copy()  # use original image for empty mask
    out = image_hwc_uint8.copy()
    out[~m.astype(bool)] = bg_value
    return out


@torch.no_grad()
def _encode_masked_image_vector(predictor: SamPredictor, image_uint8: np.ndarray, mask_hw: np.ndarray):
    """
    Mask pixels then encode with encode_image() to get 1xCxH'xW' embedding.
    Returns channel-mean 1xC vector and softmax distribution.
    """
    masked = _mask_image_uint8(image_uint8, mask_hw, bg_value=0)
    emb = predictor.encode_image(masked)                 # 1 x C x H' x W'
    v = emb.mean(dim=(2, 3))                             # 1 x C
    v_norm = v / (v.norm(dim=-1, keepdim=True) + 1e-8)  # 1 x C
    p = torch.softmax(v, dim=-1)                         # 1 x C (distribution)
    return v_norm, p


def _kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8):
    # p, q: 1 x C
    return torch.sum(p * (torch.log(p + eps) - torch.log(q + eps)), dim=-1)  # 1


def _mask_margin_from_logit(logit_high_2d, mask_np):
    p = torch.sigmoid(logit_high_2d)
    mask_t = torch.from_numpy(mask_np.astype(np.float32)).to(p.device)
    if (mask_t > 0).any() and (mask_t == 0).any():
        pin = p[mask_t > 0].mean()
        pout = p[mask_t == 0].mean()
        return float(pin - pout)
    else:
        return 0.0


def _parse_float_list(s: str, fallback: float):
    s = (s or "").strip()
    if not s:
        return [fallback]
    vals = []
    for tok in s.split(','):
        tok = tok.strip()
        if tok:
            vals.append(float(tok))
    uniq = []
    for v in vals:
        if v not in uniq:
            uniq.append(v)
    return uniq


# =========================================
# Main
# =========================================
def main():
    args = get_arguments()
    fix_randseed(args.seed)

    img_size = None if args.use_original_imgsize else args.img_size
    FSSDataset.initialize(img_size=img_size, datapath=args.datapath, use_original_imgsize=args.use_original_imgsize)
    dataloader_test = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'test', args.nshot)

    os.makedirs(args.outdir, exist_ok=True)

    p_bar = tqdm(dataloader_test, total=len(dataloader_test))

    total_mious = {}
    baseline0_mious = []
    mean_after0_list = []
    oracle_after0_list = []

    # IL sweep preparation
    alpha_list = _parse_float_list(args.alpha_list, args.score_alpha)
    beta_list  = _parse_float_list(args.beta_list,  args.score_beta)
    grid = [(a, b) for a in alpha_list for b in beta_list]
    K = len(grid)
    if K == 0:
        grid = [(args.score_alpha, args.score_beta)]
        K = 1

    il_mious_grid = [[] for _ in range(K)]
    il_iters_grid = [[] for _ in range(K)]
    per_sample_best_iters = {}

    print("\n[IL4-* index mapping]")
    for i, (a, b) in enumerate(grid, start=1):
        print(f"  IL4-{i}: alpha={a}, beta={b}")

    device = 'cuda:{}'.format(args.device) if torch.cuda.is_available() else "cpu"
    if args.sam_type == 'vit_h':
        sam_type, sam_ckpt = 'vit_h', args.ckpt
        sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).to(device)
    elif args.sam_type == 'vit_t':
        sam_type, sam_ckpt = 'vit_t', args.ckpt
        sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).to(device)
    else:
        raise ValueError(f"Unknown sam_type: {args.sam_type}")
    sam.eval()

    def _safe_mean(xs):
        return float(sum(xs) / len(xs)) if xs else 0.0

    for batch_idx, batch in enumerate(p_bar):
        class_id = batch['class_id'].item()
        obj_name = f'{batch_idx}_class-{class_id}'

        try:
            miou_list, _, _, oracle_index, best_iter_default, best_iter_list = persam_f(
                args, obj_name, batch, args.outdir, sam=sam, grid=grid)
        except Exception as e:
            warnings.warn(f"Error at {obj_name}: {e}")
            continue

        total_mious[obj_name] = miou_list

        if len(miou_list) >= 1:
            baseline0_mious.append(miou_list[0])
        if len(miou_list) > 1:
            mean_after0_list.append(sum(miou_list[1:]) / (len(miou_list) - 1))
            oracle_after0_list.append(max(miou_list))

        # Per-sample IL grid stats
        per_sample_best_iters[obj_name] = best_iter_list
        for k, it in enumerate(best_iter_list):
            if 0 <= it < len(miou_list):
                il_mious_grid[k].append(miou_list[it])
                il_iters_grid[k].append(it)

        # Periodic progress report
        if batch_idx % 100 == 0:
            print(f"\n[{batch_idx}/{len(dataloader_test)}]")
            print(f"  (1) Baseline@0    : {_safe_mean(baseline0_mious):.4f}")
            print(f"  (2) Mean(1..T)   : {_safe_mean(mean_after0_list):.4f}")
            print(f"  (3) Oracle(0..T) : {_safe_mean(oracle_after0_list):.4f}")
            for i in range(K):
                print(f"  (4) IL4-{i+1}     : {_safe_mean(il_mious_grid[i]):.4f} | Avg iter {_safe_mean(il_iters_grid[i]):.2f}")

    # Final report
    print(f"\n--- Final Results (seed:{args.seed})---")
    res1 = _safe_mean(baseline0_mious)
    res2 = _safe_mean(mean_after0_list)
    res3 = _safe_mean(oracle_after0_list)
    print(f"1) Baseline (no refinement, iter 0): Avg mIoU = {res1:.4f}")
    print(f"2) Avg of per-sample mean mIoU over iters 1..T: {res2:.4f}")
    print(f"3) Oracle (best over iters 0..T): Avg mIoU = {res3:.4f}")
    for i, (a, b) in enumerate(grid):
        miou_avg = float(sum(il_mious_grid[i]) / len(il_mious_grid[i])) if il_mious_grid[i] else 0.0
        iter_avg = float(sum(il_iters_grid[i]) / len(il_iters_grid[i])) if il_iters_grid[i] else 0.0
        print(f"4-{i+1}) IL-stop (alpha={a}, beta={b}): Avg mIoU = {miou_avg:.4f} | Avg stop iter = {iter_avg:.2f} (0-based), {iter_avg+1:.2f} (1-based)")

    # CSV output
    csv_dir = os.path.join(args.outdir, f'{args.fold}/csvs')
    os.makedirs(csv_dir, exist_ok=True)

    # (A) per-sample mIoU per iteration
    df_miou = pd.DataFrame.from_dict(total_mious, orient='index')
    df_miou.index.name = 'obj_name'
    df_miou.columns = [f'iter_{i}' for i in range(df_miou.shape[1])]
    df_miou.reset_index(inplace=True)
    path_miou = os.path.join(csv_dir, 'miou_per_iter.csv')
    df_miou.to_csv(path_miou, index=False)
    print(f"Saved: {path_miou}")

    # (B) per-sample best iteration for each (alpha,beta)
    if per_sample_best_iters:
        col_names = [f'IL4_{i+1}_best_iter' for i in range(K)]
        rows = []
        for k, (obj, bests) in enumerate(per_sample_best_iters.items()):
            row = {'obj_name': obj}
            for i in range(K):
                row[col_names[i]] = bests[i] if i < len(bests) else -1
            rows.append(row)
        df_best = pd.DataFrame(rows, columns=['obj_name'] + col_names)
        path_best = os.path.join(csv_dir, 'best_iters_grid.csv')
        df_best.to_csv(path_best, index=False)
        print(f"Saved: {path_best}")


# =========================================
# Core: PerSAM-F with pixel-masked re-encoding for IL selection
# =========================================
def persam_f(args, obj_name, batch, output_path, sam=None, grid=None):
    device = 'cuda:{}'.format(args.device) if torch.cuda.is_available() else "cpu"

    T = args.nested
    eta = args.eta
    gamma = args.gamma
    num_extra = (args.points_num or 2) // 2
    noise_factor = np.sqrt(gamma)
    grid = grid or [(args.score_alpha, args.score_beta)]
    K = len(grid)

    # Load support/query
    ref_image = batch['support_imgs'].squeeze().permute(1, 2, 0).numpy()
    ref_image = (ref_image * 255).astype(np.uint8)  # HxWx3 uint8
    ref_mask_3c = batch['support_masks'].squeeze(0).permute(1, 2, 0).repeat(1, 1, 3).numpy()
    ref_mask_3c = 255 * ref_mask_3c.astype(np.uint8)
    gt_mask_flat = batch['support_masks'].squeeze().flatten()[None, ...].to(device)

    # Freeze SAM parameters
    for _, p in sam.named_parameters():
        p.requires_grad = False
    predictor = SamPredictor(sam)

    # Support image encoding and target feature (baseline path)
    ref_mask_pad = predictor.set_image(ref_image, ref_mask_3c)
    ref_feat_hwC = predictor.features.squeeze().permute(1, 2, 0)   # h x w x C
    ref_mask_pad = F.interpolate(ref_mask_pad, size=ref_feat_hwC.shape[0:2], mode="bilinear").squeeze()[0]

    h, w, C = ref_feat_hwC.shape
    target_feat_pixels = ref_feat_hwC[ref_mask_pad > 0] if ref_feat_hwC[ref_mask_pad > 0].any().item() else ref_feat_hwC.reshape(h * w, C)
    target_feat_mean = target_feat_pixels.mean(0)
    target_feat_max  = torch.max(target_feat_pixels, dim=0)[0]
    target_feat = (target_feat_max / 2 + target_feat_mean / 2).unsqueeze(0)
    target_feat = target_feat / (target_feat.norm(dim=-1, keepdim=True) + 1e-8)  # 1 x C

    # Support object representation via pixel-masked re-encoding (for IL selection only)
    support_mask_bin = batch['support_masks'].squeeze().detach().cpu().numpy().astype(np.uint8)  # HxW
    with torch.no_grad():
        support_vec, support_dist = _encode_masked_image_vector(predictor, ref_image, support_mask_bin)

    # PerSAM-F mask weight fine-tuning (baseline)
    sim = _compute_sim_from_features(predictor, target_feat, predictor.features)
    topk_xy, topk_label = point_selection(sim, device, num_extra, topk=1)

    mask_weights = Mask_Weights().to(device)
    mask_weights.train()
    optimizer = torch.optim.AdamW(mask_weights.parameters(), lr=args.lr, eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.train_epoch)

    for _ in range(args.train_epoch):
        masks, scores, logits, logits_high = predictor.predict(
            point_coords=topk_xy,
            point_labels=topk_label,
            multimask_output=True
        )
        logits_high_flat = logits_high.flatten(1)
        weights = torch.cat((1 - mask_weights.weights.sum(0).unsqueeze(0), mask_weights.weights), dim=0)
        logits_high_w = logits_high_flat * weights
        logits_high_w = logits_high_w.sum(0).unsqueeze(0)

        dice_loss  = calculate_dice_loss(logits_high_w, gt_mask_flat)
        focal_loss = calculate_sigmoid_focal_loss(logits_high_w, gt_mask_flat)
        loss = dice_loss + focal_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    mask_weights.eval()
    weights = torch.cat((1 - mask_weights.weights.sum(0).unsqueeze(0), mask_weights.weights), dim=0)
    weights_np = weights.detach()

    # First prediction on query image (baseline)
    test_image = batch['query_img'].squeeze().permute(1, 2, 0).numpy()
    test_image = (test_image * 255).astype(np.uint8)  # HxWx3 uint8

    predictor.set_image(test_image)
    sim = _compute_sim_from_features(predictor, target_feat, predictor.features)
    topk_xy, topk_label = point_selection(sim, device, num_extra, topk=1)

    # Refinement loop
    grad_map = torch.zeros_like(predictor.features)
    miou_list = []
    best_miou = -np.inf
    best_iteration_oracle = 0

    # IL selection state
    best_scores = [-1e9 for _ in range(K)]
    best_iters  = [0    for _ in range(K)]

    use_kl = (args.f.lower() == 'kl')

    for t in range(T):
        # (1) no-grad: initial mask/box
        with torch.no_grad():
            masks_np, scores_np, logits_t, high_logits_t = predictor.predict(
                point_coords=topk_xy,
                point_labels=topk_label,
                multimask_output=True
            )
            high_w     = high_logits_t * weights.unsqueeze(-1)
            logit_high = high_w.sum(0)  # HxW pre-sigmoid
            mask = (logit_high > 0).detach().cpu().numpy()

            logits_w = logits_t * weights_np[..., None]
            logit    = logits_w.sum(0)  # 256x256

            # bounding box
            try:
                y, x = np.nonzero(mask)
                x_min, x_max, y_min, y_max = x.min(), x.max(), y.min(), y.max()
                input_box = np.array([x_min, y_min, x_max, y_max])
            except:
                input_box = None

            masks_np2, scores_np2, logits_t2, _ = predictor.predict(
                point_coords=topk_xy,
                point_labels=topk_label,
                box=input_box[None, :] if input_box is not None else None,
                mask_input=logit[None, :, :],
                multimask_output=True
            )
            best_idx = int(np.argmax(scores_np2))

        # (2) Gradient path: features -> logits_t3 (baseline)
        predictor.features = predictor.features.detach().requires_grad_(True)
        masks_np3, scores_np3, logits_t3, high_logits_t3 = predictor.predict(
            point_coords=topk_xy,
            point_labels=topk_label,
            box=input_box[None, :] if input_box is not None else None,
            mask_input=logits_t2[best_idx:best_idx + 1, :, :],
            multimask_output=True
        )
        best_idx = int(np.argmax(scores_np3))

        gradient_feature = torch.autograd.grad(
            outputs=logits_t3[best_idx: best_idx + 1, :, :],
            inputs=predictor.features,
            grad_outputs=torch.ones_like(logits_t3[best_idx: best_idx + 1, :, :]),
            create_graph=False, retain_graph=False
        )[0]

        # (3) Update grad_map and recompute similarity
        with torch.no_grad():
            v = torch.clamp(gradient_feature, min=-0.1, max=0.1)
            grad_map = grad_map + eta * v + np.sqrt(2 * eta) * noise_factor * torch.randn_like(predictor.features)

            test_feat = predictor.features.squeeze(0) + grad_map.squeeze(0)
            C, h, w = test_feat.shape
            tf = test_feat / (test_feat.norm(dim=0, keepdim=True) + 1e-8)
            tf = tf.reshape(C, h * w)
            sim = target_feat @ tf
            sim = sim.reshape(1, 1, h, w)
            sim = F.interpolate(sim, scale_factor=4, mode="bilinear")
            sim = predictor.model.postprocess_masks(
                sim, predictor.input_size, predictor.original_size
            ).squeeze()

        # (4) Resample prompts
        topk_xy, topk_label = point_selection(sim, device, num_extra, topk=1)

        # (5) Evaluate mIoU (uses GT)
        try:
            area_intersection, area_union = Evaluator.classify_prediction(
                torch.tensor(masks_np3[best_idx], dtype=torch.float32).unsqueeze(0), batch
            )
            miou = (area_intersection.float() / (area_union.float() + 1e-10)).mean().item()
        except:
            miou = 0.0

        miou_list.append(miou)
        if miou > best_miou:
            best_miou = miou
            best_iteration_oracle = t

        # (6) IL selection score update via pixel-masked re-encoding (no GT)
        with torch.no_grad():
            curr_mask = (logit_high > 0).detach().cpu().numpy().astype(np.uint8)
            q_vec, q_dist = _encode_masked_image_vector(predictor, test_image, curr_mask)

            sam_score = float(scores_np3[best_idx])
            logit3_high = (high_logits_t3 * weights.unsqueeze(-1)).sum(0)  # HxW pre-sigmoid
            margin = _mask_margin_from_logit(logit3_high, curr_mask.astype(np.float32))

            if use_kl:
                base = float(torch.sum(support_vec * q_vec).item())
            else:
                base = float(torch.sum(support_vec * q_vec).item())  # cosine

            for k, (alpha, beta) in enumerate(grid):
                sel_score = base + alpha * float(sam_score) + beta * float(margin)
                if sel_score > best_scores[k]:
                    best_scores[k] = sel_score
                    best_iters[k]  = t

        predictor.features = predictor.features.detach().requires_grad_(True)

    return miou_list, None, None, best_iteration_oracle, best_iters[0], best_iters


# =========================================
# Entrypoint
# =========================================
if __name__ == '__main__':
    main()
