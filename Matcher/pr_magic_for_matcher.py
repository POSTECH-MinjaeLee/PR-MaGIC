r""" Matcher testing code for one-shot segmentation (with IL-selection & PerSAM CSV) """
import argparse
import os
import sys
from datetime import datetime
import warnings

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append('./')

from matcher.common.logger import Logger, AverageMeter
from matcher.common.evaluation import Evaluator
from matcher.common import utils
from matcher.data.dataset import FSSDataset
from matcher.Matcher_PR import build_matcher_oss


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


def test(matcher, dataloader, logger, args=None):
    r""" Test Matcher with per-iter IL-selection scoring + PerSAM CSV payload """
    utils.fix_randseed(args.seed)

    # Per-step cumulative statistics
    average_meter_list = [AverageMeter(dataloader.dataset) for _ in range(args.nested)]

    # PerSAM-style aggregation containers
    total_mious = {}
    baseline0_mious = []
    mean_after0_list = []
    oracle_after0_list = []

    # IL grid (same logic as PerSAM)
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

    logger.info("\n[IL4-* index mapping]")
    for i, (a, b) in enumerate(grid, start=1):
        logger.info(f"  IL4-{i}: alpha={a}, beta={b}")

    p_bar = tqdm(dataloader, total=len(dataloader))

    for idx, batch in enumerate(p_bar):
        batch = utils.to_cuda(batch, args.device)
        query_img, query_mask, support_imgs, support_masks = \
            batch['query_img'], batch['query_mask'], batch['support_imgs'], batch['support_masks']

        class_id = batch['class_id'].item() if isinstance(batch['class_id'], torch.Tensor) else int(batch['class_id'])
        obj_name = f'{idx}_class-{class_id}'

        # Set reference and target
        matcher.set_reference(support_imgs, support_masks)
        matcher.set_target(query_img)

        miou_per_iter = []
        metas_per_iter = []

        # Nested inference
        for t in range(args.nested):
            pred = matcher.predict(t, idx, args.benchmark, args.fold, args)
            if isinstance(pred, tuple):
                pred_mask, meta = pred
            else:
                pred_mask, meta = pred, {}

            assert pred_mask.size() == batch['query_mask'].size(), \
                f'pred {pred_mask.size()} ori {batch["query_mask"].size()}'

            # Evaluate (mIoU)
            area_inter, area_union = Evaluator.classify_prediction(pred_mask.clone(), batch)
            average_meter_list[t].update(area_inter, area_union, batch['class_id'], loss=None)

            miou_t = (area_inter.float() / (area_union.float().clamp_min(1e-10))).mean().item()
            miou_per_iter.append(miou_t)

            # Record meta for IL selection
            base      = float(meta.get('base', 0.0))
            sam_score = float(meta.get('sam_score', 0.0))
            margin    = float(meta.get('margin', 0.0))
            metas_per_iter.append((base, sam_score, margin))

            average_meter_list[t].write_process(idx, len(dataloader), epoch=-1, write_batch_idx=100, nested=t + 1)

        # Per-sample aggregation and IL grid selection
        total_mious[obj_name] = miou_per_iter

        if len(miou_per_iter) >= 1:
            baseline0_mious.append(miou_per_iter[0])
        if len(miou_per_iter) > 1:
            mean_after0_list.append(sum(miou_per_iter[1:]) / (len(miou_per_iter) - 1))
            oracle_after0_list.append(max(miou_per_iter))

        # Best iteration per grid point
        best_iters_this_sample = []
        for k, (alpha, beta) in enumerate(grid):
            best_score = -1e9
            best_iter = 0
            for t in range(len(miou_per_iter)):
                base, sam_score, margin = metas_per_iter[t]
                sel_score = base + alpha * sam_score + beta * margin
                if sel_score > best_score:
                    best_score = sel_score
                    best_iter = t
            best_iters_this_sample.append(best_iter)
            il_mious_grid[k].append(miou_per_iter[best_iter])
            il_iters_grid[k].append(best_iter)

        per_sample_best_iters[obj_name] = best_iters_this_sample

        matcher.clear()

        # Periodic progress report
        if idx % 100 == 0:
            def _safe_mean(xs):
                return float(sum(xs) / len(xs)) if xs else 0.0
            logger.info(f"\n[{idx}/{len(dataloader)}]")
            logger.info(f"  (1) Baseline@0    : {_safe_mean(baseline0_mious):.4f}")
            logger.info(f"  (2) Mean(1..T)   : {_safe_mean(mean_after0_list):.4f}")
            logger.info(f"  (3) Oracle(0..T) : {_safe_mean(oracle_after0_list):.4f}")
            for i in range(K):
                logger.info(f"  (4) IL4-{i+1}     : {_safe_mean(il_mious_grid[i]):.4f} | Avg iter {_safe_mean(il_iters_grid[i]):.2f}")

    # Per-step result log
    for t in range(args.nested):
        average_meter_list[t].write_result('Test', 0)

    result_list = [average_meter_list[i].compute_iou() for i in range(args.nested)]
    csv_payload = dict(
        total_mious=total_mious,
        per_sample_best_iters=per_sample_best_iters,
        grid=grid,
        il_mious_grid=il_mious_grid,
        il_iters_grid=il_iters_grid,
        baseline0_mious=baseline0_mious,
        mean_after0_list=mean_after0_list,
        oracle_after0_list=oracle_after0_list,
    )
    return result_list, csv_payload


if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='Matcher Pytorch Implementation for One-shot Segmentation (with IL-selection & PerSAM CSV)')

    # Dataset parameters
    parser.add_argument('--datapath', type=str, default='../data')
    parser.add_argument('--benchmark', type=str, default='coco',
                        choices=['fss', 'coco', 'pascal', 'lvis', 'paco_part', 'pascal_part', 'dis'])
    parser.add_argument('--bsz', type=int, default=1)
    parser.add_argument('--nworker', type=int, default=0)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--nshot', type=int, default=1)
    parser.add_argument('--img-size', type=int, default=518)
    parser.add_argument('--use_original_imgsize', action='store_true')
    parser.add_argument('--log-root', type=str, default='',
                        help='Output directory (overrides auto-generated path if set)')
    parser.add_argument('--eta', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=1e-1)
    parser.add_argument('--seed', type=int, default=42)

    # DINOv2 / SAM
    parser.add_argument('--dinov2-size', type=str, default="vit_large")
    parser.add_argument('--sam-size', type=str, default="vit_h")
    parser.add_argument('--dinov2-weights', type=str, default="../weights/dinov2_vitl14_pretrain.pth")
    parser.add_argument('--sam-weights', type=str, default="../weights/sam_vit_h_4b8939.pth")

    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--targets', metavar='N', type=int, nargs='+', help='a list of integers')
    parser.add_argument('--points_per_side', type=int, default=64)
    parser.add_argument('--pred_iou_thresh', type=float, default=0.88)
    parser.add_argument('--sel_stability_score_thresh', type=float, default=0.0)
    parser.add_argument('--stability_score_thresh', type=float, default=0.95)
    parser.add_argument('--iou_filter', type=float, default=0.0)
    parser.add_argument('--box_nms_thresh', type=float, default=1.0)
    parser.add_argument('--output_layer', type=int, default=3)
    parser.add_argument('--dense_multimask_output', type=int, default=0)
    parser.add_argument('--use_dense_mask', type=int, default=0)
    parser.add_argument('--multimask_output', type=int, default=0)

    # Matcher parameters
    parser.add_argument('--num_centers', type=int, default=8, help='K centers for kmeans')
    parser.add_argument('--use_box', action='store_true', help='use box as an extra prompt for sam')
    parser.add_argument('--use_points_or_centers', action='store_true', help='points:T, center: F')
    parser.add_argument('--sample-range', type=str, default="(4,6)", help='sample points number range')
    parser.add_argument('--max_sample_iterations', type=int, default=30)
    parser.add_argument('--alpha', type=float, default=1.)
    parser.add_argument('--beta', type=float, default=0.)
    parser.add_argument('--exp', type=float, default=0.)
    parser.add_argument('--emd_filter', type=float, default=0.0, help='use emd_filter')
    parser.add_argument('--purity_filter', type=float, default=0.0, help='use purity_filter')
    parser.add_argument('--coverage_filter', type=float, default=0.0, help='use coverage_filter')
    parser.add_argument('--use_score_filter', action='store_true')
    parser.add_argument('--deep_score_norm_filter', type=float, default=0.1)
    parser.add_argument('--deep_score_filter', type=float, default=0.33)
    parser.add_argument('--topk_scores_threshold', type=float, default=0.7)
    parser.add_argument('--num_merging_mask', type=int, default=10, help='topk masks for merging')
    parser.add_argument('--nested', type=int, default=10)

    # IL-selection options (same grid sweep as PerSAM)
    parser.add_argument('--score_alpha', type=float, default=0.25)  # SAM score weight
    parser.add_argument('--score_beta',  type=float, default=0.10)  # margin weight
    parser.add_argument('--alpha_list',  type=str, default="")
    parser.add_argument('--beta_list',   type=str, default="")
    parser.add_argument('--il-metric',   type=str, default='cos', choices=['cos', 'kl'],
                        help='base term for IL selection (cosine or -KL)')
    parser.add_argument('--il-report',   action='store_true', help='print IL-stop summary at the end')

    # CSV output path (same format as PerSAM)
    parser.add_argument('--outdir', type=str, default='')

    args = parser.parse_args()

    import ast
    args.sample_range = ast.literal_eval(args.sample_range)

    if args.log_root:
        args.outdir = args.log_root
    elif not args.outdir:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.outdir = os.path.join('./save', args.benchmark, f"fold:{args.fold}-{ts}-eta{args.eta}-gamma{args.gamma}")
    os.makedirs(args.outdir, exist_ok=True)

    Logger.initialize(args, root=args.outdir)

    # Device setup
    dev = str(args.device).strip().lower()
    device = dev if (dev == 'cpu' or dev.startswith('cuda')) else f"cuda:{dev}"
    device = device if (torch.cuda.is_available() and device != 'cpu') else 'cpu'
    args.device = device
    Logger.info(f'# using {args.device} available GPUs: {torch.cuda.device_count()}')
    Logger.info(f'Target image : {args.targets}')

    # Model initialization
    matcher = build_matcher_oss(args)

    # Helper classes
    Evaluator.initialize()

    # Dataset
    FSSDataset.initialize(img_size=args.img_size, datapath=args.datapath, use_original_imgsize=args.use_original_imgsize)
    dataloader_test = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'test', args.nshot)

    # Test
    result_list, csv_payload = test(matcher, dataloader_test, Logger, args=args)

    # CSV output (same format as PerSAM)
    csv_dir = os.path.join(args.outdir, 'csvs')
    os.makedirs(csv_dir, exist_ok=True)

    # (A) per-sample mIoU per iteration
    total_mious = csv_payload['total_mious']
    df_miou = pd.DataFrame.from_dict(total_mious, orient='index')
    df_miou.index.name = 'obj_name'
    df_miou.columns = [f'iter_{i}' for i in range(df_miou.shape[1])]
    df_miou.reset_index(inplace=True)
    path_miou = os.path.join(csv_dir, 'miou_per_iter.csv')
    df_miou.to_csv(path_miou, index=False)
    Logger.info(f"Saved: {path_miou}")

    # (B) per-sample best iteration for each (alpha,beta)
    per_sample_best_iters = csv_payload['per_sample_best_iters']
    grid = csv_payload['grid']
    K = len(grid)
    if per_sample_best_iters:
        col_names = [f'IL4_{i+1}_best_iter' for i in range(K)]
        rows = []
        for obj, bests in per_sample_best_iters.items():
            row = {'obj_name': obj}
            for i in range(K):
                row[col_names[i]] = bests[i] if i < len(bests) else -1
            rows.append(row)
        df_best = pd.DataFrame(rows, columns=['obj_name'] + col_names)
        path_best = os.path.join(csv_dir, 'best_iters_grid.csv')
        df_best.to_csv(path_best, index=False)
        print(f"Saved: {path_best}")

    # Final report
    def _safe_mean(xs):
        return float(sum(xs) / len(xs)) if xs else 0.0

    baseline0_mious = csv_payload['baseline0_mious']
    mean_after0_list = csv_payload['mean_after0_list']
    oracle_after0_list = csv_payload['oracle_after0_list']
    il_mious_grid = csv_payload['il_mious_grid']
    il_iters_grid = csv_payload['il_iters_grid']

    Logger.info(f"\n--- Final Results (seed:{args.seed})---")
    Logger.info(f"1) Baseline (no refinement, iter 0): Avg mIoU = {_safe_mean(baseline0_mious):.4f}")
    Logger.info(f"2) Avg of per-sample mean mIoU over iters 1..T: {_safe_mean(mean_after0_list):.4f}")
    Logger.info(f"3) Oracle (best over iters 0..T): Avg mIoU = {_safe_mean(oracle_after0_list):.4f}")
    for i, (a, b) in enumerate(grid):
        miou_avg = _safe_mean(il_mious_grid[i])
        iter_avg = _safe_mean(il_iters_grid[i])
        Logger.info(f"4-{i+1}) IL-stop (alpha={a}, beta={b}): Avg mIoU = {miou_avg:.4f} | Avg stop iter = {iter_avg:.2f} (0-based), {iter_avg+1:.2f} (1-based)")
