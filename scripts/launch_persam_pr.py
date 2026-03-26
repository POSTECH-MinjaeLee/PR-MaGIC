"""
PR-MaGIC: PerSAM-PR evaluation launcher.

Usage (from PR-MaGIC root):
    # Single benchmark
    python scripts/launch_persam_pr.py --benchmarks coco

    # Multiple benchmarks
    python scripts/launch_persam_pr.py --benchmarks coco lvis fss

    # All benchmarks (default)
    python scripts/launch_persam_pr.py

    # Specify GPUs
    python scripts/launch_persam_pr.py --benchmarks coco --gpus 0,1,2,3
"""

import subprocess, time, os, sys, argparse
from threading import Thread

REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERSAM_DIR = os.path.join(REPO_ROOT, "Personalize-SAM")
SEEDS      = [42, 43, 44, 45, 46]

# Pre-defined per-benchmark settings: eta, points_num, folds
BENCHMARK_CONFIGS = {
    'coco':         {'eta': '1e-3', 'points': 5, 'folds': [0, 1, 2, 3]},
    'fss':          {'eta': '1e-3', 'points': 5, 'folds': [0]},
    'lvis':         {'eta': '1e-3', 'points': 5, 'folds': list(range(10))},
    'paco_part':    {'eta': '1e-4', 'points': 3, 'folds': [0, 1, 2, 3]},
    'pascal_part':  {'eta': '1e-4', 'points': 3, 'folds': [0, 1, 2, 3]},
    'dis':          {'eta': '1e-4', 'points': 3, 'folds': [0]},
}

def build_tasks(benchmarks):
    tasks = []
    for bm in benchmarks:
        cfg = BENCHMARK_CONFIGS[bm]
        for fold in cfg['folds']:
            for seed in SEEDS:
                tasks.append({**cfg, 'benchmark': bm, 'fold': fold, 'seed': seed})
    return tasks

def run_experiment(task, gpu_id):
    bm, fold, seed = task['benchmark'], task['fold'], task['seed']
    out_dir = os.path.join(REPO_ROOT, "results", "persam", bm)
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        sys.executable, "pr_magic_for_persam.py",
        "--benchmark",  bm,
        "--fold",       str(fold),
        "--seed",       str(seed),
        "--eta",        task['eta'],
        "--gamma",      "1e-1",
        "--points_num", str(task['points']),
        "--nested",     "6",
        "--alpha_list", "0.0",
        "--beta_list",  "0.0",
        "--log-root",   os.path.join(out_dir, f"f{fold}_s{seed}"),
    ]

    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu_id))
    print(f"[GPU {gpu_id}] Start : persam-pr  {bm}  fold={fold}  seed={seed}")
    with open(os.path.join(out_dir, f"f{fold}_s{seed}.out"), "w") as f:
        subprocess.run(cmd, env=env, cwd=PERSAM_DIR, stdout=f, stderr=subprocess.STDOUT)
    print(f"[GPU {gpu_id}] Done  : persam-pr  {bm}  fold={fold}  seed={seed}")

def worker(gpu_id, queue):
    while queue:
        try:
            run_experiment(queue.pop(0), gpu_id)
        except IndexError:
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks", nargs="+",
                        choices=list(BENCHMARK_CONFIGS), default=list(BENCHMARK_CONFIGS),
                        help="Benchmarks to evaluate (default: all)")
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5",
                        help="Comma-separated GPU IDs")
    args = parser.parse_args()

    gpus  = [int(g) for g in args.gpus.split(",")]
    queue = build_tasks(args.benchmarks)
    print(f"Benchmarks : {args.benchmarks}")
    print(f"Total tasks: {len(queue)}  |  GPUs: {gpus}")

    threads = [Thread(target=worker, args=(g, queue), daemon=True) for g in gpus]
    for t in threads:
        t.start()
        time.sleep(2)

    for t in threads:
        try:
            while t.is_alive():
                t.join(timeout=1.0)
        except KeyboardInterrupt:
            print("\nStopped."); sys.exit(1)

    print("All PerSAM-PR experiments completed.")
