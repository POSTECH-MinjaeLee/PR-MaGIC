"""
PR-MaGIC: Matcher-PR evaluation launcher.

Usage (from PR-MaGIC root):
    # Single benchmark
    python scripts/launch_matcher_pr.py --benchmarks coco

    # Multiple benchmarks
    python scripts/launch_matcher_pr.py --benchmarks coco lvis fss

    # All benchmarks (default)
    python scripts/launch_matcher_pr.py

    # Specify GPUs
    python scripts/launch_matcher_pr.py --benchmarks coco --gpus 0,1,2,3
"""

import subprocess, time, os, sys, argparse
from threading import Thread, Lock

REPO_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATCHER_DIR  = os.path.join(REPO_ROOT, "Matcher")
SEEDS        = [42, 43, 44, 45, 46]

# Pre-defined per-benchmark settings: eta, nshots, folds, alpha, beta, exp, merging_mask
BENCHMARK_CONFIGS = {
    'coco':         {'eta': 0.001,  'nshots': [1, 5], 'folds': [0,1,2,3],      'alpha': 1.0, 'beta': 0.0, 'exp': 0.0, 'merging': 9,  'num_centers': None},
    'fss':          {'eta': 0.001,  'nshots': [1, 5], 'folds': [0],            'alpha': 0.8, 'beta': 0.2, 'exp': 1.0, 'merging': 10, 'num_centers': None},
    'lvis':         {'eta': 0.001,  'nshots': [1, 5], 'folds': list(range(10)),'alpha': 1.0, 'beta': 0.0, 'exp': 0.0, 'merging': 9,  'num_centers': None},
    'paco_part':    {'eta': 0.0001, 'nshots': [1, 5], 'folds': [0,1,2,3],      'alpha': 0.5, 'beta': 0.5, 'exp': 0.0, 'merging': 5,  'num_centers': 5},
    'pascal_part':  {'eta': 0.0001, 'nshots': [1, 5], 'folds': [0,1,2,3],      'alpha': 0.5, 'beta': 0.5, 'exp': 0.0, 'merging': 5,  'num_centers': 5},
    'dis':          {'eta': 0.0001, 'nshots': [1],    'folds': [0],            'alpha': 0.8, 'beta': 0.2, 'exp': 1.0, 'merging': 10, 'num_centers': 5},
}

def build_tasks(benchmarks):
    tasks = []
    for bm in benchmarks:
        cfg = BENCHMARK_CONFIGS[bm]
        for fold in cfg['folds']:
            for nshot in cfg['nshots']:
                for seed in SEEDS:
                    tasks.append({**cfg, 'benchmark': bm, 'fold': fold,
                                  'nshot': nshot, 'seed': seed})
    return tasks

def run_experiment(task, gpu_id, lock):
    bm, fold, nshot, seed = task['benchmark'], task['fold'], task['nshot'], task['seed']
    log_dir = os.path.join(REPO_ROOT, "results", "matcher", bm)
    os.makedirs(log_dir, exist_ok=True)

    cmd = [
        sys.executable, "pr_magic_for_matcher.py",
        "--benchmark",           bm,
        "--nshot",               str(nshot),
        "--fold",                str(fold),
        "--seed",                str(seed),
        "--eta",                 str(task['eta']),
        "--gamma",               "0.1",
        "--alpha",               str(task['alpha']),
        "--beta",                str(task['beta']),
        "--exp",                 str(task['exp']),
        "--num_merging_mask",    str(task['merging']),
        "--max_sample_iterations","30",
        "--nested",              "6",
        "--alpha_list",          "0.0",
        "--beta_list",           "0.0",
        "--use_score_filter",
        "--box_nms_thresh",      "0.65",
        "--sample-range",        "(4,6)",
        "--multimask_output",    "0",
        "--log-root",            os.path.join(REPO_ROOT, "results", "matcher", bm, f"f{fold}_n{nshot}_s{seed}"),
    ]
    if task['num_centers'] is not None:
        cmd += ["--num_centers", str(task['num_centers'])]

    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu_id))
    print(f"[GPU {gpu_id}] Start : matcher-pr  {bm}  fold={fold}  nshot={nshot}  seed={seed}")
    with open(os.path.join(log_dir, f"f{fold}_n{nshot}_s{seed}.out"), "w") as f:
        subprocess.run(cmd, env=env, cwd=MATCHER_DIR, stdout=f, stderr=subprocess.STDOUT)
    print(f"[GPU {gpu_id}] Done  : matcher-pr  {bm}  fold={fold}  nshot={nshot}  seed={seed}")

def worker(gpu_id, queue, lock):
    while True:
        with lock:
            if not queue: return
            task = queue.pop(0)
        try:
            run_experiment(task, gpu_id, lock)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks", nargs="+",
                        choices=list(BENCHMARK_CONFIGS), default=list(BENCHMARK_CONFIGS),
                        help="Benchmarks to evaluate (default: all)")
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5",
                        help="Comma-separated GPU IDs")
    args = parser.parse_args()

    gpus  = [int(g) for g in args.gpus.split(",")]
    lock  = Lock()
    queue = build_tasks(args.benchmarks)
    print(f"Benchmarks : {args.benchmarks}")
    print(f"Total tasks: {len(queue)}  |  GPUs: {gpus}")

    threads = [Thread(target=worker, args=(g, queue, lock), daemon=True) for g in gpus]
    for t in threads:
        t.start()
        time.sleep(1)

    for t in threads:
        try:
            while t.is_alive():
                t.join(timeout=1.0)
        except KeyboardInterrupt:
            print("\nStopped."); sys.exit(1)

    print("All Matcher-PR experiments completed.")
