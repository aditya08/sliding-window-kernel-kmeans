import argparse
import pandas as pd
import subprocess
import os

def run_experiment(application_path, data_path, n_clusters, block_size, max_iters, tol, seed, max_rows):
    command = [
        application_path,
        "--data", data_path,
        "--n_clusters", str(n_clusters),
        "--block_size", str(block_size),
        "--max_iters", str(max_iters),
        "--tol", str(tol),
        "--seed", str(seed),
        "--max_rows", str(max_rows)
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    print("STDOUT:")
    print(result.stdout)
    return result.stdout

def parse_output(output):
    lines = output.strip().split("\n")
    metrics = {}
    for line in lines:
        if "Fitting completed in" in line:
            parts = line.split()
            metrics['time'] = float(parts[3])
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Ablation Study for Kernel K-Means")
    parser.add_argument("--application_path", type=str, required=True, help="Path to the application to run")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--block_size", type=int, nargs='+', default=[512, 1024, 2048], help="List of values for the block size")
    parser.add_argument("--n_clusters", type=int, nargs='+', default=[16, 32, 128], help="List of values for the number of clusters")
    parser.add_argument("--max_iters", type=int, default=100, help="Maximum number of iterations")
    parser.add_argument("--tol", type=float, default=1e-4, help="Tolerance for convergence")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization")
    parser.add_argument("--max_rows", type=int, default=2**17, help="Maximum number of rows to load from the dataset")
    parser.add_argument("--output_csv", type=str, default="ablation_study_results.csv", help="Output CSV file for results")

    args = parser.parse_args()

    results = []

    for n_clusters in args.n_clusters:
        for block_size in args.block_size:
            print(f"Running experiment with n_clusters={n_clusters}, block_size={block_size}, max_rows={args.max_rows}")
            output = run_experiment(
				application_path=args.application_path,
                data_path=args.data_path,
                n_clusters=n_clusters,
                block_size=block_size,
                max_iters=args.max_iters,
                tol=args.tol,
                seed=args.seed,
                max_rows=args.max_rows
            )
            metrics = parse_output(output)
            metrics.update({
                'n_clusters': n_clusters,
                'block_size': block_size,
                'max_rows': args.max_rows
            })
            results.append(metrics)

    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    print(f"Results saved to {args.output_csv}")

if __name__ == "__main__":
    main()
