"""Benchmark predicted vs actual initial tensor-network memory."""

import argparse
import csv
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Sweep tensor-count range and compare average predicted and actual initial memory "
            "across repeated runs."
        )
    )
    parser.add_argument(
        "--initial-num-tensors", type=int, default=10, help="Initial tensor count in sweep."
    )
    parser.add_argument(
        "--max-num-tensors", type=int, default=30, help="Maximum tensor count in sweep."
    )
    parser.add_argument(
        "--runs-per-size",
        type=int,
        default=30,
        help="Number of random networks to generate for each tensor count.",
    )
    parser.add_argument(
        "--average-rank", type=int, default=4, help="Average tensor rank used in generation."
    )
    parser.add_argument(
        "--max-dim", type=int, default=4, help="Maximum dimension size for any index."
    )
    parser.add_argument("--seed", type=int, default=1, help="Base random seed.")
    parser.add_argument(
        "--dir-name",
        type=str,
        default="initial_memory_prediction_comparison",
        help="Output directory name saved under ./data/.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the benchmark and save results."""
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from memory.calculator import MemoryCalculator
    from tensor_network.utils.random import generate_random_tn

    args = parse_args()
    if args.initial_num_tensors <= 0:
        raise ValueError("--initial-num-tensors must be a positive integer.")
    if args.max_num_tensors <= 0:
        raise ValueError("--max-num-tensors must be a positive integer.")
    if args.initial_num_tensors > args.max_num_tensors:
        raise ValueError("--initial-num-tensors must be less than or equal to --max-num-tensors.")
    if args.runs_per_size <= 0:
        raise ValueError("--runs-per-tensor must be a positive integer.")

    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    calculator = MemoryCalculator()
    rows: list[dict[str, float | int]] = []
    rng = random.Random(args.seed)

    for num_tensors in range(args.initial_num_tensors, args.max_num_tensors + 1):
        predicted_values: list[int] = []
        actual_values: list[int] = []
        differences: list[int] = []

        for run_idx in range(args.runs_per_size):
            tn = generate_random_tn(
                num_tensors=num_tensors,
                average_rank=args.average_rank,
                max_dim=args.max_dim,
                seed=rng.randint(0, 10**9) + run_idx,
                generate_arrays=True,
            )

            predicted_initial_memory = calculator.calculate_total_memory(tn, []).to_bytes
            actual_memory = sum(sys.getsizeof(tensor) for tensor in tn.arrays)
            difference = actual_memory - predicted_initial_memory

            predicted_values.append(predicted_initial_memory)
            actual_values.append(actual_memory)
            differences.append(difference)

        avg_predicted = sum(predicted_values) / len(predicted_values)
        avg_actual = sum(actual_values) / len(actual_values)
        avg_difference = sum(differences) / len(differences)
        avg_absolute_difference = sum(abs(value) for value in differences) / len(differences)

        rows.append(
            {
                "num_tensors": num_tensors,
                "runs": args.runs_per_size,
                "avg_predicted_initial_bytes": avg_predicted,
                "avg_actual_bytes": avg_actual,
                "avg_difference_bytes": avg_difference,
                "avg_absolute_difference_bytes": avg_absolute_difference,
            }
        )

        print(
            f"Tensors = {num_tensors}: Avg predicted = {avg_predicted:.2f} bytes, "
            f"Avg actual = {avg_actual:.2f} bytes, "
            f"Avg difference = {avg_difference:.2f} bytes"
        )

    global_average_difference = sum(float(row["avg_difference_bytes"]) for row in rows) / len(rows)
    global_average_absolute_difference = sum(
        float(row["avg_absolute_difference_bytes"]) for row in rows
    ) / len(rows)

    dir_path = data_dir / args.dir_name
    dir_path.mkdir(parents=True, exist_ok=True)
    graph_path = dir_path / "graph.png"
    csv_path = dir_path / "data.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    tensor_numbers = [int(row["num_tensors"]) for row in rows]
    avg_predicted_values = [float(row["avg_predicted_initial_bytes"]) for row in rows]
    avg_actual_values = [float(row["avg_actual_bytes"]) for row in rows]
    avg_differences = [float(row["avg_difference_bytes"]) for row in rows]

    _, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(tensor_numbers, avg_predicted_values, marker="o", label="Avg predicted memory")
    axes[0].plot(tensor_numbers, avg_actual_values, marker="x", label="Avg actual memory")
    axes[0].set_ylabel("Memory (bytes)")
    axes[0].set_title("Average Predicted vs Actual Initial Memory by Tensor Count")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        tensor_numbers,
        avg_differences,
        marker="s",
        color="tab:orange",
        label="Avg difference",
    )
    axes[1].axhline(0, color="black", linewidth=1, linestyle="--")
    axes[1].set_xlabel("Number of tensors")
    axes[1].set_ylabel("Avg(actual - predicted) (bytes)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(graph_path, dpi=150)
    plt.close()

    print(f"Saved plot: {graph_path}")
    print(f"Saved csv: {csv_path}")
    print(f"Global average difference (actual - predicted): {global_average_difference:.2f} bytes")
    print(f"Global average absolute difference: {global_average_absolute_difference:.2f} bytes")


if __name__ == "__main__":
    main()
