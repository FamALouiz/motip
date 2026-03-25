"""Benchmark script to compare predicted peak memory usage against actual peak memory."""

import argparse
import csv
import os
import random
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import cotengra as ctg
import matplotlib.pyplot as plt
import psutil


def get_memory_system_command(pid: int) -> int:
    """Get memory usage of a process using system commands. Returns memory in bytes."""
    try:
        process = psutil.Process(pid)
        return process.memory_info().rss
    except Exception as e:
        print(f"Error getting memory usage: {e}")
    return 0


class SystemMemoryPoller:
    """System memory poller."""

    def __init__(self, pid: Optional[int] = None, interval: float = 0.01) -> None:
        """Initialize the memory poller."""
        self.pid = pid or os.getpid()
        print(f"Monitoring memory for PID: {self.pid}")
        self.interval = interval
        self.peak_memory = 0
        self._stop = False
        self._thread: Optional[threading.Thread] = None

    def poll(self) -> None:
        """Poll memory usage at regular intervals and track peak memory."""
        while not self._stop:
            mem = get_memory_system_command(self.pid)
            self.peak_memory = max(self.peak_memory, mem)
            time.sleep(self.interval)

    def start(self) -> None:
        """Start the memory polling thread."""
        self._stop = False
        self.peak_memory = get_memory_system_command(self.pid)
        self._thread = threading.Thread(target=self.poll, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the memory polling thread."""
        self._stop = True
        if self._thread:
            self._thread.join()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-num-tensors", type=int, default=4)
    parser.add_argument("--max-num-tensors", type=int, default=10)
    parser.add_argument("--runs-per-size", type=int, default=10)
    parser.add_argument("--average-rank", type=int, default=3)
    parser.add_argument("--max-dim", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dir-name", type=str, default="peak_memory_prediction_comparison")
    parser.add_argument("--poll-interval", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    """Run the benchmark and save results."""
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"

    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from contraction.tensor_network import contract_network
    from memory.calculator import MemoryCalculator
    from tensor_network.utils.random import generate_random_tn

    args = parse_args()
    if args.initial_num_tensors <= 0 or args.max_num_tensors <= 0:
        raise ValueError()
    if args.initial_num_tensors > args.max_num_tensors:
        raise ValueError()
    if args.runs_per_size <= 0:
        raise ValueError()

    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    calculator = MemoryCalculator()
    rows = []
    rng = random.Random(args.seed)

    poller = SystemMemoryPoller(interval=args.poll_interval)

    for num_tensors in range(args.initial_num_tensors, args.max_num_tensors + 1):
        predicted_values = []
        actual_values = []
        differences = []

        for run_idx in range(args.runs_per_size):
            tn = generate_random_tn(
                num_tensors=num_tensors,
                average_rank=args.average_rank,
                max_dim=args.max_dim,
                seed=rng.randint(0, 10**9) + run_idx,
                generate_arrays=True,
                num_output_indices=rng.randint(0, max(1, num_tensors - 2)),
            )

            actual_initial_memory = sum(sys.getsizeof(tensor) for tensor in tn.arrays)

            try:
                contraction_tree = ctg.array_contract_tree(
                    inputs=tn.input_indices,
                    output=tn.output_indices,
                    size_dict=tn.size_dict,
                    shapes=tn.shapes,
                )
                path = contraction_tree.get_path()
            except Exception:
                continue

            predicted_peak_memory = calculator.calculate_peak_memory(tn, path).bytes

            base_mem = get_memory_system_command(poller.pid)

            poller.start()
            _ = contract_network(tn, path)
            poller.stop()

            tracked_peak_mem = poller.peak_memory
            increase = max(0, tracked_peak_mem - base_mem)
            actual_memory = actual_initial_memory + increase

            difference = actual_memory - predicted_peak_memory

            predicted_values.append(predicted_peak_memory)
            actual_values.append(actual_memory)
            differences.append(difference)

        if not predicted_values:
            continue

        avg_predicted = sum(predicted_values) / len(predicted_values)
        avg_actual = sum(actual_values) / len(actual_values)
        avg_difference = sum(differences) / len(differences)
        avg_abs_diff = sum(abs(v) for v in differences) / len(differences)

        rows.append(
            {
                "num_tensors": num_tensors,
                "runs": len(predicted_values),
                "avg_predicted_peak_bytes": avg_predicted,
                "avg_actual_peak_bytes": avg_actual,
                "avg_difference_bytes": avg_difference,
                "avg_absolute_difference_bytes": avg_abs_diff,
            }
        )
        print(
            f"Number of tensors: {num_tensors}, Average predicted memory: {avg_predicted}, "
            f"Average actual memory: {avg_actual}, Average difference {avg_difference}"
        )

    if not rows:
        return

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
    avg_predicted_values = [float(row["avg_predicted_peak_bytes"]) for row in rows]
    avg_actual_values = [float(row["avg_actual_peak_bytes"]) for row in rows]
    avg_differences = [float(row["avg_difference_bytes"]) for row in rows]

    _, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(
        tensor_numbers, avg_predicted_values, marker="o", label="Avg predicted peak memory"
    )
    axes[0].plot(tensor_numbers, avg_actual_values, marker="x", label="Avg actual peak memory")
    axes[0].set_ylabel("Memory (bytes)")
    axes[0].set_title("Average Predicted vs Actual Peak Memory by Tensor Count")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        tensor_numbers, avg_differences, marker="s", color="tab:orange", label="Avg difference"
    )
    axes[1].axhline(0, color="black", linewidth=1, linestyle="--")
    axes[1].set_xlabel("Number of tensors")
    axes[1].set_ylabel("Avg(actual - predicted) (bytes)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(graph_path, dpi=150)
    plt.close()

    print(f"{global_average_difference} {global_average_absolute_difference}")


if __name__ == "__main__":
    main()
