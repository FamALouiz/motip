"""Benchmark script for greedy permutation strategy against baseline contraction.

This script sweeps over tensor network configurations and k values for the greedy algorithm,
measuring peak and total memory usage for both greedy-optimized and baseline contractions.
It outputs a CSV with all measurements and generates graphs comparing memory efficiency.
"""

from __future__ import annotations

import argparse
import csv
import sys
import threading
import time
import tracemalloc
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import cotengra as ctg
import matplotlib.pyplot as plt
import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "greedy_strategy_benchmark"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sweep_script import AbstractSweepScript  # noqa: E402

from operations.contraction.path import ContractionPath  # noqa: E402
from operations.strategy.greedy import GreedyPermutationStrategy  # noqa: E402
from operations.strategy.local_optimal import LocalOptimalPermutationStrategy  # noqa: E402
from tensor_network.tn import TensorNetwork  # noqa: E402
from tensor_network.utils.random import generate_random_tn  # noqa: E402

Row = dict[str, int | float | str]

FIELDNAMES = [
    "size",
    "seed",
    "average_rank",
    "max_dim",
    "num_output_indices",
    "k",
    "baseline_peak_bytes",
    "baseline_total_bytes",
    "greedy_peak_bytes",
    "greedy_total_bytes",
    "peak_improvement_pct",
    "total_improvement_pct",
]


@dataclass(frozen=True)
class GreedyBenchmarkWorkItem:
    """Independent work item for the greedy strategy benchmark."""

    size: int
    seed: int
    average_rank: int
    max_dim: int
    num_output_indices: int
    k: int


@dataclass
class MemorySnapshot:
    """Snapshot of memory metrics at a point in time."""

    peak_bytes: int
    total_bytes: int


@dataclass
class GreedyBenchmarkResult:
    """Result of benchmarking one work item."""

    row: Row
    baseline_peak: int
    baseline_total: int
    greedy_peak: int
    greedy_total: int


@dataclass
class GreedyBenchmarkAggregate:
    """Aggregate state for the benchmark sweep."""

    rows: list[Row] = field(default_factory=list)


class MemoryMonitor:
    """Monitor memory usage during tensor contractions."""

    def __init__(self) -> None:
        """Initialize the memory monitor."""
        self.process = psutil.Process()
        self.peak_rss = 0
        self.initial_rss = 0
        self.total_allocated = 0

    def start(self) -> None:
        """Start monitoring memory."""
        self.process.memory_info()
        self.initial_rss = self.process.memory_info().rss
        self.peak_rss = self.initial_rss
        self.total_allocated = 0
        tracemalloc.start()

    def stop(self) -> MemorySnapshot:
        """Stop monitoring and return memory metrics."""
        tracemalloc.stop()
        current_rss = self.process.memory_info().rss
        peak_rss = self.peak_rss

        if current_rss > peak_rss:
            peak_rss = current_rss

        current, peak = tracemalloc.get_traced_memory()
        total_allocated = peak

        return MemorySnapshot(peak_bytes=peak_rss, total_bytes=total_allocated)

    def update(self) -> None:
        """Update peak memory measurement."""
        current_rss = self.process.memory_info().rss
        if current_rss > self.peak_rss:
            self.peak_rss = current_rss


def percentage_improvement(baseline: int, greedy: int) -> float:
    """Calculate percentage improvement (positive = greedy is better)."""
    if baseline == 0:
        return 0.0
    return (baseline - greedy) / baseline * 100.0


def _run_monitor_update_loop(
    monitor: MemoryMonitor, should_monitor: dict[str, bool], interval: float = 0.0
) -> None:
    """Run monitor.update() in a loop until should_monitor is set to False."""
    while should_monitor.get("active", True):
        monitor.update()
        time.sleep(interval)


def run_baseline_contraction(
    network: TensorNetwork, contraction_tree: ctg.ContractionTree
) -> MemorySnapshot:
    """Run contraction without greedy strategy and measure memory."""
    monitor = MemoryMonitor()
    monitor.start()

    network_arrays = list(deepcopy(network.arrays))

    should_monitor = {"active": True}
    monitor_thread = threading.Thread(
        target=_run_monitor_update_loop, args=(monitor, should_monitor)
    )
    monitor_thread.daemon = True
    monitor_thread.start()

    contraction_tree.contract(network_arrays)

    should_monitor["active"] = False
    monitor_thread.join(timeout=1.0)

    memory = monitor.stop()

    # Cotengra calculates the abstract peak memory with no permutations, so a local optimal
    # assumption is used here
    return MemorySnapshot(
        LocalOptimalPermutationStrategy.get_peak_memory(
            network, contraction_tree.get_path()
        ).to_bytes,
        memory.total_bytes,
    )


def run_greedy_contraction(
    network: TensorNetwork,
    contraction_path: ContractionPath,
    k: int,
) -> MemorySnapshot:
    """Run contraction with greedy strategy and measure memory."""
    return MemorySnapshot(
        peak_bytes=GreedyPermutationStrategy.get_peak_memory(
            network,
            contraction_path,
            k=k,
        ).bytes,
        total_bytes=GreedyPermutationStrategy.get_total_memory(
            network, contraction_path=contraction_path, k=k
        ).bytes,
    )


class GreedyStrategyBenchmarkScript(
    AbstractSweepScript[
        GreedyBenchmarkWorkItem,
        GreedyBenchmarkResult,
        GreedyBenchmarkAggregate,
        list[Row],
    ]
):
    """Sweep runner for greedy strategy benchmarking."""

    description = "Benchmark greedy permutation strategy against baseline contraction"

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        """Configure script-specific CLI arguments."""
        parser.add_argument("--seed-start", type=int, default=0)
        parser.add_argument("--seed-end", type=int, default=2)
        parser.add_argument("--size-start", type=int, default=5)
        parser.add_argument("--size-end", type=int, default=10)
        parser.add_argument("--average-rank-start", type=int, default=4)
        parser.add_argument("--average-rank-end", type=int, default=4)
        parser.add_argument("--max-dim-start", type=int, default=12)
        parser.add_argument("--max-dim-end", type=int, default=12)
        parser.add_argument("--num-output-indices-start", type=int, default=2)
        parser.add_argument("--num-output-indices-end", type=int, default=2)
        parser.add_argument("--k-start", type=int, default=1)
        parser.add_argument("--k-end", type=int, default=3)
        parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    def validate_args(self, args: argparse.Namespace) -> None:
        """Validate CLI arguments."""
        if args.seed_start > args.seed_end:
            raise ValueError("seed-start must be <= seed-end")
        if args.size_start > args.size_end:
            raise ValueError("size-start must be <= size-end")
        if args.size_start < 2:
            raise ValueError("size-start must be >= 2")
        if args.average_rank_start > args.average_rank_end:
            raise ValueError("average-rank-start must be <= average-rank-end")
        if args.max_dim_start > args.max_dim_end:
            raise ValueError("max-dim-start must be <= max-dim-end")
        if args.num_output_indices_start > args.num_output_indices_end:
            raise ValueError("num-output-indices-start must be <= num-output-indices-end")
        if args.k_start > args.k_end:
            raise ValueError("k-start must be <= k-end")
        if args.k_start < 1:
            raise ValueError("k-start must be >= 1")

    def create_work_items(self, args: argparse.Namespace) -> list[GreedyBenchmarkWorkItem]:
        """Create the independent work items for the configured sweep."""
        work_items: list[GreedyBenchmarkWorkItem] = []
        for size in range(args.size_start, args.size_end + 1):
            for seed in range(args.seed_start, args.seed_end + 1):
                for average_rank in range(args.average_rank_start, args.average_rank_end + 1):
                    for max_dim in range(args.max_dim_start, args.max_dim_end + 1):
                        for num_output_indices in range(
                            args.num_output_indices_start, args.num_output_indices_end + 1
                        ):
                            for k in range(args.k_start, args.k_end + 1):
                                work_items.append(
                                    GreedyBenchmarkWorkItem(
                                        size=size,
                                        seed=seed,
                                        average_rank=average_rank,
                                        max_dim=max_dim,
                                        num_output_indices=num_output_indices,
                                        k=k,
                                    )
                                )
        return work_items

    @staticmethod
    def evaluate_work_item(item: GreedyBenchmarkWorkItem) -> GreedyBenchmarkResult:
        """Evaluate the benchmark for one sweep point."""
        tn = generate_random_tn(
            num_tensors=item.size,
            average_rank=item.average_rank,
            max_dim=item.max_dim,
            seed=item.seed,
            generate_arrays=True,
            num_output_indices=item.num_output_indices,
        )

        contraction_tree = ctg.array_contract_tree(
            inputs=tn.input_indices,
            output=tn.output_indices,
            size_dict=tn.size_dict,
            shapes=tn.shapes,
        )
        path = contraction_tree.get_path()

        max_k = len(tn.tensors) + len(path)
        k = min(item.k, max_k)

        baseline_memory = run_baseline_contraction(tn, contraction_tree)

        greedy_memory = run_greedy_contraction(tn, path, k)

        peak_improvement = percentage_improvement(
            baseline_memory.peak_bytes, greedy_memory.peak_bytes
        )
        total_improvement = percentage_improvement(
            baseline_memory.total_bytes, greedy_memory.total_bytes
        )

        row: Row = {
            "size": item.size,
            "seed": item.seed,
            "average_rank": item.average_rank,
            "max_dim": item.max_dim,
            "num_output_indices": item.num_output_indices,
            "k": item.k,
            "baseline_peak_bytes": baseline_memory.peak_bytes,
            "baseline_total_bytes": baseline_memory.total_bytes,
            "greedy_peak_bytes": greedy_memory.peak_bytes,
            "greedy_total_bytes": greedy_memory.total_bytes,
            "peak_improvement_pct": peak_improvement,
            "total_improvement_pct": total_improvement,
        }

        return GreedyBenchmarkResult(
            row=row,
            baseline_peak=baseline_memory.peak_bytes,
            baseline_total=baseline_memory.total_bytes,
            greedy_peak=greedy_memory.peak_bytes,
            greedy_total=greedy_memory.total_bytes,
        )

    def create_aggregate(self) -> GreedyBenchmarkAggregate:
        """Create the aggregate state used during the sweep."""
        return GreedyBenchmarkAggregate()

    def consume_work_result(
        self,
        aggregate: GreedyBenchmarkAggregate,
        result: GreedyBenchmarkResult,
    ) -> None:
        """Merge a completed work result into the aggregate state."""
        aggregate.rows.append(result.row)

    def build_output(
        self, aggregate: GreedyBenchmarkAggregate, args: argparse.Namespace
    ) -> list[Row]:
        """Build the final output from the aggregate state."""
        return aggregate.rows

    def describe_run(
        self,
        args: argparse.Namespace,
        work_items: list[GreedyBenchmarkWorkItem],
        parallel: bool,
        max_workers: int,
    ) -> str | None:
        """Return an optional description printed before execution starts."""
        return (
            f"Benchmarking greedy permutation strategy\n"
            f"  Work items: {len(work_items)}\n"
            f"  Parallel: {parallel} (workers: {max_workers})\n"
            f"  Output directory: {args.output_dir}"
        )


def write_csv(rows: list[Row], output_dir: Path) -> Path:
    """Write benchmark results to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "benchmark_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def build_peak_memory_plot(rows: list[Row], output_path: Path) -> None:
    """Build a plot comparing peak memory usage."""
    by_size_k: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for row in rows:
        size = int(row["size"])
        k = int(row["k"])
        baseline = float(row["baseline_peak_bytes"])
        greedy = float(row["greedy_peak_bytes"])
        key = (size, k)
        if key not in by_size_k:
            by_size_k[key] = []
        by_size_k[key].append((baseline, greedy))

    by_size_k_avg: dict[tuple[int, int], tuple[float, float]] = {}
    for key, values in by_size_k.items():
        baselines = [v[0] for v in values]
        greedy_vals = [v[1] for v in values]
        by_size_k_avg[key] = (float(np.mean(baselines)), float(np.mean(greedy_vals)))

    plt.figure(figsize=(12, 7))
    sizes = sorted(set(k[0] for k in by_size_k_avg.keys()))
    ks = sorted(set(k[1] for k in by_size_k_avg.keys()))

    x = np.arange(len(sizes))
    width = 0.15

    plt.get_cmap("tab10")
    for idx, k in enumerate(ks):
        baselines = []
        greedy_vals = []
        for size in sizes:
            key = (size, k)
            if key in by_size_k_avg:
                b, g = by_size_k_avg[key]
                baselines.append(b / 1e9)
                greedy_vals.append(g / 1e9)
            else:
                baselines.append(0)
                greedy_vals.append(0)

        offset = (idx - len(ks) / 2) * width
        plt.bar(x + offset, baselines, width, label=f"Baseline (k={k})", alpha=0.7)
        plt.bar(
            x + offset + len(ks) * width, greedy_vals, width, label=f"Greedy (k={k})", alpha=0.7
        )

    plt.xlabel("Tensor Network Size")
    plt.ylabel("Peak Memory (GB)")
    plt.title("Peak Memory: Baseline vs Greedy Strategy")
    plt.xticks(x + width * len(ks) / 2, sizes)  # type: ignore[arg-type]
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220)
    plt.close()


def build_improvement_plot(rows: list[Row], output_path: Path, metric: str) -> None:
    """Build a plot showing percentage improvement for a metric."""
    by_size_k: dict[tuple[int, int], list[float]] = {}
    for row in rows:
        size = int(row["size"])
        k = int(row["k"])
        improvement = float(row[metric])
        key = (size, k)
        if key not in by_size_k:
            by_size_k[key] = []
        by_size_k[key].append(improvement)

    by_size_k_avg: dict[tuple[int, int], float] = {}
    for key, values in by_size_k.items():
        by_size_k_avg[key] = float(np.mean(values))

    plt.figure(figsize=(12, 7))
    sizes = sorted(set(k[0] for k in by_size_k_avg.keys()))
    ks = sorted(set(k[1] for k in by_size_k_avg.keys()))

    metric_label = "Peak Memory" if "peak" in metric else "Total Memory"
    for k in ks:
        improvements = []
        for size in sizes:
            key = (size, k)
            if key in by_size_k_avg:
                improvements.append(by_size_k_avg[key])
            else:
                improvements.append(0)

        plt.plot(
            sizes,
            improvements,
            marker="o",
            linewidth=2,
            label=f"k={k}",
        )

    plt.xlabel("Tensor Network Size")
    plt.ylabel("Improvement (%)")
    plt.title(f"{metric_label} Improvement: Greedy vs Baseline")
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220)
    plt.close()


def main() -> None:
    """Run the benchmark script."""
    script = GreedyStrategyBenchmarkScript()
    args = script.parse_args()
    script.validate_args(args)

    output = script.run(args)

    csv_path = write_csv(output, args.output_dir)
    print(f"Results written to: {csv_path}")

    peak_plot_path = args.output_dir / "peak_memory_comparison.png"
    build_peak_memory_plot(output, peak_plot_path)
    print(f"Peak memory plot saved to: {peak_plot_path}")

    improvement_peak_path = args.output_dir / "peak_memory_improvement.png"
    build_improvement_plot(output, improvement_peak_path, "peak_improvement_pct")
    print(f"Peak improvement plot saved to: {improvement_peak_path}")

    improvement_total_path = args.output_dir / "total_memory_improvement.png"
    build_improvement_plot(output, improvement_total_path, "total_improvement_pct")
    print(f"Total improvement plot saved to: {improvement_total_path}")


if __name__ == "__main__":
    main()
