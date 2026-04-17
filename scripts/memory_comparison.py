"""Compare permutation strategies and visualize average memory metrics by size.

This script sweeps over tensor network sizes and random seeds, evaluates all
permutation strategies, and then builds line graphs where each line is one
strategy and each point is the average over all seeds for that size.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import multiprocessing as mp
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cotengra as ctg
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "memory_strategies_comparison"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from permutation.strategy import IPermutationStrategy  # noqa: I001, E402
from permutation.strategy.canonical_contracted_first import (  # noqa: I001, E402
    CanonicalContractedFirstPermutationStrategy,
)
from permutation.strategy.canonical_free_first import (  # noqa: I001, E402
    CanonicalFreeFirstPermutationStrategy,
)  # noqa: I001, E402
from permutation.strategy.greedy import GreedyPermutationStrategy  # noqa: I001, E402
from permutation.strategy.local_optimal import LocalOptimalPermutationStrategy  # noqa: I001, E402
from permutation.strategy.next_use_aware import NextUseAwarePermutationStrategy  # noqa: I001, E402
from permutation.strategy.preserve_layout import PreserveLayoutPermutationStrategy  # noqa: I001, E402
from tensor_network.utils.random import generate_random_tn  # noqa: I001, E402

SweepKey = tuple[int, int, int, int, int, str]
WorkItem = tuple[int, int, int, int, int]
PartialAggregate = dict[SweepKey, tuple[int, int]]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for sweep configuration."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare all permutation strategies by averaging memory metrics over "
            "seed sweeps for each tensor network size."
        )
    )
    parser.add_argument("--seed-start", type=int, default=5, help="Inclusive initial seed.")
    parser.add_argument("--seed-end", type=int, default=50, help="Inclusive final seed.")
    parser.add_argument("--size-start", type=int, default=5, help="Inclusive initial size.")
    parser.add_argument("--size-end", type=int, default=20, help="Inclusive final size.")
    parser.add_argument(
        "--average-rank-start",
        type=int,
        default=3,
        help="Minimum average tensor rank for generated random tensor networks.",
    )
    parser.add_argument(
        "--average-rank-end",
        type=int,
        default=3,
        help="Maximum average tensor rank for generated random tensor networks.",
    )
    parser.add_argument(
        "--max-dim-start",
        type=int,
        default=8,
        help="Minimum maximum dimension size for generated tensor indices.",
    )
    parser.add_argument(
        "--max-dim-end",
        type=int,
        default=8,
        help="Maximum dimension size for generated tensor indices.",
    )
    parser.add_argument(
        "--num-output-indices-start",
        type=int,
        default=2,
        help="Minimum number of output indices in the generated tensor network.",
    )
    parser.add_argument(
        "--num-output-indices-end",
        type=int,
        default=2,
        help="Number of output indices in the generated tensor network.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for CSV and graphs.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively in addition to saving them.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of worker processes for --parallel mode (default: CPU count).",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose mode.")
    return parser.parse_args()


def get_strategies() -> list[IPermutationStrategy]:
    """Return all permutation strategies in this repository."""
    return [
        GreedyPermutationStrategy,
        LocalOptimalPermutationStrategy,
        NextUseAwarePermutationStrategy,
        CanonicalContractedFirstPermutationStrategy,
        CanonicalFreeFirstPermutationStrategy,
        PreserveLayoutPermutationStrategy,
        # RandomTTGTPermutationStrategy,
    ]


def strategy_kwargs(strategy_cls: IPermutationStrategy, seed: int) -> dict[str, int]:
    """Provide kwargs for strategies that accept optional seed parameters."""
    kwargs: dict[str, int] = {}
    peak_sig = inspect.signature(strategy_cls.get_peak_memory)
    total_sig = inspect.signature(strategy_cls.get_total_memory)
    if "seed" in peak_sig.parameters and "seed" in total_sig.parameters:
        kwargs["seed"] = seed
    return kwargs


def validate_ranges(args: argparse.Namespace) -> None:
    """Validate seed and size sweep ranges."""
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
    if args.num_workers is not None and args.num_workers < 1:
        raise ValueError("num-workers must be >= 1")


def create_work_items(args: argparse.Namespace) -> list[WorkItem]:
    """Build independent work items for dynamic scheduling across processes."""
    sizes = range(args.size_start, args.size_end + 1)
    seeds = range(args.seed_start, args.seed_end + 1)
    average_ranks = range(args.average_rank_start, args.average_rank_end + 1)
    max_dims = range(args.max_dim_start, args.max_dim_end + 1)
    num_output_indices_list = range(args.num_output_indices_start, args.num_output_indices_end + 1)

    work_items: list[WorkItem] = []
    for size in sizes:
        for seed in seeds:
            for average_rank in average_ranks:
                for max_dim in max_dims:
                    for num_output_indices in num_output_indices_list:
                        work_items.append((size, seed, average_rank, max_dim, num_output_indices))
    return work_items


def evaluate_work_item(item: WorkItem) -> PartialAggregate:
    """Evaluate all strategies for a single sweep item and return partial sums."""
    size, seed, average_rank, max_dim, num_output_indices = item
    tn = generate_random_tn(
        num_tensors=size,
        average_rank=average_rank,
        max_dim=max_dim,
        seed=seed,
        generate_arrays=False,
        num_output_indices=num_output_indices,
    )
    contraction_tree = ctg.array_contract_tree(
        inputs=tn.input_indices,
        output=tn.output_indices,
        size_dict=tn.size_dict,
        shapes=tn.shapes,
    )
    path = contraction_tree.get_path()

    partial: PartialAggregate = {}
    for strategy_cls in get_strategies():
        kwargs = strategy_kwargs(strategy_cls, seed)
        peak_memory = strategy_cls.get_peak_memory(tn, path, **kwargs)
        total_memory = strategy_cls.get_total_memory(tn, path, **kwargs)
        key: SweepKey = (
            size,
            seed,
            average_rank,
            max_dim,
            num_output_indices,
            strategy_cls.__name__,  # type: ignore[attr-defined]
        )
        partial[key] = (peak_memory.to_bytes, total_memory.to_bytes)
    return partial


def build_rows_from_aggregates(
    args: argparse.Namespace,
    strategies: list[IPermutationStrategy],
    peak_values: dict[SweepKey, int],
    total_values: dict[SweepKey, int],
) -> list[dict[str, float | int | str]]:
    """Build final output rows from accumulated memory values."""
    sizes = range(args.size_start, args.size_end + 1)
    seeds = range(args.seed_start, args.seed_end + 1)
    average_ranks = range(args.average_rank_start, args.average_rank_end + 1)
    max_dims = range(args.max_dim_start, args.max_dim_end + 1)
    num_output_indices_list = range(args.num_output_indices_start, args.num_output_indices_end + 1)

    rows: list[dict[str, float | int | str]] = []
    for size in sizes:
        for seed in seeds:
            for average_rank in average_ranks:
                for max_dim in max_dims:
                    for num_output_indices in num_output_indices_list:
                        for strategy_cls in strategies:
                            strategy_name = strategy_cls.__name__  # type: ignore[attr-defined]
                            key: SweepKey = (
                                size,
                                seed,
                                average_rank,
                                max_dim,
                                num_output_indices,
                                strategy_name,
                            )
                            if key not in peak_values:
                                continue
                            rows.append(
                                {
                                    "size": size,
                                    "seed": seed,
                                    "average_rank": average_rank,
                                    "max_dim": max_dim,
                                    "num_output_indices": num_output_indices,
                                    "strategy": strategy_name,
                                    "avg_peak_bytes": peak_values[key],
                                    "avg_total_bytes": total_values[key],
                                }
                            )
    return rows


def run_sweep(args: argparse.Namespace) -> list[dict[str, float | int | str]]:
    """Execute the configured size/seed sweep and return averaged rows."""
    strategies = get_strategies()
    sizes = range(args.size_start, args.size_end + 1)
    seeds = range(args.seed_start, args.seed_end + 1)
    average_ranks = range(args.average_rank_start, args.average_rank_end + 1)
    max_dims = range(args.max_dim_start, args.max_dim_end + 1)
    num_output_indices_list = range(args.num_output_indices_start, args.num_output_indices_end + 1)

    peak_values: dict[tuple[int, int, int, int, int, str], int] = {}
    total_values: dict[tuple[int, int, int, int, int, str], int] = {}

    print(f"\n{'=' * 60}")
    print(
        f"Sweeping from size {args.size_start} to {args.size_end}\n"
        f"with seeds {args.seed_start}..{args.seed_end}\n"
        f"with average rank {args.average_rank_start}..{args.average_rank_end}\n"
        f"with max dim {args.max_dim_start}..{args.max_dim_end}\n"
        f"with num output indices {args.num_output_indices_start}..{args.num_output_indices_end}\n"
        f"and strategies {[cls.__name__ for cls in strategies]}"  # type: ignore[attr-defined]
    )

    for size in sizes:
        print("Processing size:", size)
        for seed in seeds:
            for average_rank in average_ranks:
                for max_dim in max_dims:
                    for num_output_indices in num_output_indices_list:
                        if args.verbose:
                            print(f"\n{'=' * 60}")
                            print(
                                f"Sweeping size {size}\n"
                                f"with seed {seed}\n"
                                f"with average rank {average_rank}\n"
                                f"with max dim {max_dim}\n"
                                f"with num output indices {num_output_indices}"
                            )
                        tn = generate_random_tn(
                            num_tensors=size,
                            average_rank=average_rank,
                            max_dim=max_dim,
                            seed=seed,
                            generate_arrays=False,
                            num_output_indices=num_output_indices,
                        )
                        contraction_tree = ctg.array_contract_tree(
                            inputs=tn.input_indices,
                            output=tn.output_indices,
                            size_dict=tn.size_dict,
                            shapes=tn.shapes,
                        )
                        path = contraction_tree.get_path()

                        for strategy_cls in strategies:
                            kwargs = strategy_kwargs(strategy_cls, seed)
                            peak_memory = strategy_cls.get_peak_memory(tn, path, **kwargs)
                            total_memory = strategy_cls.get_total_memory(tn, path, **kwargs)
                            key = (
                                size,
                                seed,
                                average_rank,
                                max_dim,
                                num_output_indices,
                                strategy_cls.__name__,  # type: ignore[attr-defined]
                            )
                            peak_values[key] = peak_memory.to_bytes
                            total_values[key] = total_memory.to_bytes

    return build_rows_from_aggregates(args, strategies, peak_values, total_values)


def run_sweep_parallel(args: argparse.Namespace) -> list[dict[str, float | int | str]]:
    """Execute the configured sweep in parallel and return averaged rows."""
    strategies = get_strategies()
    work_items = create_work_items(args)
    max_workers = args.num_workers or os.cpu_count() or 1

    peak_values: dict[SweepKey, int] = {}
    total_values: dict[SweepKey, int] = {}
    failed_items: list[WorkItem] = []

    print(f"\n{'=' * 60}")
    print(
        f"Running parallel sweep with {max_workers} workers\n"
        f"Total work items: {len(work_items)}\n"
        f"and strategies {[cls.__name__ for cls in strategies]}"  # type: ignore[attr-defined]
    )

    mp_context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_context) as executor:
        futures = {executor.submit(evaluate_work_item, item): item for item in work_items}

        for completed, future in enumerate(as_completed(futures), start=1):
            item = futures[future]
            try:
                partial = future.result()
            except Exception as exc:
                failed_items.append(item)
                if args.verbose:
                    print(f"Failed item {item}: {exc}")
                continue

            for key, (peak_val, total_val) in partial.items():
                peak_values[key] = peak_val
                total_values[key] = total_val

            if args.verbose and completed % (len(work_items) // 100) == 0:
                print(f"Completed {completed / len(work_items) * 100:.1f}% of work items", end="\r")

    if failed_items:
        print(f"Warning: {len(failed_items)} work items failed and were skipped.")
        preview = failed_items[:5]
        print(f"First failed items: {preview}")

    return build_rows_from_aggregates(args, strategies, peak_values, total_values)


def write_csv(rows: list[dict[str, float | int | str]], output_dir: Path) -> Path:
    """Write aggregated sweep results to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "data.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "size",
                "seed",
                "average_rank",
                "max_dim",
                "num_output_indices",
                "strategy",
                "avg_peak_bytes",
                "avg_total_bytes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def build_line_plot(
    rows: list[dict[str, float | int | str]],
    metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
    show: bool,
) -> None:
    """Build and save a line plot for a chosen metric by size and strategy."""
    by_strategy: dict[str, list[tuple[int, float]]] = defaultdict(list)
    sizes: set[int] = set()
    for row in rows:
        strategy = str(row["strategy"])
        if strategy == "PreserveLayoutPermutationStrategy":
            continue
        size = int(row["size"])
        value = float(row[metric])
        sizes.add(size)
        by_strategy[strategy].append((size, value))

    plt.figure(figsize=(12, 7))
    cmap = plt.get_cmap("tab10")
    for idx, (strategy, points) in enumerate(sorted(by_strategy.items())):
        ordered = sorted(points, key=lambda item: item[0])
        x_values = [item[0] for item in ordered]
        y_values = [item[1] for item in ordered]
        plt.plot(
            x_values,
            y_values,
            marker="o",
            linewidth=2,
            color=cmap(idx % 10),
            label=strategy,
        )

    plt.title(title)
    plt.xlabel("Tensor Network Size (num_tensors)")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220)
    if show:
        plt.show()
    plt.close()


def build_overlapping_bar_plot(
    rows: list[dict[str, float | int | str]],
    metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
    show: bool,
) -> None:
    """Build and save an overlapping bar plot for a chosen metric by size and strategy."""
    by_strategy: dict[str, dict[int, float]] = defaultdict(dict)
    sizes: set[int] = set()
    for row in rows:
        strategy = str(row["strategy"])
        if strategy == "PreserveLayoutPermutationStrategy":
            continue
        size = int(row["size"])
        value = float(row[metric])
        sizes.add(size)
        by_strategy[strategy][size] = value

    sorted_sizes = sorted(sizes)
    strategies = sorted(by_strategy.keys())

    cmap = plt.get_cmap("tab10")
    color_map = {strategy: cmap(idx % 10) for idx, strategy in enumerate(strategies)}

    plt.figure(figsize=(12, 7))
    for size in sorted_sizes:
        strategy_values = [
            (strategy, by_strategy[strategy].get(size, 0.0)) for strategy in strategies
        ]
        strategy_values.sort(key=lambda item: -item[1])

        for strategy, value in strategy_values:
            plt.bar(
                size,
                value,
                width=0.75,
                color=color_map[strategy],
            )

    plt.title(title)
    plt.xlabel("Tensor Network Size (num_tensors)")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    legend_handles = [
        Patch(facecolor=color_map[strategy], label=strategy) for strategy in strategies
    ]
    plt.legend(handles=legend_handles, loc="best", fontsize=9)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220)
    if show:
        plt.show()
    plt.close()


def build_comaprison_with_preserve_layout_plot(
    rows: list[dict[str, float | int | str]],
    metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
    show: bool,
) -> None:
    """Build and save a line graph of differences vs PreserveLayout by size."""
    by_strategy: dict[str, dict[int, float]] = defaultdict(dict)
    sizes: set[int] = set()
    for row in rows:
        strategy = str(row["strategy"])
        size = int(row["size"])
        value = float(row[metric])
        sizes.add(size)
        by_strategy[strategy][size] = value

    sorted_sizes = sorted(sizes)
    strategies = sorted(by_strategy.keys())
    if "PreserveLayoutPermutationStrategy" not in strategies:
        raise ValueError("PreserveLayoutPermutationStrategy data is required for this plot")

    preserve_layout_values = by_strategy["PreserveLayoutPermutationStrategy"]

    cmap = plt.get_cmap("tab10")
    color_map = {
        strategy: cmap(idx % 10)
        for idx, strategy in enumerate(strategies)
        if strategy != "PreserveLayoutPermutationStrategy"
    }

    plt.figure(figsize=(12, 7))
    for strategy in strategies:
        if strategy == "PreserveLayoutPermutationStrategy":
            continue

        x_values = sorted_sizes
        y_values = [
            (by_strategy[strategy].get(size, 0.0) - preserve_layout_values.get(size, 0.0))
            / preserve_layout_values.get(size, 1.0)
            * 100
            for size in sorted_sizes
        ]
        plt.plot(
            x_values,
            y_values,
            marker="o",
            linewidth=2,
            color=color_map[strategy],
            label=strategy,
        )

    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.title(title)
    plt.xlabel("Tensor Network Size (num_tensors)")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    legend_handles = [
        Patch(facecolor=color_map[strategy], label=strategy)
        for strategy in strategies
        if strategy != "PreserveLayoutPermutationStrategy"
    ]
    plt.legend(handles=legend_handles, loc="best", fontsize=9)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220)
    if show:
        plt.show()
    plt.close()


def main() -> None:
    """Run the sweep, export CSV, and save strategy-comparison line graphs."""
    args = parse_args()
    validate_ranges(args)

    encoded_folder_name = (
        f"{args.size_start}_{args.size_end}_"
        f"{args.seed_start}_{args.seed_end}_"
        f"{args.average_rank_start}_{args.average_rank_end}_"
        f"{args.max_dim_start}_{args.max_dim_end}_"
        f"{args.num_output_indices_start}_{args.num_output_indices_end}"
    )
    folder_path = args.output_dir / encoded_folder_name

    if os.path.exists(folder_path):
        print("A similar sweep already exists.")
        sys.exit(1)

    if args.num_workers != 1:
        rows = run_sweep_parallel(args)
    else:
        rows = run_sweep(args)

    csv_path = write_csv(rows, folder_path)
    os.makedirs(folder_path, exist_ok=True)

    build_line_plot(
        rows=rows,
        metric="avg_peak_bytes",
        title="Average Peak Memory by Size (Averaged Across Seed Sweep)",
        ylabel="Average Peak Memory (bytes)",
        output_path=folder_path / "avg_peak_memory_by_size.png",
        show=args.show,
    )
    build_line_plot(
        rows=rows,
        metric="avg_total_bytes",
        title="Average Total Memory by Size (Averaged Across Seed Sweep)",
        ylabel="Average Total Memory (bytes)",
        output_path=folder_path / "avg_total_memory_by_size.png",
        show=args.show,
    )
    build_comaprison_with_preserve_layout_plot(
        rows=rows,
        metric="avg_peak_bytes",
        title="Difference in Average Peak Memory Compared to PreserveLayout",
        ylabel="Average Peak Memory Difference (percentage)",
        output_path=folder_path / "peak_memory_difference_from_preserve_layout.png",
        show=args.show,
    )
    build_comaprison_with_preserve_layout_plot(
        rows=rows,
        metric="avg_total_bytes",
        title="Difference in Average Total Memory Compared to PreserveLayout",
        ylabel="Average Total Memory Difference (percentage)",
        output_path=folder_path / "total_memory_difference_from_preserve_layout.png",
        show=args.show,
    )

    print("\nCompleted strategy comparison sweep.")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved graphs in: {folder_path}")


if __name__ == "__main__":
    main()
