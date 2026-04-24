"""Script to analyze the impact of permuting the largest intermediate tensor on memory usage."""

from __future__ import annotations

import argparse
import csv
import inspect
import os
import sys
from pathlib import Path
from typing import Any

import cotengra as ctg
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "biggest_intermediate_memory_analysis"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from contraction.path import ContractionPath, PersistentContractionPath  # noqa: E402
from memory.calculator.calculator import MemoryCalculator  # noqa: E402
from memory.utils import (  # noqa: E402
    get_largest_intermediate_tensor_in_path,
    get_largest_tensor_in_network,
)
from tensor_network.tn import TensorNetwork  # noqa: E402
from tensor_network.utils.random import generate_random_tn  # noqa: E402


def parse_args() -> argparse.Namespace:  # noqa: D103
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=10)
    parser.add_argument("--size-start", type=int, default=5)
    parser.add_argument("--size-end", type=int, default=20)
    parser.add_argument("--average-rank-start", type=int, default=4)
    parser.add_argument("--average-rank-end", type=int, default=4)
    parser.add_argument("--max-dim-start", type=int, default=12)
    parser.add_argument("--max-dim-end", type=int, default=12)
    parser.add_argument("--num-output-indices-start", type=int, default=2)
    parser.add_argument("--num-output-indices-end", type=int, default=2)
    parser.add_argument("--strategy", type=str, default="local_optimal")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:  # noqa: D103
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


def strategy_find_kwargs(strategy_cls: Any, seed: int) -> dict[str, int]:  # noqa: D103
    kwargs: dict[str, int] = {}
    signature = inspect.signature(strategy_cls.find_optimal_permutation)
    if "seed" in signature.parameters:
        kwargs["seed"] = seed
    return kwargs


def simulate_memory(  # noqa: D103
    network: TensorNetwork,
    contraction_path: ContractionPath,
    skip_largest: bool,
) -> tuple[int, int]:
    calculator = MemoryCalculator()
    persistent_path = PersistentContractionPath.from_contraction_path(network, contraction_path)

    current_memory = calculator.calculate_memory_for_tensors(network.tensors)
    peak_memory = current_memory
    total_memory_movement = 0

    largest_step_idx, _ = get_largest_intermediate_tensor_in_path(network, contraction_path)
    largest_initial_idx: int | None = None
    if largest_step_idx < 0:
        largest_initial_idx, _ = get_largest_tensor_in_network(network)

    live_sources: list[tuple[str, int]] = [("initial", i) for i in range(len(network.tensors))]
    initial_permutation_applied = [False for _ in network.tensors]

    for step, (left_pos, right_pos) in enumerate(contraction_path):
        state_before = persistent_path.get_state(step)
        left_tensor = state_before.tensors[left_pos]
        right_tensor = state_before.tensors[right_pos]

        left_memory = calculator.calculate_memory_for_tensor(left_tensor)
        right_memory = calculator.calculate_memory_for_tensor(right_tensor)

        left_source = live_sources[left_pos]
        if left_source[0] == "initial":
            initial_idx = left_source[1]
            if not initial_permutation_applied[initial_idx]:
                should_permute_initial = True
                if skip_largest and largest_step_idx < 0 and initial_idx == largest_initial_idx:
                    should_permute_initial = False

                if should_permute_initial:
                    peak_memory = max(peak_memory, current_memory + left_memory)
                    total_memory_movement += left_memory.to_bytes + left_memory.to_bytes

                initial_permutation_applied[initial_idx] = True

        right_source = live_sources[right_pos]
        if right_source[0] == "initial":
            initial_idx = right_source[1]
            if not initial_permutation_applied[initial_idx]:
                should_permute_initial = True
                if skip_largest and largest_step_idx < 0 and initial_idx == largest_initial_idx:
                    should_permute_initial = False

                if should_permute_initial:
                    peak_memory = max(peak_memory, current_memory + right_memory)
                    total_memory_movement += right_memory.to_bytes + right_memory.to_bytes

                initial_permutation_applied[initial_idx] = True

        result_tensor = persistent_path.get_state(step + 1).tensors[left_pos]
        result_memory = calculator.calculate_memory_for_tensor(result_tensor)

        peak_memory = max(peak_memory, current_memory + result_memory)
        current_memory += result_memory
        current_memory -= left_memory + right_memory
        total_memory_movement += (
            left_memory.to_bytes + right_memory.to_bytes + result_memory.to_bytes
        )

        should_permute_result = True
        if skip_largest and step == largest_step_idx:
            should_permute_result = False

        if should_permute_result:
            peak_memory = max(peak_memory, current_memory + result_memory)
            total_memory_movement += result_memory.to_bytes + result_memory.to_bytes

        live_sources[left_pos] = ("intermediate", step)
        live_sources.pop(right_pos)

    return peak_memory.to_bytes, total_memory_movement


def percentage_delta(new_value: int, baseline: int) -> float:  # noqa: D103
    if baseline == 0:
        return 0.0
    return (new_value - baseline) / baseline * 100.0


def run_sweep(  # noqa: D103
    args: argparse.Namespace,
) -> tuple[list[dict[str, int | float | str]], list[dict[str, int | float | str]]]:
    rows: list[dict[str, int | float | str]] = []
    summary: dict[int, dict[str, float | int]] = {}

    sizes = range(args.size_start, args.size_end + 1)
    seeds = range(args.seed_start, args.seed_end + 1)
    average_ranks = range(args.average_rank_start, args.average_rank_end + 1)
    max_dims = range(args.max_dim_start, args.max_dim_end + 1)
    num_output_indices_list = range(args.num_output_indices_start, args.num_output_indices_end + 1)

    for size in sizes:
        print("=" * 50)
        print(f"Running size: {size} from seeds {args.seed_start} to {args.seed_end}")
        summary[size] = {
            "runs": 0,
            "permute_peak_sum": 0.0,
            "skip_peak_sum": 0.0,
            "peak_diff_sum": 0.0,
            "peak_diff_pct_sum": 0.0,
            "permute_total_sum": 0.0,
            "skip_total_sum": 0.0,
            "total_diff_sum": 0.0,
            "total_diff_pct_sum": 0.0,
        }

        for seed in seeds:
            for average_rank in average_ranks:
                for max_dim in max_dims:
                    for num_output_indices in num_output_indices_list:
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

                        largest_step_idx, _ = get_largest_intermediate_tensor_in_path(tn, path)

                        permute_peak, permute_total = simulate_memory(
                            tn,
                            path,
                            skip_largest=False,
                        )
                        skip_peak, skip_total = simulate_memory(
                            tn,
                            path,
                            skip_largest=True,
                        )

                        peak_diff = permute_peak - skip_peak
                        total_diff = permute_total - skip_total

                        peak_diff_pct = percentage_delta(permute_peak, skip_peak)
                        total_diff_pct = percentage_delta(permute_total, skip_total)

                        rows.append(
                            {
                                "strategy": args.strategy,
                                "size": size,
                                "seed": seed,
                                "average_rank": average_rank,
                                "max_dim": max_dim,
                                "num_output_indices": num_output_indices,
                                "largest_intermediate_step": largest_step_idx,
                                "permute_largest_peak_bytes": permute_peak,
                                "skip_largest_peak_bytes": skip_peak,
                                "peak_diff_bytes": peak_diff,
                                "peak_diff_pct": peak_diff_pct,
                                "permute_largest_total_bytes": permute_total,
                                "skip_largest_total_bytes": skip_total,
                                "total_diff_bytes": total_diff,
                                "total_diff_pct": total_diff_pct,
                            }
                        )

                        summary[size]["runs"] = int(summary[size]["runs"]) + 1
                        summary[size]["permute_peak_sum"] = (
                            float(summary[size]["permute_peak_sum"]) + permute_peak
                        )
                        summary[size]["skip_peak_sum"] = (
                            float(summary[size]["skip_peak_sum"]) + skip_peak
                        )
                        summary[size]["peak_diff_sum"] = (
                            float(summary[size]["peak_diff_sum"]) + peak_diff
                        )
                        summary[size]["peak_diff_pct_sum"] = (
                            float(summary[size]["peak_diff_pct_sum"]) + peak_diff_pct
                        )
                        summary[size]["permute_total_sum"] = (
                            float(summary[size]["permute_total_sum"]) + permute_total
                        )
                        summary[size]["skip_total_sum"] = (
                            float(summary[size]["skip_total_sum"]) + skip_total
                        )
                        summary[size]["total_diff_sum"] = (
                            float(summary[size]["total_diff_sum"]) + total_diff
                        )
                        summary[size]["total_diff_pct_sum"] = (
                            float(summary[size]["total_diff_pct_sum"]) + total_diff_pct
                        )

    summary_rows: list[dict[str, int | float | str]] = []
    for size in sizes:
        runs = int(summary[size]["runs"])
        summary_rows.append(
            {
                "strategy": args.strategy,
                "size": size,
                "runs": runs,
                "avg_permute_largest_peak_bytes": float(summary[size]["permute_peak_sum"]) / runs,
                "avg_skip_largest_peak_bytes": float(summary[size]["skip_peak_sum"]) / runs,
                "avg_peak_diff_bytes": float(summary[size]["peak_diff_sum"]) / runs,
                "avg_peak_diff_pct": float(summary[size]["peak_diff_pct_sum"]) / runs,
                "avg_permute_largest_total_bytes": float(summary[size]["permute_total_sum"]) / runs,
                "avg_skip_largest_total_bytes": float(summary[size]["skip_total_sum"]) / runs,
                "avg_total_diff_bytes": float(summary[size]["total_diff_sum"]) / runs,
                "avg_total_diff_pct": float(summary[size]["total_diff_pct_sum"]) / runs,
            }
        )

    return rows, summary_rows


def write_csv(  # noqa: D103
    rows: list[dict[str, int | float | str]],
    fieldnames: list[str],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_plot(rows: list[dict[str, int | float | str]], output_path: Path) -> None:  # noqa: D103
    ordered = sorted(rows, key=lambda row: int(row["size"]))
    sizes = [int(row["size"]) for row in ordered]
    peak_diff_pct = [float(row["avg_peak_diff_pct"]) for row in ordered]
    total_diff_pct = [float(row["avg_total_diff_pct"]) for row in ordered]

    plt.figure(figsize=(10, 6))
    plt.plot(sizes, peak_diff_pct, marker="o", linewidth=2, label="Peak memory change (%)")
    plt.plot(sizes, total_diff_pct, marker="s", linewidth=2, label="Total memory change (%)")
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Tensor network size")
    plt.ylabel("Difference when permuting largest intermediate tensor (%)")
    plt.title("Impact of permuting largest intermediate tensor")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220)
    plt.close()


def main() -> None:  # noqa: D103
    args = parse_args()
    validate_args(args)

    encoded_folder_name = (
        f"{args.size_start}_{args.size_end}_"
        f"{args.seed_start}_{args.seed_end}_"
        f"{args.average_rank_start}_{args.average_rank_end}_"
        f"{args.max_dim_start}_{args.max_dim_end}_"
        f"{args.num_output_indices_start}_{args.num_output_indices_end}"
    )

    if os.path.exists(args.output_dir / encoded_folder_name):
        print("A similar sweep already exists.")
        sys.exit(1)

    rows, summary_rows = run_sweep(args)

    os.makedirs(args.output_dir / encoded_folder_name, exist_ok=True)

    per_run_csv = args.output_dir / encoded_folder_name / "data.csv"
    per_size_csv = args.output_dir / encoded_folder_name / "summary_by_size.csv"
    plot_path = args.output_dir / encoded_folder_name / "impact_by_size.png"

    write_csv(
        rows,
        [
            "strategy",
            "size",
            "seed",
            "average_rank",
            "max_dim",
            "num_output_indices",
            "largest_intermediate_step",
            "permute_largest_peak_bytes",
            "skip_largest_peak_bytes",
            "peak_diff_bytes",
            "peak_diff_pct",
            "permute_largest_total_bytes",
            "skip_largest_total_bytes",
            "total_diff_bytes",
            "total_diff_pct",
        ],
        per_run_csv,
    )

    write_csv(
        summary_rows,
        [
            "strategy",
            "size",
            "runs",
            "avg_permute_largest_peak_bytes",
            "avg_skip_largest_peak_bytes",
            "avg_peak_diff_bytes",
            "avg_peak_diff_pct",
            "avg_permute_largest_total_bytes",
            "avg_skip_largest_total_bytes",
            "avg_total_diff_bytes",
            "avg_total_diff_pct",
        ],
        per_size_csv,
    )
    build_plot(summary_rows, plot_path)

    print(f"Saved per-run results to: {per_run_csv}")
    print(f"Saved per-size summary to: {per_size_csv}")
    print(f"Saved plot to: {plot_path}")


if __name__ == "__main__":
    main()
