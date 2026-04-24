"""Analyze the impact of ignoring the largest intermediate tensors."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass, field
from heapq import nlargest
from pathlib import Path

import cotengra as ctg
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "biggest_intermediate_memory_analysis"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sweep_script import AbstractSweepScript  # noqa: E402

from contraction.path import ContractionPath, PersistentContractionPath  # noqa: E402
from memory.calculator.calculator import MemoryCalculator  # noqa: E402
from tensor_network.tn import TensorNetwork  # noqa: E402
from tensor_network.utils.random import generate_random_tn  # noqa: E402

Row = dict[str, int | float | str]

PER_RUN_FIELDNAMES = [
    "strategy",
    "largest_k",
    "size",
    "seed",
    "average_rank",
    "max_dim",
    "num_output_indices",
    "largest_intermediate_steps",
    "largest_intermediate_memory_bytes",
    "permute_all_peak_bytes",
    "ignore_top_k_peak_bytes",
    "peak_diff_bytes",
    "peak_diff_pct",
    "permute_all_total_bytes",
    "ignore_top_k_total_bytes",
    "total_diff_bytes",
    "total_diff_pct",
]

SUMMARY_FIELDNAMES = [
    "strategy",
    "largest_k",
    "size",
    "runs",
    "avg_permute_all_peak_bytes",
    "avg_ignore_top_k_peak_bytes",
    "avg_peak_diff_bytes",
    "avg_peak_diff_pct",
    "avg_permute_all_total_bytes",
    "avg_ignore_top_k_total_bytes",
    "avg_total_diff_bytes",
    "avg_total_diff_pct",
]


@dataclass(frozen=True)
class BiggestIntermediateWorkItem:
    """Independent work item for the intermediate-memory sweep."""

    strategy: str
    largest_k: int
    size: int
    seed: int
    average_rank: int
    max_dim: int
    num_output_indices: int


@dataclass
class BiggestIntermediateWorkResult:
    """Metrics produced by a single work item."""

    row: Row
    size: int
    permute_all_peak: int
    ignore_top_k_peak: int
    peak_diff: int
    peak_diff_pct: float
    permute_all_total: int
    ignore_top_k_total: int
    total_diff: int
    total_diff_pct: float


@dataclass
class SizeSummaryAccumulator:
    """Running totals for one tensor-network size."""

    runs: int = 0
    permute_all_peak_sum: float = 0.0
    ignore_top_k_peak_sum: float = 0.0
    peak_diff_sum: float = 0.0
    peak_diff_pct_sum: float = 0.0
    permute_all_total_sum: float = 0.0
    ignore_top_k_total_sum: float = 0.0
    total_diff_sum: float = 0.0
    total_diff_pct_sum: float = 0.0


@dataclass
class BiggestIntermediateAggregate:
    """Aggregate state while the sweep is running."""

    rows: list[Row] = field(default_factory=list)
    summaries_by_size: dict[int, SizeSummaryAccumulator] = field(default_factory=dict)


@dataclass(frozen=True)
class BiggestIntermediateOutput:
    """Final output of the intermediate-memory analysis script."""

    rows: list[Row]
    summary_rows: list[Row]


def get_largest_k_created_intermediate_tensors_in_path(
    network: TensorNetwork,
    contraction_path: ContractionPath,
    k: int,
) -> tuple[list[int], list[int]]:
    """Return the contraction steps and memory of the largest created intermediates."""
    calculator = MemoryCalculator()
    persistent_path = PersistentContractionPath.from_contraction_path(network, contraction_path)
    intermediate_tensors: list[tuple[int, int]] = []

    for step_idx, contraction_pair in enumerate(contraction_path):
        result_tensor = persistent_path.get_state(step_idx + 1).tensors[contraction_pair[0]]
        intermediate_tensors.append(
            (step_idx, calculator.calculate_memory_for_tensor(result_tensor).to_bytes)
        )

    largest_k_intermediate_tensors = nlargest(k, intermediate_tensors, key=lambda item: item[1])
    largest_step_indices = [step_idx for step_idx, _ in largest_k_intermediate_tensors]
    largest_memories = [memory for _, memory in largest_k_intermediate_tensors]
    return largest_step_indices, largest_memories


def simulate_memory(
    network: TensorNetwork,
    contraction_path: ContractionPath,
    ignored_intermediate_steps: set[int] | None = None,
) -> tuple[int, int]:
    """Simulate peak and total memory movement for a contraction path."""
    calculator = MemoryCalculator()
    persistent_path = PersistentContractionPath.from_contraction_path(network, contraction_path)
    ignored_intermediate_steps = ignored_intermediate_steps or set()

    current_memory = calculator.calculate_memory_for_tensors(network.tensors)
    peak_memory = current_memory
    total_memory_movement = 0

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
                peak_memory = max(peak_memory, current_memory + left_memory)
                total_memory_movement += left_memory.to_bytes + left_memory.to_bytes
                initial_permutation_applied[initial_idx] = True

        right_source = live_sources[right_pos]
        if right_source[0] == "initial":
            initial_idx = right_source[1]
            if not initial_permutation_applied[initial_idx]:
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

        if step not in ignored_intermediate_steps:
            peak_memory = max(peak_memory, current_memory + result_memory)
            total_memory_movement += result_memory.to_bytes + result_memory.to_bytes

        live_sources[left_pos] = ("intermediate", step)
        live_sources.pop(right_pos)

    return peak_memory.to_bytes, total_memory_movement


def percentage_delta(new_value: int, baseline: int) -> float:
    """Return the percentage difference between two memory metrics."""
    if baseline == 0:
        return 0.0
    return (new_value - baseline) / baseline * 100.0


def write_csv(rows: list[Row], fieldnames: list[str], output_path: Path) -> None:
    """Write rows to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_plot(rows: list[Row], output_path: Path) -> None:
    """Build the summary plot for the intermediate-memory analysis."""
    ordered = sorted(rows, key=lambda row: int(row["size"]))
    sizes = [int(row["size"]) for row in ordered]
    peak_diff_pct = [float(row["avg_peak_diff_pct"]) for row in ordered]
    total_diff_pct = [float(row["avg_total_diff_pct"]) for row in ordered]
    largest_k = int(rows[0]["largest_k"])

    plt.figure(figsize=(10, 6))
    plt.plot(sizes, peak_diff_pct, marker="o", linewidth=2, label="Peak memory change (%)")
    plt.plot(sizes, total_diff_pct, marker="s", linewidth=2, label="Total memory change (%)")
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Tensor network size")
    plt.ylabel(f"Difference when ignoring top {largest_k} intermediate tensors (%)")
    plt.title(f"Impact of ignoring top {largest_k} intermediate tensors")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220)
    plt.close()


class BiggestIntermediateMemoryAnalysisScript(
    AbstractSweepScript[
        BiggestIntermediateWorkItem,
        BiggestIntermediateWorkResult,
        BiggestIntermediateAggregate,
        BiggestIntermediateOutput,
    ]
):
    """Sweep runner for the biggest-intermediate-memory analysis."""

    description = (
        "Analyze the impact of ignoring the permutation of the largest intermediate tensors."
    )

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        """Configure script-specific CLI arguments."""
        parser.add_argument("--largest-k", type=int, default=1)
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

    def validate_args(self, args: argparse.Namespace) -> None:
        """Validate CLI arguments."""
        if args.largest_k < 1:
            raise ValueError("largest-k must be >= 1")
        if args.seed_start > args.seed_end:
            raise ValueError("seed-start must be <= seed-end")
        if args.size_start > args.size_end:
            raise ValueError("size-start must be <= size-end")
        if args.size_start < 2:
            raise ValueError("size-start must be >= 2")
        if args.largest_k > args.size_start - 1:
            raise ValueError("largest-k must be <= size-start - 1")
        if args.average_rank_start > args.average_rank_end:
            raise ValueError("average-rank-start must be <= average-rank-end")
        if args.max_dim_start > args.max_dim_end:
            raise ValueError("max-dim-start must be <= max-dim-end")
        if args.num_output_indices_start > args.num_output_indices_end:
            raise ValueError("num-output-indices-start must be <= num-output-indices-end")

    def create_work_items(self, args: argparse.Namespace) -> list[BiggestIntermediateWorkItem]:
        """Create the independent work items for the configured sweep."""
        work_items: list[BiggestIntermediateWorkItem] = []
        for size in range(args.size_start, args.size_end + 1):
            for seed in range(args.seed_start, args.seed_end + 1):
                for average_rank in range(args.average_rank_start, args.average_rank_end + 1):
                    for max_dim in range(args.max_dim_start, args.max_dim_end + 1):
                        for num_output_indices in range(
                            args.num_output_indices_start, args.num_output_indices_end + 1
                        ):
                            work_items.append(
                                BiggestIntermediateWorkItem(
                                    strategy=args.strategy,
                                    largest_k=args.largest_k,
                                    size=size,
                                    seed=seed,
                                    average_rank=average_rank,
                                    max_dim=max_dim,
                                    num_output_indices=num_output_indices,
                                )
                            )
        return work_items

    @staticmethod
    def evaluate_work_item(item: BiggestIntermediateWorkItem) -> BiggestIntermediateWorkResult:
        """Evaluate the memory metrics for one sweep point."""
        tn = generate_random_tn(
            num_tensors=item.size,
            average_rank=item.average_rank,
            max_dim=item.max_dim,
            seed=item.seed,
            generate_arrays=False,
            num_output_indices=item.num_output_indices,
        )
        contraction_tree = ctg.array_contract_tree(
            inputs=tn.input_indices,
            output=tn.output_indices,
            size_dict=tn.size_dict,
            shapes=tn.shapes,
        )
        path = contraction_tree.get_path()

        largest_step_idxs, largest_memories = get_largest_k_created_intermediate_tensors_in_path(
            tn,
            path,
            k=item.largest_k,
        )

        permute_all_peak, permute_all_total = simulate_memory(tn, path)
        ignore_top_k_peak, ignore_top_k_total = simulate_memory(
            tn,
            path,
            ignored_intermediate_steps=set(largest_step_idxs),
        )

        peak_diff = permute_all_peak - ignore_top_k_peak
        total_diff = permute_all_total - ignore_top_k_total

        peak_diff_pct = percentage_delta(permute_all_peak, ignore_top_k_peak)
        total_diff_pct = percentage_delta(permute_all_total, ignore_top_k_total)

        row: Row = {
            "strategy": item.strategy,
            "largest_k": item.largest_k,
            "size": item.size,
            "seed": item.seed,
            "average_rank": item.average_rank,
            "max_dim": item.max_dim,
            "num_output_indices": item.num_output_indices,
            "largest_intermediate_steps": "|".join(str(step_idx) for step_idx in largest_step_idxs),
            "largest_intermediate_memory_bytes": "|".join(
                str(memory) for memory in largest_memories
            ),
            "permute_all_peak_bytes": permute_all_peak,
            "ignore_top_k_peak_bytes": ignore_top_k_peak,
            "peak_diff_bytes": peak_diff,
            "peak_diff_pct": peak_diff_pct,
            "permute_all_total_bytes": permute_all_total,
            "ignore_top_k_total_bytes": ignore_top_k_total,
            "total_diff_bytes": total_diff,
            "total_diff_pct": total_diff_pct,
        }

        return BiggestIntermediateWorkResult(
            row=row,
            size=item.size,
            permute_all_peak=permute_all_peak,
            ignore_top_k_peak=ignore_top_k_peak,
            peak_diff=peak_diff,
            peak_diff_pct=peak_diff_pct,
            permute_all_total=permute_all_total,
            ignore_top_k_total=ignore_top_k_total,
            total_diff=total_diff,
            total_diff_pct=total_diff_pct,
        )

    def create_aggregate(self) -> BiggestIntermediateAggregate:
        """Create the aggregate state used during the sweep."""
        return BiggestIntermediateAggregate()

    def consume_work_result(
        self,
        aggregate: BiggestIntermediateAggregate,
        result: BiggestIntermediateWorkResult,
    ) -> None:
        """Merge one completed work item into the aggregate state."""
        aggregate.rows.append(result.row)

        summary = aggregate.summaries_by_size.setdefault(result.size, SizeSummaryAccumulator())
        summary.runs += 1
        summary.permute_all_peak_sum += result.permute_all_peak
        summary.ignore_top_k_peak_sum += result.ignore_top_k_peak
        summary.peak_diff_sum += result.peak_diff
        summary.peak_diff_pct_sum += result.peak_diff_pct
        summary.permute_all_total_sum += result.permute_all_total
        summary.ignore_top_k_total_sum += result.ignore_top_k_total
        summary.total_diff_sum += result.total_diff
        summary.total_diff_pct_sum += result.total_diff_pct

    def build_output(
        self,
        aggregate: BiggestIntermediateAggregate,
        args: argparse.Namespace,
    ) -> BiggestIntermediateOutput:
        """Build the final per-run and per-size output rows."""
        ordered_rows = sorted(
            aggregate.rows,
            key=lambda row: (
                int(row["size"]),
                int(row["seed"]),
                int(row["average_rank"]),
                int(row["max_dim"]),
                int(row["num_output_indices"]),
            ),
        )

        summary_rows: list[Row] = []
        for size in range(args.size_start, args.size_end + 1):
            summary = aggregate.summaries_by_size.get(size)
            if summary is None or summary.runs == 0:
                continue

            summary_rows.append(
                {
                    "strategy": args.strategy,
                    "largest_k": args.largest_k,
                    "size": size,
                    "runs": summary.runs,
                    "avg_permute_all_peak_bytes": summary.permute_all_peak_sum / summary.runs,
                    "avg_ignore_top_k_peak_bytes": summary.ignore_top_k_peak_sum / summary.runs,
                    "avg_peak_diff_bytes": summary.peak_diff_sum / summary.runs,
                    "avg_peak_diff_pct": summary.peak_diff_pct_sum / summary.runs,
                    "avg_permute_all_total_bytes": summary.permute_all_total_sum / summary.runs,
                    "avg_ignore_top_k_total_bytes": summary.ignore_top_k_total_sum / summary.runs,
                    "avg_total_diff_bytes": summary.total_diff_sum / summary.runs,
                    "avg_total_diff_pct": summary.total_diff_pct_sum / summary.runs,
                }
            )

        return BiggestIntermediateOutput(rows=ordered_rows, summary_rows=summary_rows)

    def describe_run(
        self,
        args: argparse.Namespace,
        work_items: list[BiggestIntermediateWorkItem],
        parallel: bool,
        max_workers: int,
    ) -> str:
        """Describe the configured sweep before it starts."""
        mode = "parallel" if parallel else "sequential"
        return (
            f"Running {mode} intermediate-memory sweep\n"
            f"Total work items: {len(work_items)}\n"
            f"Largest k: {args.largest_k}\n"
            f"Sizes: {args.size_start}..{args.size_end}\n"
            f"Seeds: {args.seed_start}..{args.seed_end}\n"
            f"Average rank: {args.average_rank_start}..{args.average_rank_end}\n"
            f"Max dim: {args.max_dim_start}..{args.max_dim_end}\n"
            f"Num output indices: {args.num_output_indices_start}..{args.num_output_indices_end}\n"
            f"Workers: {max_workers}"
        )


def main() -> None:
    """Run the sweep, export CSVs, and save the summary plot."""
    script = BiggestIntermediateMemoryAnalysisScript()
    args = script.parse_args()

    encoded_folder_name = (
        f"{args.largest_k}_"
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

    output = script.run(args)

    os.makedirs(folder_path, exist_ok=True)

    per_run_csv = folder_path / "data.csv"
    per_size_csv = folder_path / "summary_by_size.csv"
    plot_path = folder_path / "impact_by_size.png"

    write_csv(output.rows, PER_RUN_FIELDNAMES, per_run_csv)
    write_csv(output.summary_rows, SUMMARY_FIELDNAMES, per_size_csv)
    build_plot(output.summary_rows, plot_path)

    print(f"Saved per-run results to: {per_run_csv}")
    print(f"Saved per-size summary to: {per_size_csv}")
    print(f"Saved plot to: {plot_path}")


if __name__ == "__main__":
    main()
