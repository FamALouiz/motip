"""Reusable execution pipeline for sweep-style analysis scripts."""

from __future__ import annotations

import argparse
import multiprocessing as mp
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Generic, TypeVar

WorkItemT = TypeVar("WorkItemT")
WorkResultT = TypeVar("WorkResultT")
AggregateT = TypeVar("AggregateT")
OutputT = TypeVar("OutputT")


class AbstractSweepScript(ABC, Generic[WorkItemT, WorkResultT, AggregateT, OutputT]):
    """Base class for scripts that sweep over independent work items."""

    description: str | None = None

    def build_parser(self) -> argparse.ArgumentParser:
        """Build the command-line parser for the script."""
        parser = argparse.ArgumentParser(description=self.description)
        self.configure_parser(parser)
        self.add_execution_arguments(parser)
        return parser

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        """Configure script-specific command-line arguments."""

    def add_execution_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add the standard execution controls shared by sweep scripts."""
        parser.add_argument(
            "--num-workers",
            type=int,
            default=1,
            help="Number of worker processes. Use 1 for sequential execution.",
        )
        parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose mode.")

    def parse_args(self) -> argparse.Namespace:
        """Parse command-line arguments."""
        return self.build_parser().parse_args()

    def __validate_internal_arguments(self, args: argparse.Namespace) -> None:
        """Validate shared arguments before the sweep begins."""
        if args.num_workers < 1:
            raise ValueError("num-workers must be >= 1")

    @abstractmethod
    def validate_args(self, args: argparse.Namespace) -> None:
        """Validate the parsed arguments before execution begins."""

    @abstractmethod
    def create_work_items(self, args: argparse.Namespace) -> list[WorkItemT]:
        """Build the independent work items for the configured sweep."""

    @staticmethod
    @abstractmethod
    def evaluate_work_item(item: WorkItemT) -> WorkResultT:
        """Evaluate a single work item."""

    @abstractmethod
    def create_aggregate(self) -> AggregateT:
        """Create the mutable aggregation state for completed work items."""

    @abstractmethod
    def consume_work_result(
        self,
        aggregate: AggregateT,
        result: WorkResultT,
    ) -> None:
        """Merge a completed work result into the aggregate state."""

    @abstractmethod
    def build_output(self, aggregate: AggregateT, args: argparse.Namespace) -> OutputT:
        """Build the final script output from the aggregate state."""

    def get_num_workers(self, args: argparse.Namespace) -> int:
        """Return the configured worker count."""
        return args.num_workers

    def should_run_parallel(self, args: argparse.Namespace) -> bool:
        """Return whether the sweep should run in parallel."""
        return self.get_num_workers(args) > 1

    def continue_on_error(self, args: argparse.Namespace) -> bool:
        """Return whether failed work items should be skipped."""
        return False

    def describe_run(
        self,
        args: argparse.Namespace,
        work_items: list[WorkItemT],
        parallel: bool,
        max_workers: int,
    ) -> str | None:
        """Return an optional description printed before execution starts."""
        return None

    def on_work_item_failed(
        self,
        item: WorkItemT,
        exc: Exception,
        args: argparse.Namespace,
    ) -> None:
        """Handle a failed work item when continue_on_error is enabled."""
        if self.should_report_failed_items(args):
            print(f"Failed work item {item}: {exc}")

    def on_failed_items(self, failed_items: list[WorkItemT], args: argparse.Namespace) -> None:
        """Handle any skipped work items after execution completes."""
        print(f"Warning: {len(failed_items)} work items failed and were skipped.")
        print(f"First failed items: {failed_items[:5]}")

    def should_report_failed_items(self, args: argparse.Namespace) -> bool:
        """Return whether failed work items should be reported when continue_on_error is enabled."""
        return True

    def should_report_progress(self, args: argparse.Namespace) -> bool:
        """Return whether the script should emit progress updates."""
        return getattr(args, "verbose", False)

    def progress_interval(self, total_items: int) -> int:
        """Return the number of completed items between progress updates."""
        return max(1, total_items // 100)

    def report_progress(
        self,
        completed: int,
        total_items: int,
        args: argparse.Namespace,
        *,
        parallel: bool,
    ) -> None:
        """Emit a generic progress update when verbose mode is enabled."""
        interval = self.progress_interval(total_items)
        if completed % interval != 0 and completed != total_items:
            return

        message = f"Completed {completed}/{total_items} work items"
        if parallel and completed != total_items:
            print(message, end="\r")
            return
        print(message)

    def run(self, args: argparse.Namespace) -> OutputT:
        """Execute the sweep sequentially or in parallel and return the final output."""
        self.__validate_internal_arguments(args)
        self.validate_args(args)
        work_items = self.create_work_items(args)
        aggregate = self.create_aggregate()
        parallel = self.should_run_parallel(args)
        max_workers = self.get_num_workers(args) if parallel else 1

        description = self.describe_run(args, work_items, parallel, max_workers)
        if description:
            print(f"\n{'=' * 60}")
            print(description)

        if parallel:
            failed_items = self._run_parallel(work_items, aggregate, args, max_workers=max_workers)
        else:
            failed_items = self._run_sequential(work_items, aggregate, args)

        if failed_items and self.should_report_failed_items(args):
            self.on_failed_items(failed_items, args)
        return self.build_output(aggregate, args)

    def _run_sequential(
        self,
        work_items: list[WorkItemT],
        aggregate: AggregateT,
        args: argparse.Namespace,
    ) -> list[WorkItemT]:
        """Execute all work items in the current process."""
        failed_items: list[WorkItemT] = []

        for completed, item in enumerate(work_items, start=1):
            try:
                result = self.evaluate_work_item(item)
            except Exception as exc:
                if not self.continue_on_error(args):
                    raise RuntimeError(f"Work item {item} failed") from exc
                failed_items.append(item)
                self.on_work_item_failed(item, exc, args)
                continue

            self.consume_work_result(aggregate, result)

            if self.should_report_progress(args):
                self.report_progress(completed, len(work_items), args, parallel=False)

        return failed_items

    def _run_parallel(
        self,
        work_items: list[WorkItemT],
        aggregate: AggregateT,
        args: argparse.Namespace,
        *,
        max_workers: int,
    ) -> list[WorkItemT]:
        """Execute all work items across a process pool."""
        failed_items: list[WorkItemT] = []
        mp_context = mp.get_context("spawn")

        with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_context) as executor:
            futures = {executor.submit(self.evaluate_work_item, item): item for item in work_items}

            for completed, future in enumerate(as_completed(futures), start=1):
                item = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    if not self.continue_on_error(args):
                        raise RuntimeError(f"Work item {item} failed") from exc
                    failed_items.append(item)
                    self.on_work_item_failed(item, exc, args)
                    continue

                self.consume_work_result(aggregate, result)
                if self.should_report_progress(args):
                    self.report_progress(completed, len(work_items), args, parallel=True)

        return failed_items
