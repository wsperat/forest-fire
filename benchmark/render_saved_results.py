from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import BenchmarkResult, plot_grid_comparison, write_summary_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate benchmark grid plots and summaries from saved JSON "
            "results without rerunning the benchmark grid."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=Path("docs/benchmarks"))
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=("training", "prediction"),
        default=("training", "prediction"),
    )
    return parser.parse_args()


def load_results(path: Path) -> list[BenchmarkResult]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [BenchmarkResult(**item) for item in payload]


def render_grid(output_dir: Path, benchmark: str) -> None:
    for path in sorted(output_dir.glob(f"{benchmark}_grid_results_*.json")):
        results = load_results(path)
        if not results:
            continue

        family = results[0].family
        problem = results[0].problem
        row_grid = sorted({result.rows for result in results})
        feature_grid = sorted({result.n_features for result in results})
        metric = "fit_seconds" if benchmark == "training" else "predict_seconds"
        ylabel = (
            "fit time (seconds)"
            if benchmark == "training"
            else "predict time (seconds)"
        )
        title_prefix = (
            f"Training time | {family} | {problem}"
            if benchmark == "training"
            else f"Prediction time | {family} | {problem}"
        )
        summary_title = (
            f"Training summary | {family} | {problem}"
            if benchmark == "training"
            else f"Prediction summary | {family} | {problem}"
        )
        suffix = f"{family}_{problem}"

        plot_grid_comparison(
            results,
            metric=metric,
            row_grid=row_grid,
            feature_grid=feature_grid,
            title_prefix=title_prefix,
            ylabel=ylabel,
            output_path=output_dir / f"{benchmark}_grid_{suffix}.png",
        )
        write_summary_markdown(
            results,
            metric=metric,
            title=summary_title,
            output_path=output_dir / f"{benchmark}_summary_{suffix}.md",
        )


def main() -> None:
    args = parse_args()
    for benchmark in args.benchmarks:
        render_grid(args.output_dir, benchmark)


if __name__ == "__main__":
    main()
