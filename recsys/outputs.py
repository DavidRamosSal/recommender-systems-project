"""Shared directory layout for final results, figures and supporting data."""

import csv
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

from filelock import SoftFileLock


RESULT_COLUMNS = [
    "Model", "Variant / settings", "Similarity", "k", "Precision@10",
    "Recall@10", "NDCG@10", "Catalogue coverage", "Seed", "Notes",
]
PART2_PREFIX = "Part 2. "


def write_results_csv(path, rows):
    with TemporaryDirectory(dir=path.parent) as directory:
        staged = Path(directory) / path.name
        with staged.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=RESULT_COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        staged.replace(path)


def update_overall_results(root, rows, *, part, partial=False):
    """Refresh one part of results.csv. The original Part 1 rows have no prefix."""
    destination = Path(root) / "results.csv"
    with SoftFileLock(str(destination) + ".lock"):
        previous = []
        if destination.exists():
            with destination.open(encoding="utf-8", newline="") as source:
                reader = csv.DictReader(source)
                if reader.fieldnames != RESULT_COLUMNS:
                    raise ValueError(f"Unexpected results schema in {destination}")
                previous = list(reader)
        part1 = [r for r in previous if not r["Notes"].startswith(PART2_PREFIX)]
        part2 = [r for r in previous if r["Notes"].startswith(PART2_PREFIX)]
        if part == 1:
            part1 = rows
        else:
            updated = {r["Model"]: {**r, "Notes": PART2_PREFIX + r["Notes"]} for r in rows}
            retained = {r["Model"]: r for r in part2} if partial else {}
            retained.update(updated)
            part2 = list(retained.values())
        write_results_csv(destination, [*part1, *part2])


def copy_outputs(source, destination, files):
    """Reuse completed experiment tables or figures when replaying a saved run."""
    for name in files:
        src, dst = Path(source) / name, Path(destination) / name
        if not src.is_file():
            raise FileNotFoundError(f"Missing saved result {src}. Run with --retrain to regenerate it.")
        if src.resolve() != dst.resolve():
            dst.parent.mkdir(parents=True, exist_ok=True)
            copyfile(src, dst)


class OutputPaths:
    def __init__(self, root):
        self.root = Path(root)
        self.images = self.root / "images"
        self.intermediate = self.root / "intermediate"

    def create(self):
        for directory in (self.root, self.images, self.intermediate):
            directory.mkdir(parents=True, exist_ok=True)
        return self
