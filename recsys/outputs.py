"""Shared directory layout for final results, figures and supporting data."""

from pathlib import Path
from shutil import copyfile


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
