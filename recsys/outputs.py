"""Shared directory layout for final results, figures and supporting data."""

from pathlib import Path


class OutputPaths:
    def __init__(self, root):
        self.root = Path(root)
        self.images = self.root / "images"
        self.intermediate = self.root / "intermediate"

    def create(self):
        for directory in (self.root, self.images, self.intermediate):
            directory.mkdir(parents=True, exist_ok=True)
        return self
