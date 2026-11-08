import argparse
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile


URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def download_data(data_dir: str | Path = "data") -> Path:
    """Download and extract unless all three required data files already exist."""
    data_dir = Path(data_dir)
    destination = data_dir / "ml-1m"
    if all((destination / name).is_file() for name in ("ratings.dat", "movies.dat", "users.dat")):
        return destination

    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL}")
    with urlopen(URL, timeout=60) as response:
        archive_bytes = response.read()
    with ZipFile(BytesIO(archive_bytes)) as archive:
        archive.extractall(data_dir)

    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download MovieLens 1M.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Parent directory for ml-1m (default: data).",
    )
    args = parser.parse_args()
    print(f"Dataset ready: {download_data(args.data_dir)}")
