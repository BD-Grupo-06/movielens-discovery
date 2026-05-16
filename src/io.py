import shutil
import urllib.request
import zipfile
from pathlib import Path
import polars as pl

def ensure_dir(path: Path) -> None:
    """Ensure that a directory exists."""
    path.mkdir(parents=True, exist_ok=True)

def read_parquet(path: Path) -> pl.DataFrame:
    """Read a parquet file and return a Polars DataFrame."""
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pl.read_parquet(path)

def download_file(url: str, destination: Path) -> None:
    """Download a file from a URL to a destination path."""
    with urllib.request.urlopen(url) as response, destination.open("wb") as target:
        shutil.copyfileobj(response, target)

def safe_extract(zip_path: Path, destination_dir: Path) -> None:
    """Safely extract a zip file to a destination directory."""
    destination_dir = destination_dir.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            resolved = (destination_dir / member.filename).resolve()
            if destination_dir not in resolved.parents and resolved != destination_dir:
                raise ValueError(f"Unsafe path in archive: {member.filename}")
        archive.extractall(destination_dir)
