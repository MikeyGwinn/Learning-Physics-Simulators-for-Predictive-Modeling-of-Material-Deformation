"""Select and cache raw Skärberg TIFF slices from the published ZIP archive."""

from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path
import urllib.request
import uuid
import zipfile
import zlib

from PIL import Image


ARCHIVE_URL = "https://zenodo.org/records/4317170/files/fib_sem_cnn.zip?download=1"
ARCHIVE_SIZE = 9_216_974_498
DATASET_DOI = "10.5281/zenodo.4317170"
SAMPLES = (22, 30, 45)
SLICE_INDICES = range(0, 200, 2)


class RemoteZip(io.RawIOBase):
    """Seek over the remote ZIP using verified HTTP byte-range responses."""

    def __init__(self, url: str = ARCHIVE_URL, size: int = ARCHIVE_SIZE):
        self.url, self.size, self.position = url, size, 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = 0) -> int:
        position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if not 0 <= position <= self.size:
            raise ValueError("seek outside archive")
        self.position = position
        return position

    def tell(self) -> int:
        return self.position

    def read(self, n: int = -1) -> bytes:
        n = self.size - self.position if n < 0 else min(n, self.size - self.position)
        if n <= 0:
            return b""
        start = self.position
        url = f"{self.url}&range_start={start}&range_size={n}"
        request = urllib.request.Request(
            url,
            headers={"Range": f"bytes={start}-{start+n-1}", "Accept-Encoding": "identity"},
        )
        with urllib.request.urlopen(request, timeout=90) as response:
            expected = f"bytes {start}-{start+n-1}/{self.size}"
            if response.status != 206 or response.headers.get("Content-Range") != expected:
                raise RuntimeError("archive server did not honor the requested byte range")
            data = response.read(n + 1)
        if len(data) != n:
            raise RuntimeError("truncated or oversized archive byte range")
        self.position += n
        return data


def selected_images() -> list[dict]:
    """Return 300 stable image IDs and their archive member paths."""
    return [
        {
            "id": f"HPC{sample}_{index:04d}",
            "sample": f"HPC{sample}",
            "slice_index": index,
            "archive_member": f"data/raw/HPC{sample}/HPC{sample}_{index+1:04d}.tif",
        }
        for sample in SAMPLES for index in SLICE_INDICES
    ]


def stage_images(
    records: list[dict], cache: Path, *, archive: Path | None = None,
    seed_dir: Path | None = None,
) -> list[dict]:
    """Verify cached TIFFs, then copy or download missing/corrupt files.

    ZIP reads verify each member's CRC; cached and seeded files are checked
    against the archive's CRC and uncompressed size before use.
    """
    cache = Path(cache).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    source = Path(archive) if archive is not None else RemoteZip()
    staged = []
    with zipfile.ZipFile(source) as bundle:
        for record in records:
            member = record["archive_member"]
            info = bundle.getinfo(member)
            path = cache / f"{record['id']}.tif"
            data = None
            cache_valid = False
            if path.is_file():
                candidate = path.read_bytes()
                if len(candidate) == info.file_size and zlib.crc32(candidate) == info.CRC:
                    data = candidate
                    cache_valid = True
            if data is None and seed_dir is not None:
                seed = Path(seed_dir) / f"{record['id']}.tif"
                if seed.is_file():
                    candidate = seed.read_bytes()
                    if len(candidate) == info.file_size and zlib.crc32(candidate) == info.CRC:
                        data = candidate
            if data is None:
                print(f"Fetching {record['id']}", flush=True)
                data = bundle.read(member)
            if len(data) != info.file_size or zlib.crc32(data) != info.CRC:
                raise ValueError(f"archive CRC mismatch for {record['id']}")
            with Image.open(io.BytesIO(data)) as image:
                if image.mode not in ("I;16", "I;16L", "I;16B"):
                    raise ValueError(f"{record['id']} is not a 16-bit grayscale TIFF")
                width, height = image.size
            if not cache_valid:
                temporary = path.with_name(path.name + f".{os.getpid()}.{uuid.uuid4().hex}.part")
                try:
                    temporary.write_bytes(data)
                    os.replace(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
            staged.append({
                **record,
                "path": str(path),
                "sha256": hashlib.sha256(data).hexdigest(),
                "archive_crc32": f"{info.CRC:08x}",
                "width": width,
                "height": height,
            })
    return staged
