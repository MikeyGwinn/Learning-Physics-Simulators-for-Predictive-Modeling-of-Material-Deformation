"""Fetch selected real Skarberg image/mask pairs using ZIP HTTP ranges.

No full archive download. ZIP CRCs validate extracted members; archive MD5
identifies the upstream release but cannot validate a partial download.
"""
import argparse
import hashlib
import io
import json
import urllib.request
import zipfile
import zlib
from pathlib import Path

URL = "https://zenodo.org/records/4317170/files/fib_sem_cnn.zip?download=1"
SIZE = 9216974498


class RemoteZip(io.RawIOBase):
    def __init__(self, url=URL, size=SIZE):
        self.url, self.size, self.position = url, size, 0
    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position+offset if whence == 1 else self.size+offset
        return self.position
    def tell(self):
        return self.position
    def read(self, n=-1):
        n = self.size-self.position if n < 0 else min(n, self.size-self.position)
        if n <= 0:
            return b""
        start = self.position
        req = urllib.request.Request(self.url+f"&range_start={start}&range_size={n}",
              headers={"Range": f"bytes={start}-{start+n-1}", "Accept-Encoding": "identity"})
        with urllib.request.urlopen(req, timeout=90) as response:
            if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{start+n-1}/{self.size}":
                raise RuntimeError("Server did not honor the requested ZIP byte range; refusing full download")
            data = response.read(n+1)
        if len(data) != n:
            raise RuntimeError("Truncated or oversized HTTP range")
        self.position += n
        return data


def fetch(output, samples=(22, 30, 45), indices=(0, 100), archive=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "doi": "10.5281/zenodo.4317170", "license": "CC-BY-4.0", "pixel_size_nm": 10.0,
        "slice_spacing_nm": 50.0, "archive_md5": "a2884d289ed08f450692bd4fe8aa80b0",
        "mask_kind": "authors' CNN segmentation (not manual ground truth)",
        "mask_white": "solid", "images": []}
    manifest["mask_white"] = "solid"
    records = {r["id"]: r for r in manifest["images"]}
    with zipfile.ZipFile(archive if archive else RemoteZip()) as z:
        for sample in samples:
            if sample not in (22, 30, 45):
                raise ValueError("Samples are 22, 30, 45")
            for index in indices:
                if not 0 <= index < 200:
                    raise ValueError("Slice indices are 0..199")
                key = f"HPC{sample}_{index:04d}"
                raw = f"data/raw/HPC{sample}/HPC{sample}_{index+1:04d}.tif"
                mask = f"data/seg/HPC{sample}/seg_HPC{sample}_{index:04d}.png"
                record = {"id": key, "sample": f"HPC{sample}", "slice_index": index,
                          "image": f"raw/{key}.tif", "mask": f"masks/{key}.png",
                          "archive_image": raw, "archive_mask": mask}
                for field, member in [("image", raw), ("mask", mask)]:
                    target = output / record[field]
                    target.parent.mkdir(exist_ok=True)
                    if not target.exists():
                        print(f"Fetching {member}", flush=True)
                        data = z.read(member)  # ZipFile verifies each member's CRC.
                        temp = target.with_suffix(target.suffix+".part")
                        temp.write_bytes(data)
                        temp.replace(target)
                    elif zlib.crc32(target.read_bytes()) != z.getinfo(member).CRC:
                        raise ValueError(f"Cached file differs from the upstream ZIP CRC: {target}")
                    record[field+"_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
                records[key] = record
                manifest["images"] = sorted(records.values(), key=lambda r: r["id"])
                temp_manifest = manifest_path.with_suffix(".json.part")
                temp_manifest.write_text(json.dumps(manifest, indent=2)+"\n")
                temp_manifest.replace(manifest_path)
        provenance = output / "upstream_code"
        provenance.mkdir(exist_ok=True)
        for name in ("preprocess_data.py", "postprocessing.m", "predict_random_slice.py"):
            if name in z.namelist():
                (provenance / name).write_bytes(z.read(name))
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("datasets/skarberg"))
    p.add_argument("--samples", type=int, nargs="+", default=[22, 30, 45])
    p.add_argument("--indices", type=int, nargs="+", default=[0, 100])
    p.add_argument("--archive", type=Path, help="Use an already-downloaded full ZIP")
    a = p.parse_args()
    m = fetch(a.output, a.samples, a.indices, a.archive)
    print(f"{len(m['images'])} image/mask pairs indexed at {a.output}")
