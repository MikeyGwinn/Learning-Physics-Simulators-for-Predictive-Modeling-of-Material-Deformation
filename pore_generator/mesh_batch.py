"""Mesh successful batch samples sequentially, preserving existing mesh outputs."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from .export import mesh_geometry
from .generation import Config
from .meshing import geometry_metadata, inspect_mesh, mesh_sample, require_gmsh


def completed_mesh(gmsh, sample_directory, mesh_format):
    """Return false for untouched geometry; verify completed outputs or raise."""
    sample = json.loads((sample_directory / "sample.json").read_text(encoding="utf-8"))
    if sample["schema_version"] != 1 or sample["units"] != "mm":
        raise ValueError("Expected sample schema version 1 with millimeter units")
    Config(**sample["config"])
    output = sample_directory / f"{mesh_format}_mesh"
    geo_file = output / "specimen.geo"
    if not geo_file.is_file() or geo_file.read_text(encoding="utf-8") != mesh_geometry(sample, mesh_format):
        raise ValueError("Saved geometry is missing or differs from sample.json")
    required = ["specimen.msh", "mesh_summary.json"]
    if mesh_format == "perforated":
        required.append("geometry.json")
    results = ("specimen.msh", "mesh_summary.json", "mesh_preview.svg", "geometry.json")
    if not any((output / name).exists() for name in results):
        return False
    if not all((output / name).is_file() for name in required):
        raise ValueError("Existing mesh outputs are incomplete")
    if mesh_format == "material" and (output / "geometry.json").exists():
        raise ValueError("Material mesh has unexpected perforated compatibility metadata")
    with (output / "specimen.msh").open(encoding="utf-8") as stream:
        if stream.readline() != "$MeshFormat\n" or stream.readline() != "4.1 0 8\n":
            raise ValueError("Expected an ASCII MSH 4.1 mesh")
    saved = json.loads((output / "mesh_summary.json").read_text(encoding="utf-8"))
    expected_hash = hashlib.sha256(geo_file.read_bytes()).hexdigest()
    if saved.get("geometry_hash") != expected_hash:
        raise ValueError("Mesh summary geometry hash is missing or inconsistent")
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", 1)
        gmsh.open(str(output / "specimen.msh"))
        actual, _, _ = inspect_mesh(gmsh, sample, mesh_format)
    finally:
        gmsh.finalize()
    actual["geometry_hash"] = expected_hash
    # JSON object keys are strings (boundary tags in the inspector are integers).
    if saved != json.loads(json.dumps(actual)):
        raise ValueError("Mesh summary does not match the saved mesh")
    if mesh_format == "perforated":
        metadata = json.loads((output / "geometry.json").read_text(encoding="utf-8"))
        if metadata != geometry_metadata(sample, geo_file, actual):
            raise ValueError("Compatibility metadata does not match the saved geometry and mesh")
    return True


def mesh_batch(batch_directory, mesh_format="perforated"):
    if mesh_format not in ("perforated", "material", "both"):
        raise ValueError("format must be perforated, material, or both")
    gmsh = require_gmsh()
    batch_directory = Path(batch_directory)
    with (batch_directory / "samples.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not {"seed", "status", "sample_directory"}.issubset(reader.fieldnames or []):
            raise ValueError("samples.csv is missing required columns")
        samples = [row for row in reader if row["status"] == "success"]
    formats = ("material", "perforated") if mesh_format == "both" else (mesh_format,)
    totals = {"generated": 0, "skipped": 0, "failed": 0}
    total = len(samples) * len(formats)
    with (batch_directory / "meshing.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("seed", "sample_directory", "representation", "status", "error"))
        writer.writeheader()
        stream.flush()
        for sample in samples:
            for representation in formats:
                row = {"seed": sample["seed"], "sample_directory": sample["sample_directory"],
                       "representation": representation}
                try:
                    relative = Path(sample["sample_directory"])
                    if not sample["sample_directory"] or relative.is_absolute() or ".." in relative.parts:
                        raise ValueError("Expected a sample directory relative to the batch")
                    directory = batch_directory / relative
                    if completed_mesh(gmsh, directory, representation):
                        row["status"] = "skipped"
                    else:
                        mesh_sample(directory, representation)
                        row["status"] = "generated"
                except KeyboardInterrupt:
                    row.update(status="failed", error="Interrupted by user; outputs preserved")
                    raise
                except OSError as exc:
                    row.update(status="failed", error=f"Filesystem error: {exc}")
                    raise
                except Exception as exc:
                    row.update(status="failed", error=f"{type(exc).__name__}: {exc}. "
                               "Existing outputs preserved; inspect and manually clean up before retrying.")
                finally:
                    writer.writerow(row)
                    stream.flush()
                    totals[row["status"]] += 1
                print(f"[{sum(totals.values())}/{total}] Seed {row['seed']} {representation}: {row['status']}"
                      + (f" — {row['error']}" if row.get("error") else ""), flush=True)
    print(f"Meshing complete: {totals['generated']} generated, {totals['skipped']} skipped, "
          f"{totals['failed']} failed. Records: {batch_directory / 'meshing.csv'}", flush=True)
    return totals


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True, help="Generated batch directory containing samples.csv")
    parser.add_argument("--format", choices=("perforated", "material", "both"), default="perforated")
    args = parser.parse_args(argv)
    try:
        totals = mesh_batch(args.batch, args.format)
    except KeyboardInterrupt:
        parser.exit(130, "Meshing interrupted; existing outputs and CSV rows retained.\n")
    except (OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 1 if totals["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
