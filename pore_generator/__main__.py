"""Terminal entry point: python -m pore_generator generate --seed 42 --output ..."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .export import write_sample
from .generation import Config, generate


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate synthetic circular pores; stored lengths are mm.")
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("generate", help="Save a pore sample, CSV table, and SVG preview")
    command.add_argument("--seed", type=int, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--config", type=Path, help="JSON configuration; individual flags override its values")
    for name, default in asdict(Config()).items():
        command.add_argument("--" + name.replace("_", "-"), type=float, default=argparse.SUPPRESS,
                             help=f"Default: {default}")
    mesh_command = commands.add_parser("mesh", help="Mesh tagged regions, a perforated specimen, or both using Gmsh")
    mesh_command.add_argument("--sample", type=Path, required=True, help="Directory containing sample.json")
    mesh_command.add_argument("--format", choices=["material", "perforated", "both"], default="material")
    args = parser.parse_args(argv)
    try:
        if args.command == "mesh":
            from .meshing import mesh_sample
            formats = ("material", "perforated") if args.format == "both" else (args.format,)
            for mesh_format in formats:
                summary = mesh_sample(args.sample, mesh_format)
                print(f"Saved {summary['triangle_count']} triangles to {args.sample / (mesh_format + '_mesh')} "
                      f"(minimum quality {summary['minimum_quality']:.4f})")
            return 0
        values = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
        if not isinstance(values, dict):
            raise ValueError("Configuration must be a JSON object")
        values.update({name: getattr(args, name) for name in asdict(Config()) if hasattr(args, name)})
        config = Config(**values)
        if args.output.exists():
            raise ValueError(f"Output directory already exists: {args.output}")
        sample = generate(config, args.seed)
        write_sample(sample, args.output)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.exit(2, f"error: {exc}\n")
    statistics = sample["statistics"]
    print(f"Saved {statistics['pore_count']} pores to {args.output}")
    print(f"Porosity: {statistics['achieved_porosity']:.6f} "
          f"(explicit {statistics['explicit_porosity']:.6f}, small {statistics['small_porosity']:.6f}; "
          f"overshoot {statistics['porosity_overshoot']:.6f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
