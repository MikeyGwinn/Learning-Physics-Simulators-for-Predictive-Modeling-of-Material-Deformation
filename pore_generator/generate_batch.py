"""Generate a requested number of successful samples using consecutive seeds."""

import argparse
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
import platform
import random

from .export import write_sample
from .generation import Config, generate


STATISTICS = ("pore_count", "explicit_pore_count", "small_pore_count", "achieved_porosity")
FIELDS = ("seed", "status", "sample_directory", *STATISTICS, "error")
SPLIT_RULE = "test if random.Random(f'split-{seed}').random() < test_fraction, else train"


def split_label(seed, test_fraction):
    """Label a seed from the seed alone, so labels stay fixed as a dataset grows."""
    return "test" if random.Random(f"split-{seed}").random() < test_fraction else "train"


def generate_batch(config, count, start_seed, output, test_fraction=0.2):
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("count must be a positive integer")
    if isinstance(start_seed, bool) or not isinstance(start_seed, int):
        raise ValueError("start_seed must be an integer")
    if (isinstance(test_fraction, bool) or not isinstance(test_fraction, (int, float))
            or not math.isfinite(test_fraction) or not 0 <= test_fraction <= 1):
        raise ValueError("test_fraction must be in [0, 1]")
    # Validate before creating the output directory, including direct API calls.
    config = Config(**asdict(config))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    totals = {"attempted": 0, "success": 0, "failed": 0, "interrupted": 0, "error": 0}
    batch = {
        "schema_version": 1,
        "config": asdict(config),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "requested_count": count,
        "start_seed": start_seed,
        "test_fraction": test_fraction,
        "split_rule": SPLIT_RULE,
        "status": "running",
        "totals": totals,
    }
    splits = []

    def save_record():
        (output / "batch.json").write_text(
            json.dumps(batch, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        # Train/test labels for every saved sample, including those kept after an interruption.
        with (output / "split.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["seed", "sample_directory", "split"])
            writer.writerows(splits)

    try:
        save_record()
        with (output / "samples.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            stream.flush()
            seed = start_seed
            while totals["success"] < count:
                row = {"seed": seed}
                try:
                    try:
                        sample = generate(config, seed)
                    except ValueError as exc:
                        row.update(status="failed", error=str(exc))
                    else:
                        directory = f"seed-{seed:06d}"
                        write_sample(sample, output / directory)
                        row.update(status="success", sample_directory=directory)
                        row.update({key: sample["statistics"][key] for key in STATISTICS})
                        splits.append((seed, directory, split_label(seed, test_fraction)))
                except KeyboardInterrupt:
                    row.update(status="interrupted", error="Interrupted by user")
                    raise
                except Exception as exc:
                    row.update(status="error", error=f"{type(exc).__name__}: {exc}")
                    raise
                finally:
                    writer.writerow(row)
                    stream.flush()
                    totals["attempted"] += 1
                    totals[row["status"]] += 1
                print(f"Seed {seed}: {row['status']} ({totals['success']}/{count} samples)"
                      + (f" — {row['error']}" if row.get("error") else ""), flush=True)
                seed += 1
        batch["status"] = "complete"
    except KeyboardInterrupt:
        batch["status"] = "interrupted"
    except Exception as exc:
        batch.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save_record()
    test_count = sum(label == "test" for _, _, label in splits)
    print(f"Batch {batch['status']}: {totals['success']} saved, {totals['failed']} rejected, "
          f"{totals['attempted']} attempted; {len(splits) - test_count} train, {test_count} test. "
          f"Records: {output}", flush=True)
    return batch


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=500, help="Number of successful samples (default: 500)")
    parser.add_argument("--start-seed", type=int, default=0, help="First seed to attempt (default: 0)")
    parser.add_argument("--output", type=Path, required=True, help="New batch directory; existing batches cannot resume")
    parser.add_argument("--config", type=Path, help="JSON configuration, using the single-sample keys")
    parser.add_argument("--test-fraction", type=float, default=0.2,
                        help="Expected fraction of samples labeled test in split.csv (default: 0.2)")
    args = parser.parse_args(argv)
    try:
        values = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
        if not isinstance(values, dict):
            raise ValueError("Configuration must be a JSON object")
        config = Config(**values)
        if args.count <= 0:
            raise ValueError("count must be a positive integer")
        if not 0 <= args.test_fraction <= 1:
            raise ValueError("test_fraction must be in [0, 1]")
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(2, f"error: {exc}\n")
    try:
        batch = generate_batch(config, args.count, args.start_seed, args.output, args.test_fraction)
    except OSError as exc:
        parser.exit(2, f"error: {exc}\n")
    return 130 if batch["status"] == "interrupted" else 0


if __name__ == "__main__":
    raise SystemExit(main())
