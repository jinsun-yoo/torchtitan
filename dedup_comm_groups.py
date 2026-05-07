"""
Merge all per-rank comm_groups_{r}.json files in
  torchtitan/outputs/{keyword}/profile_trace/
into a single comm_groups.json in the same directory.

Panics if the same key maps to different values across files.

Usage:
    python dedup_comm_groups.py <keyword>

Example:
    python dedup_comm_groups.py 8b_tp_2_fsdp_2
"""

import json
import sys
from glob import glob
from pathlib import Path


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <keyword>", file=sys.stderr)
        sys.exit(1)

    keyword = sys.argv[1]
    trace_dir = Path("torchtitan/outputs") / keyword / "profile_trace"
    pattern = str(trace_dir / "comm_groups_*.json")

    files = sorted(glob(pattern))
    if not files:
        print(f"Error: no comm_groups_*.json files found in {trace_dir}", file=sys.stderr)
        sys.exit(1)

    merged: dict[str, list] = {}
    for fpath in files:
        with open(fpath) as f:
            data: dict = json.load(f)
        for key, value in data.items():
            if key in merged:
                if merged[key] != value:
                    print(
                        f"Error: conflicting values for key '{key}' "
                        f"in {fpath}:\n  existing: {merged[key]}\n  new: {value}",
                        file=sys.stderr,
                    )
                    sys.exit(1)
            else:
                merged[key] = value

    # Sort keys numerically if possible, else lexicographically
    def sort_key(k: str):
        try:
            return (0, int(k))
        except ValueError:
            return (1, k)

    sorted_merged = dict(sorted(merged.items(), key=lambda item: sort_key(item[0])))

    out_path = trace_dir / "comm_groups.json"
    with open(out_path, "w") as f:
        json.dump(sorted_merged, f, indent=2)

    print(f"Wrote {len(sorted_merged)} groups from {len(files)} rank files -> {out_path}")


if __name__ == "__main__":
    main()
