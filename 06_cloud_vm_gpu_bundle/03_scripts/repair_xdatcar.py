from __future__ import annotations

import argparse
from pathlib import Path


def looks_like_coordinate(line: str) -> bool:
    parts = line.split()
    if len(parts) < 3:
        return False
    try:
        float(parts[0])
        float(parts[1])
        float(parts[2])
    except ValueError:
        return False
    return True


def normalize_xdatcar(path: Path) -> bool:
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    if len(raw_lines) < 8:
        raise ValueError(f"XDATCAR is too short: {path}")

    header = raw_lines[:7]
    counts = [int(token) for token in header[6].split()]
    natoms = sum(counts)

    payload = [line.strip() for line in raw_lines[7:]]
    payload = [line for line in payload if line]
    if payload and payload[0].lower() == "direct":
        payload = payload[1:]

    frames: list[list[str]] = []
    i = 0
    while i < len(payload):
        line = payload[i]
        if line.startswith("# Step:") or line.lower().startswith("direct configuration"):
            i += 1
        if i >= len(payload):
            break
        if not looks_like_coordinate(payload[i]):
            raise ValueError(f"Unexpected XDATCAR payload line in {path}: {payload[i]}")
        frame = payload[i : i + natoms]
        if len(frame) != natoms:
            raise ValueError(f"Incomplete XDATCAR frame in {path}")
        if not all(looks_like_coordinate(coord_line) for coord_line in frame):
            raise ValueError(f"Malformed coordinate block in {path}")
        frames.append(frame)
        i += natoms

    normalized_lines = header[:]
    for frame_index, frame in enumerate(frames, start=1):
        normalized_lines.append(f"Direct configuration= {frame_index:6d}")
        normalized_lines.extend(frame)

    normalized_text = "\n".join(normalized_lines) + "\n"
    original_text = "\n".join(raw_lines) + "\n"
    if normalized_text == original_text:
        return False
    path.write_text(normalized_text, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize UMA-written XDATCAR files to strict VASP-style format.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "04_runs",
        help="Root directory to scan recursively",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    repaired = 0
    for path in sorted(root.rglob("XDATCAR")):
        if normalize_xdatcar(path):
            repaired += 1
            print(f"[repaired] {path}")

    print(f"[summary] repaired_xdatcar_files={repaired}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
