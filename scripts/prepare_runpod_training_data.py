#!/usr/bin/env python3
"""Prepare FUTO English + Telugu synthetic manifests for RunPod training.

This wrapper keeps the expensive pod workflow repeatable. It can optionally
download the FUTO JSONL files from Hugging Face, build a Dakshina romanized
dictionary, generate Telugu synthetic swipes, filter FUTO, and merge final
train/validation manifests for ``new/train_transducer_personalized.py``.

Default outputs are written under ``data/runpod_futo_te``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "runpod_futo_te"
FUTO_FILENAMES = {
    "train": "train.jsonl",
    "dev": "dev.jsonl",
    "test": "test.jsonl",
}


def run_command(cmd: Sequence[str], cwd: Optional[Path] = None) -> None:
    printable = " ".join(str(part) for part in cmd)
    where = f" cwd={cwd}" if cwd else ""
    print(f"\n[run]{where} {printable}")
    subprocess.run(list(cmd), cwd=str(cwd) if cwd else None, check=True)


def resolve_path(value: str, base: Path = REPO_ROOT) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as file:
        return sum(1 for _ in file)


def find_jsonl(root: Path, filename: str) -> Optional[Path]:
    candidate = root / filename
    if candidate.exists():
        return candidate
    matches = sorted(root.rglob(filename)) if root.exists() else []
    return matches[0] if matches else None


def maybe_download_futo(args: argparse.Namespace, futo_dir: Path) -> None:
    if not args.download_futo:
        return
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required for --download-futo. Run `uv sync` on the pod first."
        ) from exc

    futo_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.futo_repo} JSONL files to {futo_dir}")
    snapshot_download(
        repo_id=args.futo_repo,
        repo_type="dataset",
        local_dir=str(futo_dir),
        allow_patterns=["*.jsonl"],
        token=os.environ.get("HF_TOKEN"),
    )


def ensure_dakshina_dict(args: argparse.Namespace, out_dir: Path) -> Path:
    dict_path = resolve_path(args.dict_path) if args.dict_path else None
    if dict_path and dict_path.exists():
        return dict_path

    dakshina_dir = resolve_path(args.dakshina_dir)
    builder = dakshina_dir / "build_lang_dict.py"
    if not builder.exists():
        missing = dict_path or Path(args.dict_path or "<not provided>")
        raise SystemExit(
            f"Dictionary not found ({missing}) and builder not found at {builder}."
        )

    dict_out_dir = out_dir / "dakshina_dict"
    dict_out_dir.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            sys.executable,
            str(builder),
            "--lang",
            args.lang,
            "--vocab-cap",
            str(args.vocab_cap),
            "--output-dir",
            str(dict_out_dir),
        ],
        cwd=dakshina_dir,
    )
    built = dict_out_dir / f"{args.lang}_dict.json"
    if not built.exists():
        raise SystemExit(f"Dakshina dictionary build did not produce {built}")
    return built


def filter_futo_files(
    args: argparse.Namespace, futo_dir: Path, out_dir: Path
) -> Dict[str, Path]:
    filtered_dir = out_dir / "futo_filtered"
    filtered_dir.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, Path] = {}
    filter_script = REPO_ROOT / "scripts" / "filter_and_normalize_dataset.py"

    for split, filename in FUTO_FILENAMES.items():
        source = find_jsonl(futo_dir, filename)
        if source is None:
            if split == "train" and not args.skip_futo:
                raise SystemExit(
                    f"Missing FUTO {filename} under {futo_dir}. Use --download-futo or copy the files there."
                )
            print(f"[skip] FUTO {split}: {filename} not found under {futo_dir}")
            continue
        target = filtered_dir / f"futo_{split}_filtered.jsonl"
        cmd = [
            sys.executable,
            str(filter_script),
            str(source),
            str(target),
            "--min-word-freq",
            str(args.min_word_freq),
            "--max-word-list",
            str(args.max_word_list),
        ]
        if args.max_futo_lines:
            cmd.extend(["--max-lines", str(args.max_futo_lines)])
        run_command(cmd, cwd=REPO_ROOT)
        outputs[split] = target
    return outputs


def generate_synthetic(args: argparse.Namespace, dict_path: Path, out_dir: Path) -> Tuple[Path, Path]:
    train_out = out_dir / f"{args.lang}_synthetic_train.jsonl"
    val_out = out_dir / f"{args.lang}_synthetic_val.jsonl"
    generator = REPO_ROOT / "scripts" / "generate_synthetic_swipes.py"
    cmd = [
        sys.executable,
        str(generator),
        "--dict-path",
        str(dict_path),
        "--out-train",
        str(train_out),
        "--out-val",
        str(val_out),
        "--val-split",
        str(args.synthetic_val_split),
        "--seed",
        str(args.seed),
        "--coordinate-space",
        "zero_one",
    ]
    if args.max_synthetic_words:
        cmd.extend(["--max-words", str(args.max_synthetic_words)])
    run_command(cmd, cwd=REPO_ROOT)
    return train_out, val_out


def iter_jsonl(paths: Iterable[Path]):
    for path in paths:
        if not path or not path.exists():
            continue
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if payload.get("word") and payload.get("points"):
                    yield payload


def write_merged(paths: Sequence[Path], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as out_file:
        for payload in iter_jsonl(paths):
            out_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
            count += 1
    return count


def summarize_manifest(path: Path) -> Dict[str, object]:
    words = set()
    rows = 0
    points = 0
    min_x = float("inf")
    max_x = float("-inf")
    min_y = float("inf")
    max_y = float("-inf")
    min_len = 10**9
    max_len = 0

    for payload in iter_jsonl([path]):
        rows += 1
        words.add(str(payload.get("word", "")))
        trace = payload.get("points", [])
        min_len = min(min_len, len(trace))
        max_len = max(max_len, len(trace))
        for point in trace:
            x = float(point.get("x", 0.0))
            y = float(point.get("y", 0.0))
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)
            points += 1

    if rows == 0:
        return {"rows": 0, "unique_words": 0}
    return {
        "rows": rows,
        "unique_words": len(words),
        "points": points,
        "trace_len_min": min_len,
        "trace_len_max": max_len,
        "x_min": round(min_x, 6),
        "x_max": round(max_x, 6),
        "y_min": round(min_y, 6),
        "y_max": round(max_y, 6),
    }


def write_training_commands(out_dir: Path, train_manifest: Path, val_manifest: Path) -> Path:
    command_path = out_dir / "runpod_training_commands.txt"
    smoke = " ".join(
        [
            "uv run python new/train_transducer_personalized.py",
            f"--train-manifest {train_manifest}",
            f"--val-manifest {val_manifest}",
            "--model-size tablet",
            "--batch-size 64",
            "--num-workers 2",
            "--fast-test",
            "--dry-run-first-batch",
        ]
    )
    full = " ".join(
        [
            "uv run python new/train_transducer_personalized.py",
            f"--train-manifest {train_manifest}",
            f"--val-manifest {val_manifest}",
            "--model-size tablet",
            "--batch-size 256",
            "--num-workers 8",
            "--learning-rate 1e-4",
            "--max-epochs 60",
            "--augment",
            "--profile production_balanced",
            "--val-profile validation_balanced",
            "--val-limit-batches 0.2",
        ]
    )
    command_path.write_text(
        "# Do not pass --normalize for the voice-typing [0,1] runtime contract.\n\n"
        "# Smoke check first:\n"
        f"{smoke}\n\n"
        "# Full first training run:\n"
        f"{full}\n",
        encoding="utf-8",
    )
    return command_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare RunPod manifests for FUTO English + Telugu synthetic training."
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--futo-dir", default=str(REPO_ROOT / "data" / "futo_raw"))
    parser.add_argument("--download-futo", action="store_true")
    parser.add_argument("--futo-repo", default="futo-org/swipe.futo.org")
    parser.add_argument("--skip-futo", action="store_true")
    parser.add_argument("--dict-path", default="data/dakshina/te_dict.json")
    parser.add_argument("--dakshina-dir", default="../dakshina")
    parser.add_argument("--lang", default="te")
    parser.add_argument("--vocab-cap", type=int, default=40000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic-val-split", type=float, default=0.1)
    parser.add_argument("--max-synthetic-words", type=int, default=0)
    parser.add_argument("--min-word-freq", type=int, default=3)
    parser.add_argument("--max-word-list", type=int, default=400000)
    parser.add_argument("--max-futo-lines", type=int, default=0)
    parser.add_argument(
        "--include-futo-test-in-val",
        action="store_true",
        help="Use FUTO test in validation. By default it is kept as a holdout file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = resolve_path(args.out_dir)
    futo_dir = resolve_path(args.futo_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    maybe_download_futo(args, futo_dir)
    futo_outputs = {} if args.skip_futo else filter_futo_files(args, futo_dir, out_dir)

    dict_path = ensure_dakshina_dict(args, out_dir)
    synthetic_train, synthetic_val = generate_synthetic(args, dict_path, out_dir)

    train_inputs: List[Path] = []
    if "train" in futo_outputs:
        train_inputs.append(futo_outputs["train"])
    train_inputs.append(synthetic_train)

    val_inputs: List[Path] = []
    if "dev" in futo_outputs:
        val_inputs.append(futo_outputs["dev"])
    if args.include_futo_test_in_val and "test" in futo_outputs:
        val_inputs.append(futo_outputs["test"])
    val_inputs.append(synthetic_val)

    train_manifest = out_dir / "train_manifest.jsonl"
    val_manifest = out_dir / "val_manifest.jsonl"
    train_count = write_merged(train_inputs, train_manifest)
    val_count = write_merged(val_inputs, val_manifest)
    if train_count == 0 or val_count == 0:
        raise SystemExit(
            f"Prepared empty manifest: train={train_count}, val={val_count}. Check inputs."
        )

    summary = {
        "coordinate_space": "zero_one",
        "train_manifest": str(train_manifest),
        "val_manifest": str(val_manifest),
        "inputs": {
            "train": [str(path) for path in train_inputs],
            "val": [str(path) for path in val_inputs],
            "futo_test_holdout": str(futo_outputs.get("test", "")),
        },
        "train": summarize_manifest(train_manifest),
        "val": summarize_manifest(val_manifest),
    }
    summary_path = out_dir / "manifest_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    commands_path = write_training_commands(out_dir, train_manifest, val_manifest)

    print("\nPrepared RunPod training data")
    print(f"  train rows : {train_count:,} -> {train_manifest}")
    print(f"  val rows   : {val_count:,} -> {val_manifest}")
    print(f"  summary    : {summary_path}")
    print(f"  commands   : {commands_path}")
    print("\nNext: run the smoke command from runpod_training_commands.txt on the pod.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())