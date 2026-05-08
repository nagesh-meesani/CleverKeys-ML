#!/usr/bin/env python3
"""Generate synthetic romanized swipe traces from a word dictionary.

Input dictionaries are JSON objects of ``word -> score`` such as the Dakshina
outputs produced by ``dakshina/build_lang_dict.py``. The output is JSONL using
the trainer manifest contract:

    {"word": "cheppu", "points": [{"x": 0.1, "y": 0.2, "t": 0.0}, ...]}

By default coordinates are emitted in [0,1] to match the current voice-typing
Android runtime and FUTO traces. Use ``--coordinate-space centered`` only for
legacy experiments that train on [-1,1] coordinates.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


ROWS = ("qwertyuiop", "asdfghjkl", "zxcvbnm")
WORD_RE = re.compile(r"[a-z]{2,20}")


def build_key_centers(coordinate_space: str) -> Dict[str, Tuple[float, float]]:
    centers: Dict[str, Tuple[float, float]] = {}
    for row_index, row in enumerate(ROWS):
        for col_index, char in enumerate(row):
            x = (col_index + 0.5) / 10.0
            y = (row_index + 0.5) / 3.0
            if coordinate_space == "centered":
                x = x * 2.0 - 1.0
                y = y * 2.0 - 1.0
            centers[char] = (x, y)
    return centers


def clean_word(word: str) -> str:
    return "".join(ch for ch in word.lower() if "a" <= ch <= "z")


def load_words(path: Path, max_words: int = 0) -> List[Tuple[str, float]]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object word->score in {path}")

    words: List[Tuple[str, float]] = []
    seen = set()
    for raw_word, raw_score in sorted(payload.items(), key=lambda item: -float(item[1])):
        word = clean_word(str(raw_word))
        if word in seen or not WORD_RE.fullmatch(word):
            continue
        seen.add(word)
        words.append((word, float(raw_score)))
        if max_words > 0 and len(words) >= max_words:
            break
    return words


def word_to_controls(
    word: str, centers: Dict[str, Tuple[float, float]]
) -> List[Tuple[float, float]]:
    controls: List[Tuple[float, float]] = []
    for char in word:
        center = centers.get(char)
        if center is None:
            continue
        if not controls or controls[-1] != center:
            controls.append(center)
    return controls


def catmull_rom_segment(
    p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, count: int
) -> np.ndarray:
    t = np.linspace(0.0, 1.0, count, endpoint=False, dtype=np.float32)
    t2 = t * t
    t3 = t2 * t
    m1 = 0.5 * (p2 - p0)
    m2 = 0.5 * (p3 - p1)
    return (
        (2.0 * t3 - 3.0 * t2 + 1.0)[:, None] * p1
        + (t3 - 2.0 * t2 + t)[:, None] * m1
        + (-2.0 * t3 + 3.0 * t2)[:, None] * p2
        + (t3 - t2)[:, None] * m2
    )


def spline_path(controls: Sequence[Tuple[float, float]], point_count: int) -> np.ndarray:
    points = np.asarray(controls, dtype=np.float32)
    if len(points) == 1:
        return np.repeat(points, point_count, axis=0)

    start = 2.0 * points[0] - points[1]
    end = 2.0 * points[-1] - points[-2]
    extended = np.vstack([start, points, end])
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total_length = float(segment_lengths.sum())
    if total_length <= 0.0:
        return np.repeat(points[:1], point_count, axis=0)

    base_per_segment = np.maximum(
        3,
        np.round(segment_lengths / total_length * max(point_count - 1, 1)).astype(int),
    )
    delta = max(point_count - 1, 1) - int(base_per_segment.sum())
    base_per_segment[int(np.argmax(base_per_segment))] += delta
    base_per_segment = np.maximum(base_per_segment, 2)

    pieces = []
    for index, count in enumerate(base_per_segment):
        pieces.append(
            catmull_rom_segment(
                extended[index],
                extended[index + 1],
                extended[index + 2],
                extended[index + 3],
                int(count),
            )
        )
    pieces.append(points[-1:])
    path = np.vstack(pieces)
    if len(path) > point_count:
        path = path[:point_count]
        path[-1] = points[-1]
    return path.astype(np.float32)


def gesture_times(point_count: int, total_ms: float) -> np.ndarray:
    u = np.linspace(0.0, 1.0, point_count, dtype=np.float32)
    velocity = 0.25 + 0.75 * np.sin(math.pi * u)
    cumulative = np.cumsum(velocity)
    cumulative -= cumulative[0]
    if cumulative[-1] > 0:
        cumulative = cumulative / cumulative[-1] * total_ms
    return cumulative.astype(np.float32)


def variants_for_score(score: float) -> int:
    if score >= 6.5:
        return 8
    if score >= 5.2:
        return 5
    return 3


def generate_swipe(
    word: str,
    score: float,
    centers: Dict[str, Tuple[float, float]],
    coordinate_space: str,
    rng: random.Random,
    np_rng: np.random.Generator,
) -> Optional[Dict[str, object]]:
    controls = word_to_controls(word, centers)
    if not controls:
        return None

    distance = 0.0
    for left, right in zip(controls, controls[1:]):
        distance += math.dist(left, right)

    base_points = 28 + len(controls) * 6 + int(distance * 24)
    point_count = int(np.clip(base_points + rng.randint(-8, 12), 24, 160))
    total_ms = float(np.clip(180.0 + len(word) * rng.uniform(45.0, 80.0), 180.0, 1400.0))

    xy = spline_path(controls, point_count)
    xy += np_rng.normal(0.0, rng.uniform(0.003, 0.018), size=xy.shape).astype(np.float32)

    if len(xy) > 4 and rng.random() < 0.55:
        xy[0] += np_rng.normal(0.0, 0.012, size=(2,)).astype(np.float32)
        xy[-1] += np_rng.normal(0.0, 0.012, size=(2,)).astype(np.float32)

    low, high = (-1.0, 1.0) if coordinate_space == "centered" else (0.0, 1.0)
    xy = np.clip(xy, low, high)
    times = gesture_times(len(xy), total_ms)

    points = [
        {
            "x": round(float(xy[index, 0]), 6),
            "y": round(float(xy[index, 1]), 6),
            "t": round(float(times[index]), 3),
        }
        for index in range(len(xy))
    ]
    return {"word": word, "points": points, "source": "synthetic", "score": score}


def split_words(
    words: Sequence[Tuple[str, float]], val_split: float, seed: int
) -> Tuple[set[str], set[str]]:
    shuffled = list(words)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * val_split)) if shuffled else 0
    val_words = {word for word, _ in shuffled[:val_count]}
    train_words = {word for word, _ in shuffled[val_count:]}
    return train_words, val_words


def write_samples(
    words: Iterable[Tuple[str, float]],
    train_words: set[str],
    val_words: set[str],
    args: argparse.Namespace,
) -> Tuple[int, int]:
    centers = build_key_centers(args.coordinate_space)
    train_path = Path(args.out_train)
    val_path = Path(args.out_val)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    val_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed + 1)
    np_rng = np.random.default_rng(args.seed + 2)
    train_count = 0
    val_count = 0

    with train_path.open("w", encoding="utf-8") as train_file, val_path.open(
        "w", encoding="utf-8"
    ) as val_file:
        for word, score in words:
            output_file = val_file if word in val_words else train_file
            variants = variants_for_score(score)
            for _ in range(variants):
                sample = generate_swipe(
                    word, score, centers, args.coordinate_space, rng, np_rng
                )
                if sample is None:
                    continue
                output_file.write(json.dumps(sample, ensure_ascii=False) + "\n")
                if word in val_words:
                    val_count += 1
                elif word in train_words:
                    train_count += 1
    return train_count, val_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic swipe JSONL from a romanized dictionary."
    )
    parser.add_argument("--dict-path", required=True, help="Path to *_dict.json")
    parser.add_argument("--out-train", required=True, help="Training JSONL output")
    parser.add_argument("--out-val", required=True, help="Validation JSONL output")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-words", type=int, default=0, help="Limit words for tests")
    parser.add_argument(
        "--coordinate-space",
        choices=("zero_one", "centered"),
        default="zero_one",
        help="Output coordinates: zero_one=[0,1], centered=[-1,1]",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    words = load_words(Path(args.dict_path), max_words=args.max_words)
    if not words:
        raise SystemExit(f"No valid romanized words found in {args.dict_path}")

    train_words, val_words = split_words(words, args.val_split, args.seed)
    train_count, val_count = write_samples(words, train_words, val_words, args)

    print(f"Loaded words       : {len(words):,}")
    print(f"Train word split   : {len(train_words):,}")
    print(f"Validation split   : {len(val_words):,}")
    print(f"Train samples      : {train_count:,} -> {args.out_train}")
    print(f"Validation samples : {val_count:,} -> {args.out_val}")
    print(f"Coordinate space   : {args.coordinate_space}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
