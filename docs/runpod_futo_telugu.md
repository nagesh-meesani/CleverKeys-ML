# RunPod FUTO + Telugu Swipe Training

Date: 2026-05-08

This is the ready-to-run path for training a `voice-typing` compatible swipe model from:

- FUTO English human swipe traces.
- Telugu romanized synthetic swipe traces generated from the Dakshina dictionary.

## Important Clarification

The confusing terminal text that said synthetic coordinates were in `[-1,1]` came from inspecting an old stashed file with:

```bash
git show stash@{0}:scripts/generate_synthetic_swipes.py
```

That was not the committed implementation.

The committed implementation uses `[0,1]` by default because `voice-typing` currently feeds the ONNX model with Android-normalized coordinates:

```text
raw touch pixels -> x in [0,1], y in [0,1] -> 27 real gesture features -> 10 zero padding channels -> 37 total channels
```

So for the current `voice-typing` app, the rule is:

```text
Do not pass --normalize during training.
```

`--normalize` now means: convert `[0,1]` data into legacy centered `[-1,1]` coordinates. That is only for old CleverKeys-style experiments or if Android is changed later to emit centered coordinates. It is not the path for the current app.

## Why The Old Telugu Fine-Tune Failed

The earlier decoder-only / frozen-encoder attempt failed because the encoder was still an English gesture encoder. It had learned how English swipe shapes map into an internal representation, but Telugu romanized synthetic gestures changed the word distribution and gesture paths. If the encoder is frozen, the model can only adjust the decoder/joint layers. Those layers cannot repair a gesture representation that never learned the new trace distribution.

That is why validation stayed near total failure (`val_wer` around `0.9978`). The issue was not Telugu itself. The issue was training only the output side while keeping the gesture feature extractor fixed.

For the first serious run, train the full RNNT path with mixed English real swipes and Telugu synthetic swipes. Do not freeze the encoder.

## What Is Ready Now

- `new/train_transducer_personalized.py` defaults to `[0,1]` coordinates and 37 feature channels.
- `scripts/filter_and_normalize_dataset.py` filters FUTO but preserves `[0,1]` coordinates.
- `scripts/generate_synthetic_swipes.py` creates Telugu romanized synthetic swipes in `[0,1]` by default.
- `scripts/prepare_runpod_training_data.py` prepares the merged training/validation manifests and writes pod training commands.

## Before Starting The Pod

Push the current branch or otherwise make sure the pod can access commit `bf2a52d` or newer:

```bash
cd /home/nageswar/code/projects/CleverKeys-ML
git status --short --branch
git log -1 --oneline
```

Expected branch:

```text
feat/futo-te-training
```

Expected coordinate rule:

```text
No --normalize for voice-typing training.
```

## Pod Setup

On RunPod:

```bash
git clone <your-cleverkeys-ml-repo-url> CleverKeys-ML
cd CleverKeys-ML
git checkout feat/futo-te-training
uv sync
```

If the FUTO download requires Hugging Face credentials:

```bash
export HF_TOKEN=<your_huggingface_token>
```

## Prepare Training Data

If the pod can download FUTO directly from Hugging Face:

```bash
uv run python scripts/prepare_runpod_training_data.py \
  --download-futo \
  --futo-dir data/futo_raw \
  --dakshina-dir ../dakshina \
  --dict-path ../dakshina/te_dict.json \
  --lang te \
  --vocab-cap 40000
```

If you already copied FUTO files to the pod, place these files under `data/futo_raw`:

```text
data/futo_raw/train.jsonl
data/futo_raw/dev.jsonl
data/futo_raw/test.jsonl
```

Then run:

```bash
uv run python scripts/prepare_runpod_training_data.py \
  --futo-dir data/futo_raw \
  --dakshina-dir ../dakshina \
  --dict-path ../dakshina/te_dict.json \
  --lang te \
  --vocab-cap 40000
```

The output folder is:

```text
data/runpod_futo_te/
```

Important files:

```text
data/runpod_futo_te/train_manifest.jsonl
data/runpod_futo_te/val_manifest.jsonl
data/runpod_futo_te/manifest_summary.json
data/runpod_futo_te/runpod_training_commands.txt
```

## Smoke Check First

Run the smoke command written by the prep script:

```bash
cat data/runpod_futo_te/runpod_training_commands.txt
```

The first command should include:

```text
--fast-test --dry-run-first-batch
```

It should not include:

```text
--normalize
```

The dry-run output should show:

```text
feature_dim=37
```

## Full First Training Run

After the smoke check succeeds, run the full command from `runpod_training_commands.txt`.

Recommended first run settings:

```bash
uv run python new/train_transducer_personalized.py \
  --train-manifest data/runpod_futo_te/train_manifest.jsonl \
  --val-manifest data/runpod_futo_te/val_manifest.jsonl \
  --model-size tablet \
  --batch-size 256 \
  --num-workers 8 \
  --learning-rate 1e-4 \
  --max-epochs 60 \
  --augment \
  --profile production_balanced \
  --val-profile validation_balanced \
  --val-limit-batches 0.2
```

Do not freeze the encoder for this run.

## What To Watch

Early checks:

- Batch shape should have 37 features.
- Loss should be finite.
- Validation WER should move down from near-random over time.
- If WER stays near `1.0` for many validation intervals, stop and inspect the manifest summary, vocab, and coordinate ranges before burning more GPU time.

Data checks:

```bash
cat data/runpod_futo_te/manifest_summary.json
```

Coordinate ranges should be mostly `[0,1]`. Small overshoot can exist from human FUTO traces, but synthetic Telugu should be clipped to `[0,1]`.

## After Training

Use the best checkpoint produced by the training run for ONNX export, then replace the `voice-typing` model assets only after confirming:

- Encoder input is `audio_signal` with feature dimension 37.
- Decoder inputs match the existing Android runtime names.
- `model_config.json` says `feat_in: 37`.

Do not deploy a model trained with `--normalize` unless the Android `TrajectoryProcessor` is also changed to emit `[-1,1]` coordinates.