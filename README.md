# HATELENS

This repository contains the source code for the paper "Hate Is More Than What
We See: Precedent-Guided Structured Reasoning for Hateful Video Detection",
currently under peer review.

## Project Structure

```text
artifacts/  official P2C outputs and frame feature artifacts
configs/    fixed dataset, seed, and retrieval configurations
data/       dataset root and fixed train/valid/test splits
outputs/    generated embeddings, regenerated P2C outputs, and metrics
scripts/    runnable preprocessing, embedding, and reproduction entry points
src/        model, embedding, P2C Generator, and utility code
```

## Data

We do not redistribute raw datasets. Please obtain the datasets from their
original sources and follow their licenses and access terms:

- HateMM: https://github.com/hate-alert/HateMM
- MultiHateClip: https://github.com/social-ai-studio/multihateclip
- ImpliHateVid: https://github.com/videohatespeech/Implicit_Video_Hate

After downloading, place the files under `data/raw/`:

```text
data/raw/
  HateMM/
    annotation(new).json
    annotation(re).json
    frames/<Video_ID>/*.jpg
    quad/<Video_ID>/*.jpg
    audios/<Video_ID>.wav
  Multihateclip/
    English/
      annotation(new).json
      frames/<Video_ID>/*.jpg
      quad/<Video_ID>/*.jpg
      audios/<Video_ID>.wav
    Chinese/
      annotation(new).json
      frames/<Video_ID>/*.jpg
      quad/<Video_ID>/*.jpg
      audios/<Video_ID>.wav
  ImpliHateVid/
    annotation(new).json
    frames/<Video_ID>/*.jpg
    quad/<Video_ID>/*.jpg
    audios/<Video_ID>.wav
```

To facilitate reproduction, this repository includes fixed P2C Generator outputs
and frame feature artifacts under `artifacts/`. Generated embeddings and metrics
are written to `outputs/`.

## Environment

Install dependencies from the repository root:

```bash
pip install -r requirements.txt
```

## Run

Reproduce the main performance results from the repository root. This uses the
included P2C outputs and frame feature artifacts; it does not rerun the LLM.

```bash
python scripts/embed_p2c.py --dataset all
python scripts/embed_inputs.py --dataset all --only frames
python scripts/embed_inputs.py --dataset all --only text
python scripts/embed_inputs.py --dataset all --only audio
python scripts/reproduce_main.py --dataset all
```

## Optional Preprocessing

If only raw videos are available, prepare frames, quad images, and audio:

```bash
python scripts/preprocess.py --raw-video-dir /path/to/videos --dataset-dir data/raw/HateMM
```

Repeat for each dataset directory as needed.

## Optional P2C Generation

The official P2C outputs are already included in `artifacts/p2c_outputs/`.
To regenerate P2C outputs after preparing quad images:

```bash
export OPENAI_API_KEY=...
python scripts/generate_p2c.py --dataset HateMM --max-concurrent 10
```

Regenerated P2C outputs are written to `outputs/p2c_outputs/` and do not
overwrite the official artifacts. To embed regenerated outputs:

```bash
python scripts/embed_p2c.py --dataset HateMM --source generated
```
