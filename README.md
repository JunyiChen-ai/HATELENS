# HATELENS

Source code for "Hate Is More Than What We See: Precedent-Guided Structured
Reasoning for Hateful Video Detection" (Findings of EMNLP 2026).

## Setup

```bash
pip install -r requirements.txt
```

Python 3.12 and a CUDA GPU are recommended. Reference hardware: NVIDIA RTX
5090, PyTorch 2.8, CUDA 12.8. The scripts download `bert-base-uncased` from
Hugging Face on first run.

## Reproduce Main Results

No raw data is needed: the official P2C outputs and text/frame/audio feature
vectors ship under `artifacts/`.

```bash
python scripts/embed_p2c.py --dataset all
python scripts/embed_inputs.py --dataset all --only frames
python scripts/embed_inputs.py --dataset all --only text
python scripts/embed_inputs.py --dataset all --only audio
python scripts/reproduce_main.py --dataset all --out outputs/main_results.json
```

Metrics (ACC/F1/P/R) are printed per dataset and written to the `--out` path.
Small deviations from the reported numbers can occur on different GPU models
or library versions.

## Regenerating from Raw Data (Optional)

Annotation files (`Video_ID`, `Title`, `Transcript`, `Label` per video) are
included under `data/raw/`. Videos, frames, and audio are not redistributed;
obtain them from the original sources:

- HateMM: https://github.com/hate-alert/HateMM
- MultiHateClip: https://github.com/social-ai-studio/multihateclip
- ImpliHateVid: https://github.com/videohatespeech/Implicit_Video_Hate

```text
data/raw/
  HateMM/
    annotation(new).json
    frames/<Video_ID>/*.jpg
    quad/<Video_ID>/*.jpg
    audios/<Video_ID>.wav
  Multihateclip/
    English/   (same layout)
    Chinese/   (same layout)
  ImpliHateVid/  (same layout)
```

To regenerate a modality from raw data, delete the matching
`artifacts/*_embeddings/*.npz` and rerun `embed_inputs.py`.

Prepare frames, quad images, and audio from raw videos (requires `ffmpeg`):

```bash
python scripts/preprocess.py --raw-video-dir /path/to/videos --dataset-dir data/raw/HateMM
```

Regenerate P2C outputs (written to `outputs/p2c_outputs/`, the official
artifacts are not overwritten):

```bash
export OPENAI_API_KEY=...
python scripts/generate_p2c.py --dataset HateMM --max-concurrent 10
python scripts/embed_p2c.py --dataset HateMM --source generated
```

## Acknowledgement

This project's development referenced
[HVGuard](https://github.com/yihengjingWHU/HVGuard) (Jing et al., EMNLP 2025).
