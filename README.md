# SuperFly 🪰

Tools and GUIs for tracking fruit fly body orientation in a magnetic tether ("magnotether") using [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut) (DLC). This repository contains the software side of the project — training GUIs, batch video analysis, Kalman-filtered tracking previews, and utilities for syncing the community training dataset.

The training dataset (videos + annotations) lives on Hugging Face:
👉 **[jpcurrea/super-fly](https://huggingface.co/datasets/jpcurrea/super-fly)** — a living, community-maintained dataset for iteratively improving the model.

## Features

- **Training & analysis GUI** (`pyqt_gui_shell.py`) — a PyQt6 front end for the full DLC workflow: import videos, extract and label frames (via napari), train, evaluate, batch-analyze videos, and export models.
- **Robust batch processing** (`dlc_worker.py`) — videos are analyzed in subprocesses with per-file retry and logging (`dlc_logs/`), so one bad video doesn't kill a batch run.
- **Tracker preview** (`tracker_preview.py`) — interactive pyqtgraph viewer with a 2D constant-acceleration Kalman filter for smoothing and inspecting tracked points.
- **Orientation analysis** (`procrustean_analysis.py`) — Procrustes-based tools for computing body orientation angles from tracked landmarks.
- **Format conversion** (`convert.py`) — converts `.mat` and `.avi` recordings to `.mp4`.
- **Dataset sync** (`export.py`, `clean_up_huggingface.py`) — push new training videos/annotations to the Hugging Face dataset.
- **Notebooks** (`super_fly.ipynb`, `DLC_training.ipynb`) — exploratory analysis and training walkthroughs.

## Installation

### 1. Create the conda environment

The project runs DLC with PyTorch + CUDA. The simplest route is the provided environment file:

```bash
conda env create -f dlc.yml
conda activate DEEPLABCUT
```

Alternatively, install into an existing DLC environment:

```bash
pip install -r requirements.txt
```

### 2. (Optional) Hugging Face access

To upload data to the dataset repository, set your Hugging Face token as an environment variable:

```bash
# Windows (PowerShell)
$env:HF_TOKEN = "hf_..."

# Linux/macOS
export HF_TOKEN="hf_..."
```

## Usage

### Launch the GUI

```bash
conda activate DEEPLABCUT
python pyqt_gui_shell.py
```

On Windows you can also use `dlc_gui.bat` (edit the Python path inside to match your environment).

### Typical workflow

1. **Import videos** — convert raw `.mat`/`.avi` recordings with `convert.py` if needed.
2. **Label frames** — extract and annotate frames through the GUI (napari-based labeling).
3. **Train** — create a training dataset and train the network (experiment tracking via Weights & Biases).
4. **Analyze** — batch-process videos; optionally export angles and labeled videos.
5. **Preview** — inspect tracking quality with the Kalman-filtered tracker preview.
6. **Contribute** — upload new videos/annotations to the shared dataset with `export.py`.

## Repository structure

```
├── pyqt_gui_shell.py        # Main PyQt6 GUI application
├── train_dlc_gui.py         # Training/export GUI components
├── dlc_worker.py            # Subprocess worker for batch video analysis
├── tracker_preview.py       # Kalman-filtered tracking preview widget
├── procrustean_analysis.py  # Orientation/angle analysis utilities
├── video.py                 # Video handling helpers
├── convert.py               # .mat/.avi → .mp4 conversion
├── export.py                # Upload data to the Hugging Face dataset
├── clean_up_huggingface.py  # Remove stale files from the HF dataset
├── dlc.yml                  # Conda environment (PyTorch + CUDA + DLC)
├── requirements.txt         # pip dependencies
├── super_fly.ipynb          # Analysis notebook
├── DLC_training.ipynb       # Training notebook
└── SuperFly-Pablo-2025-07-29/  # DLC project (config, models, labeled data)
```

## Contributing

- **Code** — open issues and pull requests here on GitHub.
- **Data** — contribute videos and annotations via the [Hugging Face dataset repository](https://huggingface.co/datasets/jpcurrea/super-fly) (fork → add data → pull request).

## License

- **Software** (this repository): Apache License 2.0
- **Dataset** (Hugging Face): CC-BY-SA 4.0 — if you extend the dataset, you must share your additions under the same license.

## Citation

If you use this software, the dataset, or the resulting model weights in your research, please cite:

```bibtex
@dataset{superfly_dlc_magno_tracking,
  author    = {Currea, John Paul},
  title     = {DeepLabCut Behavior Tracking: A Living Community Dataset},
  year      = {2026},
  publisher = {Hugging Face Hub},
  version   = {1.0.0},
  url       = {https://huggingface.co/datasets/jpcurrea/super-fly}
}
```
