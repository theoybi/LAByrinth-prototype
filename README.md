# LAByrinth CV MVP

LAByrinth is a computer-vision research prototype that turns images into structured measurements for downstream AI feedback.

The current pottery demo uses:

1. **Grounding DINO** to detect configured parts of the object.
2. **SAM 2** to create segmentation masks.
3. A mask-to-JSON converter to extract measurements such as proportions, curves, and handle geometry.

## Setup

Python 3.12 is recommended.

```bash
python -m venv .venv
```

Activate the environment:

**Windows PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux**

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Add images

Set the input folder near the top of `main.py`:

```python
INPUT_DIR = Path("data/input") / "experiment3"
```

Place `.jpg` or `.jpeg` images in that folder.

## Run

```bash
python main.py
```

The first run downloads the Grounding DINO and SAM 2 model files from Hugging Face. A CUDA-capable GPU is recommended, although the code can fall back to CPU.

## Outputs

Results are saved under:

```text
outputs/<batch_name>/
```

Each image receives its own folder containing:

- a concise measurement JSON;
- a detailed run report;
- the original image;
- detection and mask overview images; and
- raw segmentation masks.

The batch folder also contains `batch_report.json`.

## Main settings

The main settings are grouped near the top of `main.py`, including:

- `CRITERIA_PACK`
- `INPUT_DIR`
- model IDs
- detection thresholds
- raw-mask saving
- debug-image saving

## Notes

- The current measurements come from a single 2D image and do not represent true 3D geometry.
- Detection or segmentation errors will affect the generated JSON.
- This is an experimental research prototype, not a clinically validated system.