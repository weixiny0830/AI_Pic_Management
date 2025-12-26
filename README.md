# GPU Photo Manager (CLIP-Based)

An **AI-powered, GPU-accelerated** tool for cleaning and organizing large local photo libraries on **Windows**.

This project uses **OpenAI CLIP** (PyTorch) to distinguish **screenshots vs real photos**, then:
- Screenshots → **Windows Recycle Bin** (recoverable)
- Photos → `Photos/YYYY/YYYY-MM/`
- Videos → `Videos/YYYY/YYYY-MM/`
- Uncertain images → `_AI_REVIEW/`
- Non-media files → **Recycle Bin**

✅ Fully local/offline (no cloud upload)  
✅ GPU acceleration via CUDA (when available)  
✅ CSV audit log with per-file scores and decisions  

> **Note:** This GitHub repo contains **source code only**.  
> You will build the Windows `.exe` yourself using **PyInstaller**.

---

## Repository Layout
```
photo_manager/
├── app.py
├── engine.py
├── PhotoManager.spec
├── version.txt
├── assets/
│ └── icon.ico
└── README.md
```

---

## Requirements

### Hardware
- **NVIDIA GPU** recommended (CUDA)
- CPU-only is supported but slower

### Software
- **Windows 10 / 11**
- **Python 3.9 or 3.10** (recommended)
  - Avoid Python 3.12+ for maximum PyTorch + PyInstaller compatibility
- Latest NVIDIA driver (for GPU mode)

---

## Setup (Developer)

### 1) Create a virtual environment (recommended)

PowerShell:

```powershell
cd C:\path\to\photo_manager
py -3.10 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
```

### 2) Install PyTorch (CUDA 11.8 example)

If you want GPU acceleration, install the CUDA build:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Verify:
```
python -c "import torch; print(torch.cuda.is_available())"
```
Expected output:
- True on a CUDA-capable machine with drivers installed
- False is fine (CPU mode)

### 3) Install remaining dependencies
```
pip install pillow send2trash tqdm pillow-heif opencv-python
pip install ftfy regex
pip install git+https://github.com/openai/CLIP.git
pip install pyside6 pyinstaller
```

- Run From Source (Dev)
'python app.py

- Build the Windows EXE (onedir recommended)
This project uses onedir packaging (a folder containing PhotoManager.exe + dependencies).
This is the most reliable option for PyTorch apps.

### 1) Clean previous builds
````
rmdir /s /q build
rmdir /s /q dist
````

### 2) Build using the spec file
'pyinstaller PhotoManager.spec

### 3) Output location
After build completes:
```
dist/
└── PhotoManager/
    ├── PhotoManager.exe
    ├── torch/
    ├── clip/
    ├── PySide6/
    └── ...
```
Run:
```
dist\PhotoManager\PhotoManager.exe
```
⚠️ Do not move PhotoManager.exe out of the PhotoManager/ folder.

## How It Works
### Images
- Screenshot (score ≥ screenshot threshold)
  → Recycle Bin
- Uncertain (review threshold ≤ score < screenshot threshold)
  → _AI_REVIEW/
- Photo (score < review threshold)
  → Photos/YYYY/YYYY-MM/

### Videos
- → Videos/YYYY/YYYY-MM/

### Other files
- → Recycle Bin

### Logs (CSV Audit Trail)
Each run generates a CSV log in the selected root folder:
```
_ai_organizer_log_YYYYMMDD_HHMMSS.csv
```
Columns include:
- file_path
- file_type
- clip_screenshot_score
- decision
- dest_path
- error

## Safety Notes
- Nothing is permanently deleted by default
- Screenshots & non-media files go to the Windows Recycle Bin
- Always review _AI_REVIEW before deleting anything

## Troubleshooting
### App starts slowly
First image classification triggers lazy initialization of CLIP + CUDA.
This is expected and keeps UI startup fast.
### GPU not used
Confirm GPU driver:
'nvidia-smi
Then confirm PyTorch CUDA:
'python -c "import torch; print(torch.cuda.is_available())"

### PyInstaller missing-module errors
PyTorch packaging can require small spec tweaks depending on your torch version.
Open an issue and include the full traceback + your pip freeze.

## License
MIT License

## Acknowledgements
- OpenAI CLIP
- PyTorch
- NVIDIA CUDA