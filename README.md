GPU Photo Manager (CLIP-Based)

An AI-powered, GPU-accelerated tool for cleaning and organizing large local photo libraries on Windows.

This project uses OpenAI CLIP (PyTorch) to distinguish screenshots vs real photos, then:
- Screenshots → Windows Recycle Bin (recoverable)
- Photos → Photos/YYYY/YYYY-MM/
- Videos → Videos/YYYY/YYYY-MM/
- Uncertain images → _AI_REVIEW/
- Non-media files → Recycle Bin

This repository contains SOURCE CODE ONLY.
You will build the Windows .exe yourself using PyInstaller.

============================================================
REPOSITORY LAYOUT
============================================================

photo_manager/
├── app.py
├── engine.py
├── PhotoManager.spec
├── version.txt
├── assets/
│   └── icon.ico
└── README.txt

============================================================
REQUIREMENTS
============================================================

Hardware:
- NVIDIA GPU recommended (CUDA)
- CPU-only mode supported (slower)

Software:
- Windows 10 / 11
- Python 3.9 or 3.10 (recommended)
- Latest NVIDIA GPU driver (for CUDA)

============================================================
SETUP (DEVELOPER)
============================================================

1) Create virtual environment

cd path\to\photo_manager
py -3.10 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip

------------------------------------------------------------

2) Install PyTorch (CUDA 11.8 example)

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

Verify:
python -c "import torch; print(torch.cuda.is_available())"

------------------------------------------------------------

3) Install remaining dependencies

pip install pillow send2trash tqdm pillow-heif opencv-python
pip install ftfy regex
pip install git+https://github.com/openai/CLIP.git
pip install pyside6 pyinstaller

============================================================
RUN FROM SOURCE
============================================================

python app.py

============================================================
BUILD WINDOWS EXE (ONEDIR)
============================================================

1) Clean previous builds

rmdir /s /q build
rmdir /s /q dist

------------------------------------------------------------

2) Build

pyinstaller PhotoManager.spec

------------------------------------------------------------

3) Output

dist\PhotoManager\PhotoManager.exe

Do NOT move the exe out of its folder.

============================================================
HOW IT WORKS
============================================================

Images:
- Screenshot → Recycle Bin
- Uncertain → _AI_REVIEW
- Photo → Photos/YYYY/YYYY-MM

Videos:
- Videos/YYYY/YYYY-MM

Other files:
- Recycle Bin

============================================================
LOGS
============================================================

_ai_organizer_log_YYYYMMDD_HHMMSS.csv

Includes per-file:
- path
- type
- CLIP score
- decision
- destination
- error (if any)

============================================================
SAFETY NOTES
============================================================

- Nothing is permanently deleted by default
- Always review _AI_REVIEW before deleting
- Screenshots can be restored from Recycle Bin

============================================================
LICENSE
============================================================

MIT License

============================================================
ACKNOWLEDGEMENTS
============================================================

OpenAI CLIP
PyTorch
NVIDIA CUDA
