# GPU-Accelerated Local Photo Manager (CLIP-Based)

An **AI-powered, GPU-accelerated** tool for automatically cleaning and organizing  
**large local photo libraries on Windows**.

This tool leverages **OpenAI CLIP** to accurately distinguish **screenshots vs real photos**,  
safely remove screenshots, and organize real photos by date.

All operations are **fully logged and auditable**, and uncertain images are separated  
for manual review.

---

## Features

- CLIP-based screenshot detection (**95–99% accuracy**)
- GPU acceleration (**100–300 images/sec**, depending on GPU)
- Screenshots moved to the **Windows Recycle Bin** (recoverable)
- Real photos automatically organized by **Year / Month**
- Uncertain images placed into a dedicated **review folder**
- Full **audit & processing logs (CSV)**
- Fully **offline, local-only** processing (no cloud upload)

---

## How It Works

1. Images are loaded and encoded using **CLIP on GPU**
2. Each image is classified as:
   - Screenshot
   - Real photo
   - Uncertain
3. Actions are applied:
   - Screenshots → **Recycle Bin**
   - Real photos → `Photos/YYYY/YYYY-MM`
   - Uncertain images → review directory
4. All actions are recorded in log files for traceability

---

## System Requirements

### Hardware

- NVIDIA GPU (**GTX 1060 or newer**, RTX series recommended)
- Minimum **4 GB VRAM** (6 GB+ recommended)

### Software

- Windows 10 or Windows 11
- **Python 3.9 or Python 3.10**
  - **Do NOT use Python 3.12 or newer**
- Latest NVIDIA GPU driver

---

## Installation (Safe Step-by-Step)

### 1. Verify NVIDIA Driver Installation

Run the following command in **PowerShell** or **Command Prompt**:

```powershell
nvidia-smi

If your GPU information is displayed, the driver is correctly installed.

---

## 2. Install PyTorch with CUDA Support (CUDA 11.8)

Use **this exact command**:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

Verify GPU availability:

```import torch
print(torch.cuda.is_available())

The expected output is:
True

------------------------------------------------------------

3. Install remaining dependencies

```pip install pillow send2trash tqdm pillow-heif opencv-python
pip install ftfy regex
pip install git+https://github.com/openai/CLIP.git

------------------------------------------------------------
USAGE
------------------------------------------------------------

Basic usage:

```python main.py --input "D:/Photos"

Optional arguments:
- --review-dir review/
- --log-file process.log
- --dry-run (no file operations, classification only)

------------------------------------------------------------
OUTPUT STRUCTURE
------------------------------------------------------------

Photos/
├── 2021/
│   ├── 01/
│   └── 02/
├── 2022/
├── review/
├── screenshots.log
└── audit.log

------------------------------------------------------------
PERFORMANCE
------------------------------------------------------------

Approximate throughput (varies by image resolution and disk speed):

RTX 3060: 200–300 images per second
RTX 2060: 120–180 images per second
GTX 1060: 80–120 images per second

------------------------------------------------------------
NOTES AND WARNINGS
------------------------------------------------------------

- Screenshots are moved to the Windows Recycle Bin and can be restored
- No files are permanently deleted by default
- It is strongly recommended to run with --dry-run first on new datasets

------------------------------------------------------------
LICENSE
------------------------------------------------------------

MIT License

------------------------------------------------------------
ACKNOWLEDGEMENTS
------------------------------------------------------------

OpenAI CLIP
PyTorch
NVIDIA CUDA