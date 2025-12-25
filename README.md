GPU-Accelerated Local Photo Manager (CLIP-Based)

An AI-powered tool that uses GPU acceleration to automatically manage and organize
large local photo libraries on Windows.

This tool leverages OpenAI CLIP to accurately distinguish screenshots from real photos,
safely remove screenshots, and organize real photos by date. All operations are fully
logged and auditable, and uncertain images are separated for manual review.

------------------------------------------------------------
FEATURES
------------------------------------------------------------

- CLIP-based screenshot detection with 95–99% accuracy
- GPU acceleration (100–300 images per second depending on GPU)
- Screenshots are moved to the Windows Recycle Bin (recoverable)
- Real photos are automatically organized by Year / Month
- Uncertain images are placed into a dedicated review folder
- Full audit and processing logs
- Fully offline, local-only processing (no cloud upload)

------------------------------------------------------------
HOW IT WORKS
------------------------------------------------------------

1. Images are loaded and encoded using CLIP on GPU
2. Each image is classified as:
   - Screenshot
   - Real photo
   - Uncertain
3. Actions are applied:
   - Screenshots are moved to the Recycle Bin
   - Real photos are organized into YYYY/MM folders
   - Uncertain images are moved to a review directory
4. All actions are recorded in log files for traceability

------------------------------------------------------------
SYSTEM REQUIREMENTS
------------------------------------------------------------

Hardware:
- NVIDIA GPU (GTX 1060 or newer, RTX series recommended)
- Minimum 4 GB VRAM (6 GB or more recommended)

Software:
- Windows 10 or Windows 11
- Python 3.9 or Python 3.10
  Do NOT use Python 3.12 or newer
- Latest NVIDIA GPU driver

------------------------------------------------------------
INSTALLATION (SAFE STEP-BY-STEP)
------------------------------------------------------------

1. Verify NVIDIA driver installation

Run the following command in PowerShell or Command Prompt:

nvidia-smi

If your GPU information is displayed, the driver is correctly installed.

------------------------------------------------------------

2. Install PyTorch with CUDA support (CUDA 11.8)

Use this exact command:

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

Verify GPU availability:

import torch
print(torch.cuda.is_available())

The expected output is:
True

------------------------------------------------------------

3. Install remaining dependencies

pip install pillow send2trash tqdm pillow-heif opencv-python
pip install ftfy regex
pip install git+https://github.com/openai/CLIP.git

------------------------------------------------------------
USAGE
------------------------------------------------------------

Basic usage:

python main.py --input "D:/Photos"

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