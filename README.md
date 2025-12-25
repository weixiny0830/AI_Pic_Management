# GPU Photo Manager (CLIP-Based)

An **AI-powered, GPU-accelerated Windows desktop application**  
for automatically cleaning and organizing large local photo libraries.

This app uses **OpenAI CLIP** to accurately distinguish **screenshots vs real photos**,  
safely remove screenshots, and organize photos & videos by date.

> ✔ Fully local & offline  
> ✔ GPU accelerated (CUDA)  
> ✔ No cloud upload  
> ✔ All actions are logged and auditable  

---

## 🚀 Quick Start (Recommended)

### 1. Download & Extract
Download the release package and extract it anywhere on your computer.

You will see a folder like:

```
PhotoManager/
├── PhotoManager.exe
├── torch/
├── clip/
├── assets/
└── ...
```


> ⚠️ Do **not** remove files inside this folder.

---

### 2. Run the App
Double-click: 'PhotoManager.exe


That’s it.  
No Python installation required.

---

### 3. Choose a Photo Folder
- Select the folder that contains your photos & videos
- Click **Start**
- The app will automatically process files

---

## 🧠 What the App Does

### 📸 Images
- **Screenshots**
  - Detected by AI
  - Moved to **Windows Recycle Bin** (recoverable)
- **Real photos**
  - Organized into:
    ```
    Photos/YYYY/YYYY-MM/
    ```
- **Uncertain images**
  - Moved to:
    ```
    _AI_REVIEW/
    ```

### 🎬 Videos
- Organized into: 'Videos/YYYY/YYYY-MM/

### 📄 Other files
- Safely moved to **Recycle Bin**

---

## 🖥 GPU Acceleration

- Automatically uses **NVIDIA GPU (CUDA)** if available
- Falls back to CPU if no GPU is detected
- GPU is initialized **lazily** (app opens fast)

You will see a log message like:
'CLIP initialized on device: cuda

---

## 🧾 Logs & Audit Trail

Every run generates a CSV log file in the selected folder: '_ai_organizer_log_YYYYMMDD_HHMMSS.csv


Each file record includes:
- File path
- File type
- Screenshot probability score
- Decision (moved / recycled / review)
- Destination path
- Errors (if any)

---

## 🛑 Safety Notes

- **Nothing is permanently deleted by default**
- Screenshots & non-media files go to the **Recycle Bin**
- Always review `_AI_REVIEW` before deleting
- You can stop the process at any time

---

## ⚙ System Requirements

### Required
- Windows 10 / Windows 11
- NVIDIA GPU (recommended for speed)

### Optional
- CUDA-capable GPU for acceleration
- CPU-only mode works but is slower

---

## 📦 Distribution Notes

This app is distributed as a **self-contained folder** (onedir).

- Keep the entire `PhotoManager/` folder intact
- You may create a desktop shortcut to `PhotoManager.exe`
- Do not move the `.exe` out of its folder

---

## 📜 License
MIT License

---

## 🙏 Acknowledgements
- OpenAI CLIP
- PyTorch
- NVIDIA CUDA