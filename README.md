# 🔍 DeepFake Detector

**Production-ready deepfake detection system using DenseNet121 (images) and GenConViT (videos).**

- ✅ Runs entirely on CPU — no GPU required
- ✅ Deterministic: same input → identical output, every time
- ✅ No simulation or random fallbacks — real model inference only
- ✅ Ensemble approach for higher accuracy
- ✅ Flask web UI + REST API + CLI tool

---

## Architecture

### Image Detection Pipeline
```
Image → DenseNet121 (60%) ─┐
      → Noise Analysis (15%) ├→ Ensemble → REAL/FAKE
      → Compression (10%)   ─┤
      → ELA Score (15%)     ─┘
```

### Video Detection Pipeline
```
Video → Frame Extraction (20 frames)
      → Face Detection (OpenCV DNN / Haar)
      → ConvNeXt Features ─┐
      → Swin Transformer   ├→ AE Error (45%) ─┐
      → Autoencoder ED     │   VAE Score (35%) ├→ REAL/FAKE
      → VAE Pathway        ─┘   Temporal (20%) ─┘
```

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Web Server
```bash
python app.py
# Open http://localhost:5000
```

### 3. CLI Usage
```bash
# Analyze an image
python detect.py photo.jpg

# Analyze a video
python detect.py video.mp4

# Verify determinism (3 identical runs)
python detect.py photo.jpg --verify-determinism

# Get JSON output
python detect.py photo.jpg --json

# Batch analyze a folder
python detect.py --batch /media/folder/ --output results.json
```

### 4. API Usage
```bash
# Upload file for analysis
curl -X POST http://localhost:5000/api/detect \
  -F "file=@photo.jpg"

# Analyze from URL
curl -X POST http://localhost:5000/api/detect/url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/photo.jpg"}'

# Health check
curl http://localhost:5000/health

# Model info
curl http://localhost:5000/api/models
```

---

## Project Structure
```
deepfake-detector/
├── app.py                        # Flask web application
├── detect.py                     # CLI inference tool
├── train.py                      # Fine-tuning script
├── requirements.txt
├── config/
│   └── model_config.yaml         # Model configuration
├── models/
│   ├── __init__.py
│   ├── image_detector.py         # DenseNet121 ensemble
│   ├── video_detector.py         # GenConViT pipeline
│   ├── model_manager.py          # Singleton model loader
│   └── utils/
│       ├── deterministic.py      # Seed management
│       ├── ela.py                # Error Level Analysis
│       ├── noise_analysis.py     # Noise & compression analysis
│       ├── face_detector.py      # Face detection (OpenCV)
│       └── frame_extractor.py    # Video frame extraction
├── weights/                      # Model weights (place here)
│   ├── densenet_deepfake.pth     # Fine-tuned image model
│   ├── convnext_base.pth         # ConvNeXt weights
│   ├── swin_base.pth             # Swin Transformer weights
│   ├── autoencoder_ed.pth        # Autoencoder weights
│   └── vae_pathway.pth           # VAE weights
├── templates/
│   └── index.html                # Web UI
└── tests/
    └── test_detector.py          # Test suite
```

---

## Fine-Tuning for Better Accuracy

The system works out-of-the-box using ImageNet pretrained weights, but accuracy improves significantly with fine-tuning on deepfake datasets.

### Dataset Sources
- **Real faces**: [FFHQ](https://github.com/NVlabs/ffhq-dataset), [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- **Fake faces**: [FaceForensics++](https://github.com/ondyari/FaceForensics), [DFDC](https://ai.facebook.com/datasets/dfdc/), [Celeb-DF](https://github.com/yuezunli/celeb-deepfakeforensics)
- **AI-generated**: StyleGAN2 outputs, Stable Diffusion faces

### Training
```bash
# Organize data:
# data/real/  ← real face images
# data/fake/  ← fake/generated images

python train.py \
  --real-dir data/real \
  --fake-dir data/fake \
  --output weights/densenet_deepfake.pth \
  --batch-size 16 \
  --phase1-epochs 10 \
  --phase2-epochs 20 \
  --phase3-epochs 10
```

Training phases:
- **Phase 1**: Frozen backbone, train head only (10 epochs, lr=1e-3)
- **Phase 2**: Unfreeze last 20 layers (20 epochs, lr=1e-4)
- **Phase 3**: Full fine-tune (10 epochs, lr=1e-5)

---

## Running Tests
```bash
python tests/test_detector.py
```

Tests verify:
- ✅ Determinism (3 runs → identical output)
- ✅ No random fallbacks
- ✅ ELA, noise, compression analysis
- ✅ Frame extraction uniformity
- ✅ Error handling (conservative REAL default)
- ✅ Performance within time limits

---

## API Response Format

### Image
```json
{
  "prediction": "FAKE",
  "confidence": 0.8234,
  "fake_probability": 0.8234,
  "analysis_time": 1.23,
  "image_size": "1024x768",
  "scores": {
    "densenet121": 0.91,
    "noise_analysis": 0.43,
    "compression_artifacts": 0.67,
    "ela_score": 0.55
  },
  "ensemble_weights": {"densenet": 0.60, "noise": 0.15, "compression": 0.10, "ela": 0.15},
  "model": "DenseNet121 + Signal Analysis Ensemble",
  "file_type": "image",
  "error": null,
  "success": true
}
```

### Video
```json
{
  "prediction": "REAL",
  "confidence": 0.7812,
  "fake_probability": 0.2188,
  "analysis_time": 14.5,
  "frames_analyzed": 20,
  "video_metadata": {"total_frames": 600, "fps": 30.0, "width": 1920, "height": 1080},
  "scores": {
    "autoencoder_ed": 0.18,
    "vae_pathway": 0.24,
    "temporal_consistency": 0.22
  },
  "per_frame_scores": [0.21, 0.19, 0.23, ...],
  "model": "GenConViT (ConvNeXt + Swin + AE + VAE)",
  "file_type": "video",
  "error": null,
  "success": true
}
```

---

## Performance Expectations (CPU)

| Task | Expected Time | RAM |
|------|--------------|-----|
| Image analysis | 0.5–2s | 300–600MB |
| Video analysis (20 frames) | 10–25s | 800MB–1.5GB |
| ELA + Noise | 0.1–0.2s | 50–100MB |

---

## Determinism Guarantee

The system uses comprehensive seed management:
```python
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
numpy.random.seed(42)
random.seed(42)
```

All models run in `eval()` mode with `torch.no_grad()`. The VAE uses the mean (not sampled latent) during inference, ensuring 100% reproducible outputs.

---

## Notes

- **Without fine-tuned weights**: The system uses ImageNet pretrained features. The neural network component may not be well-calibrated for deepfake detection, but ELA, noise, and compression analyses still run correctly.
- **With fine-tuned weights**: Place `densenet_deepfake.pth` in the `weights/` folder. The system automatically loads them on startup.
- **Error handling**: All errors return `{"prediction": "REAL", "confidence": 0.5}` — conservative and never random.
