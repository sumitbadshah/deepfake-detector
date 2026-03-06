"""
Deepfake Detection System - Flask Web Application.

Production-ready API server for real/fake media detection.
Uses DenseNet121 for images and GenConViT for videos.
All inference is deterministic and CPU-optimized.
"""
import os
import sys
import uuid
import logging
import time
from pathlib import Path
from typing import Optional

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask
app = Flask(__name__)
CORS(app)

# Configuration
app.config.update(
    MAX_CONTENT_LENGTH=500 * 1024 * 1024,  # 500MB max file size
    UPLOAD_FOLDER='uploads',
    ALLOWED_IMAGE_EXTENSIONS={'png', 'jpg', 'jpeg', 'webp', 'bmp', 'tiff'},
    ALLOWED_VIDEO_EXTENSIONS={'mp4', 'avi', 'mov', 'mkv', 'webm', 'flv'},
    SECRET_KEY=os.environ.get('SECRET_KEY', 'deepfake-detector-dev-key'),
)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('weights', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# Lazy-loaded detectors (initialized on first request)
_image_detector = None
_video_detector = None


def get_image_detector():
    global _image_detector
    if _image_detector is None:
        from models.image_detector import ImageDetector
        from models.utils.deterministic import set_deterministic
        set_deterministic(42)
        _image_detector = ImageDetector()
        logger.info("Image detector initialized")
    return _image_detector


def get_video_detector():
    global _video_detector
    if _video_detector is None:
        from models.video_detector import VideoDetector
        from models.utils.deterministic import set_deterministic
        set_deterministic(42)
        _video_detector = VideoDetector(num_frames=32)  # increased for better accuracy
        logger.info("Video detector initialized")
    return _video_detector


def allowed_file(filename: str, file_type: str) -> bool:
    """Check if file extension is allowed."""
    ext = Path(filename).suffix.lower().lstrip('.')
    if file_type == 'image':
        return ext in app.config['ALLOWED_IMAGE_EXTENSIONS']
    elif file_type == 'video':
        return ext in app.config['ALLOWED_VIDEO_EXTENSIONS']
    return False


def is_image(filename: str) -> bool:
    return allowed_file(filename, 'image')


def is_video(filename: str) -> bool:
    return allowed_file(filename, 'video')


def get_system_info() -> dict:
    """Get current system resource usage."""
    mem = psutil.virtual_memory()
    return {
        'memory_used_gb': round(mem.used / 1e9, 2),
        'memory_total_gb': round(mem.total / 1e9, 2),
        'memory_percent': mem.percent,
        'cpu_percent': psutil.cpu_percent(interval=0.1),
    }


# ============================================================
# Routes
# ============================================================

@app.route('/')
def index():
    """Serve the main web UI."""
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'Deepfake Detection System',
        'version': '2.0.0',
        'improvements': [
            '4-model neural ensemble (ViT×3 + SigLIP)',
            'Test-Time Augmentation (TTA, 3 crops)',
            'Temperature-scaled softmax (T=0.7)',
            'Continuous signal scoring (vs discrete buckets)',
            'Multi-quality ELA (3 quality levels)',
            'Gradient consistency analysis',
            '32-frame video analysis (vs 20)',
            'Z-score AE/VAE normalisation',
            'Cross-modal image check in video pipeline',
        ],
        'system': get_system_info(),
        'timestamp': time.time()
    })


@app.route('/api/detect', methods=['POST'])
def detect():
    """
    Main detection endpoint. Accepts image or video uploads.
    
    Returns JSON with prediction, confidence, and detailed scores.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided', 'success': False}), 400

    file = request.files['file']

    if not file.filename:
        return jsonify({'error': 'No file selected', 'success': False}), 400

    filename = secure_filename(file.filename)

    if not filename:
        return jsonify({'error': 'Invalid filename', 'success': False}), 400

    # Determine file type
    if is_image(filename):
        file_type = 'image'
    elif is_video(filename):
        file_type = 'video'
    else:
        ext = Path(filename).suffix
        supported = ', '.join(
            app.config['ALLOWED_IMAGE_EXTENSIONS'] |
            app.config['ALLOWED_VIDEO_EXTENSIONS']
        )
        return jsonify({
            'error': f'Unsupported file type: {ext}. Supported: {supported}',
            'success': False
        }), 415

    # Save with unique ID to prevent collisions
    unique_id = str(uuid.uuid4())[:8]
    safe_name = f"{unique_id}_{filename}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)

    try:
        file.save(file_path)
        logger.info(f"Saved {file_type}: {safe_name}")

        # Run detection
        if file_type == 'image':
            detector = get_image_detector()
            result = detector.analyze(file_path)
        else:
            detector = get_video_detector()
            result = detector.analyze(file_path)

        # Add metadata
        result['file_type'] = file_type
        result['filename'] = filename
        result['success'] = True

        if result.get('error'):
            result['success'] = False
            logger.warning(f"Detection returned error: {result['error']}")

        logger.info(
            f"Detection result: {result['prediction']} "
            f"({result['confidence']:.2%} confidence) "
            f"in {result['analysis_time']:.2f}s"
        )

        return jsonify(result)

    except Exception as e:
        logger.error(f"Detection failed: {e}", exc_info=True)
        return jsonify({
            'error': str(e),
            'success': False,
            'prediction': 'REAL',
            'confidence': 0.5,
        }), 500

    finally:
        # Clean up uploaded file
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.warning(f"Could not clean up file {file_path}: {e}")


@app.route('/api/detect/url', methods=['POST'])
def detect_url():
    """
    Detect from URL (image only for safety).
    Accepts JSON: {"url": "https://..."}
    """
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({'error': 'JSON body with "url" field required', 'success': False}), 400

    url = data['url']

    # Basic URL validation
    if not url.startswith(('http://', 'https://')):
        return jsonify({'error': 'Invalid URL (must start with http:// or https://)', 'success': False}), 400

    try:
        import urllib.request
        import tempfile

        # Download image
        unique_id = str(uuid.uuid4())[:8]
        suffix = Path(url.split('?')[0]).suffix or '.jpg'
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}{suffix}")

        headers = {'User-Agent': 'DeepfakeDetector/1.0'}
        req = urllib.request.Request(url, headers=headers)

        with urllib.request.urlopen(req, timeout=30) as response:
            with open(temp_path, 'wb') as f:
                f.write(response.read())

        # Detect
        detector = get_image_detector()
        result = detector.analyze(temp_path)
        result['file_type'] = 'image'
        result['source_url'] = url
        result['success'] = True

        return jsonify(result)

    except Exception as e:
        logger.error(f"URL detection failed: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

    finally:
        try:
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass


@app.route('/api/models', methods=['GET'])
def model_info():
    """Return information about loaded models."""
    from models.model_manager import ModelManager, PRETRAINED_ENSEMBLE_WEIGHTS
    manager = ModelManager()
    return jsonify({
        'loaded_models': manager.get_loaded_models(),
        'load_status': manager.get_load_status(),
        'image_model': 'ViT × 3 + SigLIP + Signal Analysis Ensemble (TTA, Temperature Scaled)',
        'video_model': 'GenConViT v2 (ConvNeXt + Swin + AE + VAE + Image-Check, 32 frames)',
        'device': 'CPU',
        'image_neural_models': {
            'vit_v2':      {'accuracy': '92.1%', 'weight': PRETRAINED_ENSEMBLE_WEIGHTS.get('vit_v2', 0)},
            'siglip_v1':   {'accuracy': '94.4%', 'weight': PRETRAINED_ENSEMBLE_WEIGHTS.get('siglip_v1', 0)},
            'vit_wvolf':   {'accuracy': '98.7%', 'weight': PRETRAINED_ENSEMBLE_WEIGHTS.get('vit_wvolf', 0)},
            'vit_dima806': {'accuracy': '99%+',  'weight': PRETRAINED_ENSEMBLE_WEIGHTS.get('vit_dima806', 0)},
        },
        'image_ensemble_weights': {
            'neural_models': 0.70,
            'noise_analysis': 0.10,
            'compression_artifacts': 0.08,
            'ela_score': 0.07,
            'frequency_analysis': 0.05,
        },
        'image_tta': {'crops': 3, 'temperature': 0.7},
        'video_ensemble_weights': {
            'autoencoder_ed': 0.40,
            'vae_pathway': 0.30,
            'temporal': 0.20,
            'image_check': 0.10,
        },
        'video_frames': 32,
    })


@app.route('/api/debug', methods=['POST'])
def debug_image():
    """
    Debug endpoint — uploads an image and returns EVERY individual sub-score.

    Shows: per-model neural predictions, signal analysis scores, and which
    models failed to load. Use this to diagnose wrong predictions.

    curl -X POST http://localhost:5000/api/debug -F "file=@photo.jpg"
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if not file.filename:
        return jsonify({'error': 'No file selected'}), 400

    filename  = secure_filename(file.filename)
    unique_id = str(uuid.uuid4())[:8]
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_{filename}")

    try:
        file.save(file_path)

        from models.model_manager import ModelManager, HF_MODELS, PRETRAINED_ENSEMBLE_WEIGHTS
        from models.utils.ela import ela_score
        from models.utils.noise_analysis import combined_signal_analysis, ai_smoothness_score
        from PIL import Image
        import torch
        import torch.nn.functional as F

        manager = ModelManager()
        image   = Image.open(file_path).convert('RGB')

        # ── Neural models ──────────────────────────────────────────────────────
        model_results = {}
        for key in HF_MODELS:
            model, processor, label_indices = manager.get_pretrained_model(key)
            status  = manager.get_load_status().get(key, 'never_tried')
            if model is None:
                model_results[key] = {'status': status, 'fake_prob': None, 'verdict': 'skipped'}
                continue

            try:
                fake_idx, real_idx = label_indices
                inputs = processor(images=image, return_tensors='pt')
                inputs = {k: v.to('cpu') for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs   = F.softmax(outputs.logits, dim=-1).squeeze(0)

                n = probs.shape[0]
                safe_fake = min(fake_idx, n - 1)
                safe_real = min(real_idx, n - 1)
                fake_prob = float(probs[safe_fake].item())
                real_prob = float(probs[safe_real].item())
                id2label  = getattr(model.config, 'id2label', {})

                model_results[key] = {
                    'status':       'loaded',
                    'fake_prob':    round(fake_prob, 4),
                    'real_prob':    round(real_prob, 4),
                    'fake_label':   id2label.get(fake_idx, str(fake_idx)),
                    'real_label':   id2label.get(real_idx, str(real_idx)),
                    'weight':       PRETRAINED_ENSEMBLE_WEIGHTS.get(key, 0),
                    'verdict':      'FAKE' if fake_prob >= 0.5 else 'REAL',
                }
            except Exception as e:
                model_results[key] = {'status': f'inference_error: {e}', 'fake_prob': None, 'verdict': 'error'}

        # ── Signal analysis ────────────────────────────────────────────────────
        signal     = combined_signal_analysis(image)
        ela        = ela_score(image)
        ai_smooth  = ai_smoothness_score(image)

        return jsonify({
            'filename':         filename,
            'image_size':       f"{image.width}x{image.height}",
            'neural_models':    model_results,
            'loaded_count':     sum(1 for v in model_results.values() if v.get('status') == 'loaded'),
            'signal_analysis': {
                'noise_score':       signal['noise_score'],
                'compression_score': signal['compression_score'],
                'frequency_score':   signal['frequency_score'],
                'gradient_score':    signal.get('gradient_score', 'N/A'),
                'ela_score':         round(ela, 4),
                'ai_smoothness':     round(ai_smooth, 4),
            },
            'diagnosis': {
                'advice': (
                    'If loaded_count is 0, the models are still downloading — wait and retry. '
                    'If fake_prob is low for all models on a known-fake image, '
                    'the image type may not match the training distribution.'
                )
            },
            'success': True,
        })

    except Exception as e:
        logger.error(f"Debug failed: {e}", exc_info=True)
        return jsonify({'error': str(e), 'success': False}), 500

    finally:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass


@app.errorhandler(413)
def file_too_large(e):
    return jsonify({
        'error': 'File too large. Maximum size is 500MB.',
        'success': False
    }), 413


@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found', 'success': False}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error', 'success': False}), 500


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Deepfake Detection Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to listen on')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Deepfake Detection System")
    logger.info(f"Starting server on {args.host}:{args.port}")
    logger.info("Models will be loaded on first request (lazy loading)")
    logger.info("=" * 60)

    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
