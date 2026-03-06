"""
Model Manager — Simple, no-frills model loader.
No threading complexity. Models cached as module-level globals.

Label indices VERIFIED from live debug output (2026-03-04):
  vit_v2:    {0:'Realism', 1:'Deepfake'} -> fake_idx=1
  siglip_v1: {0:'Fake',    1:'Real'}     -> fake_idx=0
  vit_wvolf: {0:'Real',    1:'Fake'}     -> fake_idx=1
"""

import torch
import logging
import gc

logger = logging.getLogger(__name__)

HF_MODELS = {
    'vit_v2': {
        'repo': 'prithivMLmods/Deep-Fake-Detector-v2-Model',
        'type': 'vit',
        'fake_label_idx': 1,
        'real_label_idx': 0,
        'accuracy': 0.9212,
        'description': 'ViT fine-tuned on 56k images (92.1% acc)',
    },
    'siglip_v1': {
        'repo': 'prithivMLmods/deepfake-detector-model-v1',
        'type': 'siglip',
        'fake_label_idx': 0,
        'real_label_idx': 1,
        'accuracy': 0.9444,
        'description': 'SigLIP fine-tuned on 20k images (94.4% acc)',
    },
    'vit_wvolf': {
        'repo': 'Wvolf/ViT_Deepfake_Detection',
        'type': 'vit',
        'fake_label_idx': 1,
        'real_label_idx': 0,
        'accuracy': 0.9870,
        'description': 'ViT deepfake detector (98.7% acc)',
    },
}

PRETRAINED_ENSEMBLE_WEIGHTS = {
    'vit_v2':     0.25,
    'siglip_v1':  0.30,
    'vit_wvolf':  0.45,
}

# ── Module-level cache (simple globals, no class complexity) ──────────────────
_model_cache     = {}   # key -> model
_processor_cache = {}   # key -> processor
_status_cache    = {}   # key -> 'loaded' | 'failed: ...'


def _load_one_model(model_key):
    """Load a single HF model into the module-level cache."""
    if model_key in _model_cache:
        return  # already loaded

    cfg  = HF_MODELS.get(model_key)
    repo = cfg['repo']
    logger.info(f"Loading {model_key}: {repo}")

    try:
        if cfg['type'] == 'vit':
            from transformers import ViTForImageClassification, ViTImageProcessor
            proc  = ViTImageProcessor.from_pretrained(repo)
            model = ViTForImageClassification.from_pretrained(
                        repo, torch_dtype=torch.float32, low_cpu_mem_usage=True)

        elif cfg['type'] == 'siglip':
            from transformers import AutoImageProcessor, SiglipForImageClassification
            proc  = AutoImageProcessor.from_pretrained(repo)
            model = SiglipForImageClassification.from_pretrained(
                        repo, torch_dtype=torch.float32, low_cpu_mem_usage=True)

        else:
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            proc  = AutoImageProcessor.from_pretrained(repo)
            model = AutoModelForImageClassification.from_pretrained(
                        repo, torch_dtype=torch.float32, low_cpu_mem_usage=True)

        model.eval()
        model = model.cpu()

        if hasattr(model.config, 'id2label'):
            logger.info(f"  id2label: {model.config.id2label}")

        _model_cache[model_key]     = model
        _processor_cache[model_key] = proc
        _status_cache[model_key]    = 'loaded'
        logger.info(f"  OK: {model_key} loaded")

    except Exception as e:
        logger.error(f"  FAILED {model_key}: {e}")
        _model_cache[model_key]     = None
        _processor_cache[model_key] = None
        _status_cache[model_key]    = f'failed: {e}'


def get_model_and_processor(model_key):
    """
    Returns (model, processor) — always exactly 2 values.
    Loads from HuggingFace on first call, returns cached on subsequent calls.
    """
    if model_key not in _model_cache:
        _load_one_model(model_key)

    model     = _model_cache.get(model_key, None)
    processor = _processor_cache.get(model_key, None)
    return model, processor   # exactly 2 — no class, no lock, no ambiguity


def get_all_models():
    """Load all models and return {key: (model, processor)} for loaded ones."""
    result = {}
    for key in HF_MODELS:
        model, processor = get_model_and_processor(key)
        if model is not None:
            result[key] = (model, processor)
    return result


def get_load_status():
    return dict(_status_cache)


def get_loaded_keys():
    return [k for k, v in _model_cache.items() if v is not None]


def clear_cache():
    _model_cache.clear()
    _processor_cache.clear()
    _status_cache.clear()
    gc.collect()


# ── Keep ModelManager class for app.py compatibility ─────────────────────────
class ModelManager:
    """
    Thin wrapper around module-level functions.
    Kept for backward compatibility with app.py and video_detector.py.
    """

    def get_pretrained_model(self, model_key):
        """Returns (model, processor) — exactly 2 values."""
        return get_model_and_processor(model_key)

    def get_all_pretrained(self):
        return get_all_models()

    def get_load_status(self):
        return get_load_status()

    def get_loaded_models(self):
        return get_loaded_keys()

    def get_convnext_model(self):
        if 'convnext' not in _model_cache:
            _model_cache['convnext'] = self._load_convnext()
        return _model_cache['convnext']

    def get_swin_model(self):
        if 'swin' not in _model_cache:
            _model_cache['swin'] = self._load_swin()
        return _model_cache['swin']

    def _load_convnext(self):
        try:
            import timm
            model = timm.create_model('convnext_base', pretrained=True, num_classes=0)
            model.eval()
            return model
        except Exception:
            try:
                import torchvision
                model = torchvision.models.resnet50(weights='IMAGENET1K_V1')
                model.fc = torch.nn.Identity()
                model.eval()
                return model
            except Exception as e:
                logger.error(f"ConvNeXt failed: {e}")
                return None

    def _load_swin(self):
        try:
            import timm
            model = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=0)
            model.eval()
            return model
        except Exception:
            try:
                import torchvision
                model = torchvision.models.vit_b_16(weights='IMAGENET1K_V1')
                model.heads = torch.nn.Identity()
                model.eval()
                return model
            except Exception as e:
                logger.error(f"Swin failed: {e}")
                return None

    def clear_cache(self):
        clear_cache()
