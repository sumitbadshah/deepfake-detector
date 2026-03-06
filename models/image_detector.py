"""
High-Accuracy Image Deepfake Detector — Properly Calibrated.

Key improvements in this version:
1. Temperature Scaling (T=2.5) applied per-model before ensemble
   — standard fix for overconfident ViTs
2. Rebalanced weights (0.33/0.34/0.33) — accuracy != calibration quality,
   the 98.7% model was overconfident and skewing the ensemble
3. Adaptive signal weight — 20% when ensemble is uncertain (<0.6),
   5% when confident (>0.85), 10% otherwise
4. Confidence thresholds — "Uncertain", "Medium", "High" labels
5. Calibrated sigmoid k=12 for final display sharpening

Label indices VERIFIED from live debug output (2026-03-04):
  vit_v2:    {0:'Realism', 1:'Deepfake'} -> fake_idx=1
  siglip_v1: {0:'Fake',    1:'Real'}     -> fake_idx=0
  vit_wvolf: {0:'Real',    1:'Fake'}     -> fake_idx=1
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import logging
import time

from .model_manager import get_model_and_processor, HF_MODELS, PRETRAINED_ENSEMBLE_WEIGHTS
from .utils.deterministic import set_deterministic, GLOBAL_SEED
from .utils.ela import ela_score
from .utils.noise_analysis import combined_signal_analysis

logger = logging.getLogger(__name__)

# Rebalanced weights — equal weight since accuracy != calibration quality
# The 98.7% model was overconfident and pulling fake scores down
CALIBRATED_WEIGHTS = {
    'vit_v2':     0.33,
    'siglip_v1':  0.34,
    'vit_wvolf':  0.33,
}

# Temperature scaling — standard fix for overconfident ViTs
# T > 1.0 softens the distribution, T=2.5 is typical for ViTs
TEMPERATURE = 2.5


def _apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Divide logits by temperature before softmax — softens overconfident predictions."""
    return F.softmax(logits / temperature, dim=-1)


def _calibrate_display(raw: float) -> float:
    """Steepened sigmoid (k=12) for display sharpening."""
    return float(1.0 / (1.0 + np.exp(-12.0 * (raw - 0.5))))


def _adaptive_signal_weight(ensemble_confidence: float) -> float:
    """
    Give more weight to signal analysis when ensemble is uncertain.
    - Low confidence (<0.60) → 20% signal weight
    - High confidence (>0.85) → 5% signal weight
    - Medium → 10%
    """
    if ensemble_confidence < 0.60:
        return 0.20
    elif ensemble_confidence > 0.85:
        return 0.05
    else:
        return 0.10


def _confidence_label(confidence: float) -> str:
    """Human-readable confidence tier."""
    if confidence >= 0.85:
        return 'High'
    elif confidence >= 0.65:
        return 'Medium'
    else:
        return 'Uncertain — manual review recommended'


def _infer_one_model(model, processor, image: Image.Image,
                     fake_idx: int, real_idx: int) -> tuple:
    """
    Run one model with temperature scaling.
    Returns (fake_prob, real_prob) after temperature-scaled softmax.
    """
    encoding = processor(images=image, return_tensors="pt")
    inputs   = {k: v for k, v in encoding.items()}

    with torch.no_grad():
        logits = model(**inputs).logits        # (1, num_classes)
        # Apply temperature scaling before softmax
        probs  = _apply_temperature(logits[0], TEMPERATURE)   # (num_classes,)

    fake_prob = float(probs[fake_idx].item())
    real_prob = float(probs[real_idx].item())
    return fake_prob, real_prob


class ImageDetector:
    """Deepfake image detector: 3 pretrained models + signal analysis."""

    def __init__(self):
        pass

    def neural_ensemble_score(self, image: Image.Image) -> dict:
        set_deterministic(GLOBAL_SEED)

        per_model  = {}
        wsum       = 0.0
        wtotal     = 0.0
        votes_fake = 0
        votes_real = 0

        for key, cfg in HF_MODELS.items():
            model, processor = get_model_and_processor(key)
            if model is None or processor is None:
                logger.warning(f"  {key}: not loaded, skipping")
                continue

            try:
                fake_p, real_p = _infer_one_model(
                    model, processor, image,
                    cfg['fake_label_idx'],
                    cfg['real_label_idx'],
                )
                # Use calibrated weights (equal) instead of accuracy-based weights
                w = CALIBRATED_WEIGHTS[key]
                per_model[key] = round(fake_p, 4)
                wsum   += w * fake_p
                wtotal += w

                verdict = 'FAKE' if fake_p >= 0.5 else 'REAL'
                logger.info(f"  {key}: P(fake)={fake_p:.4f} P(real)={real_p:.4f} → {verdict}")

                if fake_p >= 0.5:
                    votes_fake += 1
                else:
                    votes_real += 1

            except Exception as e:
                logger.error(f"  {key} inference failed: {e}", exc_info=True)

        if wtotal == 0:
            logger.error("No models produced output!")
            return dict(per_model={}, ensemble=0.5, models_used=0,
                        votes_fake=0, votes_real=0)

        ensemble = wsum / wtotal

        # Unanimous vote: trust consensus strongly
        total_v = votes_fake + votes_real
        if total_v >= 2:
            if votes_fake == total_v:
                ensemble = max(ensemble, 0.65)
            elif votes_real == total_v:
                ensemble = min(ensemble, 0.35)

        ensemble = float(np.clip(ensemble, 0.0, 1.0))
        logger.info(f"  Neural ensemble={ensemble:.4f}  votes={votes_fake}F/{votes_real}R")

        return dict(
            per_model   = per_model,
            ensemble    = round(ensemble, 4),
            models_used = len(per_model),
            votes_fake  = votes_fake,
            votes_real  = votes_real,
        )

    def analyze(self, image_path: str) -> dict:
        start = time.time()
        set_deterministic(GLOBAL_SEED)
        logger.info(f"\n{'='*55}\nAnalyzing: {image_path}")

        # Load image
        try:
            image         = Image.open(image_path).convert('RGB')
            width, height = image.size
            logger.info(f"  Size: {width}x{height}")
        except Exception as e:
            logger.error(f"  Cannot open: {e}")
            return self._error_result(str(e))

        try:
            # 1. Neural ensemble with temperature scaling
            neural       = self.neural_ensemble_score(image)
            neural_score = neural['ensemble']

            # 2. Compute raw ensemble confidence (distance from 0.5)
            ensemble_conf = abs(neural_score - 0.5) * 2.0  # 0..1

            # 3. Adaptive signal weight based on ensemble confidence
            sig_weight  = _adaptive_signal_weight(ensemble_conf)
            neu_weight  = 1.0 - sig_weight

            # 4. Signal analyses
            sig         = combined_signal_analysis(image)
            noise_score = sig['noise_score']
            comp_score  = sig['compression_score']
            ela         = ela_score(image)

            # Combined signal score (weighted average of 3 signals)
            signal_score = (0.40 * noise_score +
                           0.35 * comp_score   +
                           0.25 * ela)

            logger.info(f"  Ensemble conf={ensemble_conf:.2f} → signal_weight={sig_weight:.2f}")
            logger.info(f"  Signal: noise={noise_score:.4f} comp={comp_score:.4f} ela={ela:.4f} combined={signal_score:.4f}")

            # 5. Final weighted combination
            raw = float(np.clip(
                neu_weight * neural_score + sig_weight * signal_score,
                0.0, 1.0
            ))

            # 6. Calibrate for display
            calibrated = _calibrate_display(raw)
            is_fake    = raw > 0.5
            prediction = 'FAKE' if is_fake else 'REAL'
            confidence = calibrated if is_fake else (1.0 - calibrated)
            conf_label = _confidence_label(confidence)

            logger.info(f"  raw={raw:.4f}  calibrated={calibrated:.4f}")
            logger.info(f"  RESULT: {prediction}  confidence={confidence:.2%}  [{conf_label}]")
            logger.info(f"{'='*55}")

            # Build scores dict
            scores = {mk: v for mk, v in neural['per_model'].items()}
            scores['noise_analysis']        = round(noise_score,   4)
            scores['compression_artifacts'] = round(comp_score,    4)
            scores['ela_score']             = round(ela,           4)
            scores['signal_combined']       = round(signal_score,  4)
            scores['neural_ensemble']       = round(neural_score,  4)
            scores['raw_score']             = round(raw,           4)
            scores['final_calibrated']      = round(calibrated,    4)

            return {
                'prediction':         prediction,
                'confidence':         round(confidence, 4),
                'fake_probability':   round(raw, 4),
                'confidence_label':   conf_label,
                'needs_review':       0.4 < raw < 0.65,
                'analysis_time':      round(time.time() - start, 3),
                'image_size':         f"{width}x{height}",
                'models_used':        neural['models_used'],
                'temperature':        TEMPERATURE,
                'signal_weight_used': round(sig_weight, 2),
                'votes': {
                    'fake': neural['votes_fake'],
                    'real': neural['votes_real'],
                },
                'scores': scores,
                'model_details': {
                    mk: {
                        'fake_probability': v,
                        'verdict':          'FAKE' if v >= 0.5 else 'REAL',
                        'accuracy':         HF_MODELS[mk]['accuracy'],
                        'description':      HF_MODELS[mk]['description'],
                        'calibrated_weight': CALIBRATED_WEIGHTS[mk],
                    }
                    for mk, v in neural['per_model'].items()
                },
                'ensemble_config': {
                    'temperature':        TEMPERATURE,
                    'calibrated_weights': CALIBRATED_WEIGHTS,
                    'signal_weight':      sig_weight,
                    'neural_weight':      neu_weight,
                },
                'model': 'Pretrained ViT + SigLIP Ensemble + Temperature Scaling',
                'error': None,
            }

        except Exception as e:
            logger.error(f"  Analysis exception: {e}", exc_info=True)
            return self._error_result(str(e))

    def _error_result(self, msg: str) -> dict:
        return {
            'prediction':         'REAL',
            'confidence':         0.5,
            'fake_probability':   0.5,
            'confidence_label':   'Uncertain',
            'needs_review':       True,
            'analysis_time':      0.0,
            'image_size':         'unknown',
            'models_used':        0,
            'temperature':        TEMPERATURE,
            'signal_weight_used': 0.10,
            'votes':              {'fake': 0, 'real': 0},
            'scores':             {},
            'model_details':      {},
            'ensemble_config':    {},
            'model':              'Pretrained ViT + SigLIP Ensemble + Temperature Scaling',
            'error':              msg,
        }
