"""
Error Level Analysis (ELA) - Multi-Quality Deterministic Implementation.

Improvements over baseline:
  - Multi-quality ELA: runs at 3 JPEG quality levels (75, 85, 95).
    Different quality levels expose different types of manipulation.
  - Continuous scoring using calibrated sigmoid instead of linear normalisation.
  - Regional variance: checks whether ELA is uniform (real) or has
    high-error hotspots (tampered regions).
  - All operations remain fully deterministic.
"""
import numpy as np
from PIL import Image
import io
import logging

logger = logging.getLogger(__name__)


def compute_ela(image: Image.Image, quality: int = 90, scale: int = 15) -> np.ndarray:
    """
    Compute Error Level Analysis map for an image.

    Args:
        image:   PIL Image (RGB)
        quality: JPEG re-save quality (lower → more ELA in manipulated regions)
        scale:   Scale factor for visualisation

    Returns:
        numpy array of ELA differences, shape (H, W, 3)
        Fully deterministic.
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')

    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)

    compressed      = Image.open(buffer).convert('RGB')
    original_array  = np.array(image,      dtype=np.float32)
    compressed_array = np.array(compressed, dtype=np.float32)

    ela_array = np.abs(original_array - compressed_array) * scale
    ela_array = np.clip(ela_array, 0, 255).astype(np.uint8)
    return ela_array


def _ela_stats(image: Image.Image, quality: int) -> dict:
    """Compute ELA statistics at a single quality level."""
    ela_map   = compute_ela(image, quality=quality)
    ela_float = ela_map.astype(np.float32)

    mean_ela  = float(np.mean(ela_float))
    std_ela   = float(np.std(ela_float))
    max_ela   = float(np.max(ela_float))

    # Regional hotspot score: fraction of pixels above 75th percentile
    # that are ≥ 3× the global mean → tampered "islands"
    p75 = float(np.percentile(ela_float, 75))
    hotspot_frac = float(np.mean(ela_float > max(mean_ela * 3.0, p75 * 2.0)))

    return {
        'mean':    mean_ela,
        'std':     std_ela,
        'max':     max_ela,
        'hotspot': hotspot_frac,
    }


def _sigmoid_score(x: float, k: float, x0: float) -> float:
    """Calibrated sigmoid: σ(k*(x-x0)) → [0, 1]."""
    return float(1.0 / (1.0 + np.exp(-k * (x - x0))))


def ela_score(image: Image.Image) -> float:
    """
    Multi-quality ELA score for deepfake / manipulation detection.

    Runs ELA at quality 75, 85, and 95 to capture artefacts across a
    wide dynamic range.  The final score is a weighted combination of:
      - normalised mean ELA (overall level of re-compression noise)
      - normalised std ELA  (uniformity; uniform = possibly real)
      - hotspot fraction    (localised tampering regions)

    Returns:
        float in [0, 1] — higher = more likely manipulated / fake
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')

    qualities = [75, 85, 95]
    quality_weights = [0.40, 0.35, 0.25]   # lower q → more discriminative

    weighted_score = 0.0

    for q, w in zip(qualities, quality_weights):
        try:
            stats = _ela_stats(image, quality=q)

            # Component scores using calibrated sigmoids
            # Empirical calibration:
            #   Real images at q=90: mean~2-6, std~3-8
            #   Manipulated/ AI at q=90: mean~15-40, std~18-50
            mean_score    = _sigmoid_score(stats['mean'],    k=0.18, x0=10.0)
            std_score     = _sigmoid_score(stats['std'],     k=0.14, x0=12.0)
            hotspot_score = _sigmoid_score(stats['hotspot'], k=20.0, x0=0.05)

            # Regional uniformity bonus: real images tend to have uniform ELA
            # Very low hotspot fraction → might be real → pull score down slightly
            uniformity_penalty = max(0.0, 0.08 - stats['hotspot']) * 2.0  # ≤0.16 reduction

            single_q_score = (
                0.40 * mean_score +
                0.35 * std_score  +
                0.25 * hotspot_score
                - uniformity_penalty
            )
            single_q_score = float(np.clip(single_q_score, 0.0, 1.0))

            weighted_score += w * single_q_score

            logger.debug(
                f"ELA q={q}: mean={stats['mean']:.2f}  std={stats['std']:.2f}  "
                f"hotspot={stats['hotspot']:.4f}  score={single_q_score:.4f}"
            )

        except Exception as e:
            logger.warning(f"ELA at quality={q} failed: {e}")
            weighted_score += w * 0.4   # neutral fallback for this level

    return float(np.clip(weighted_score, 0.0, 1.0))


def ela_as_tensor(image: Image.Image, target_size: int = 224, quality: int = 90) -> np.ndarray:
    """
    Get ELA map resized and normalised as a tensor-ready array.

    Returns:
        numpy array of shape (3, target_size, target_size), float32 in [0, 1]
    """
    ela_map     = compute_ela(image, quality=quality)
    ela_pil     = Image.fromarray(ela_map, mode='RGB')
    ela_resized = ela_pil.resize((target_size, target_size), Image.BILINEAR)
    ela_array   = np.array(ela_resized, dtype=np.float32) / 255.0
    return ela_array.transpose(2, 0, 1)   # (H, W, C) → (C, H, W)
