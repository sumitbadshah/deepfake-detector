"""
Noise Analysis Utilities - Deterministic, Continuous-Score Implementation.

Improvements over baseline:
  - All scores are continuous (sigmoid/log-based) instead of discrete buckets.
    Continuous scores generalise better and avoid hard threshold jumps.
  - estimate_noise_level: uses both variance AND a smoothness ratio.
  - detect_compression_artifacts: adds grid-line energy as a second signal.
  - frequency_analysis: uses full radial energy profile, not two bands.
  - New: gradient_consistency — measures local gradient coherence.
"""
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)


# ── helpers ────────────────────────────────────────────────────────────────────

def _to_gray_array(image: Image.Image) -> np.ndarray:
    """Return float32 grayscale array in [0, 255]."""
    return np.array(image.convert('L'), dtype=np.float32)


def _sigmoid(x: float, k: float = 1.0, x0: float = 0.0) -> float:
    """Smooth sigmoid: σ(k*(x-x0))."""
    return float(1.0 / (1.0 + np.exp(-k * (x - x0))))


# ── 1. Noise level estimator ───────────────────────────────────────────────────

def estimate_noise_level(image: Image.Image) -> float:
    """
    Estimate noise-based fakeness score using two complementary signals:

    a) Laplacian variance  – AI images are often too smooth (very low) or
       over-sharpened post-generation (very high).
    b) Smoothness ratio    – ratio of near-zero Laplacian pixels; AI images
       have unnaturally many flat regions.

    Returns a continuous float in [0, 1].
    High score → more likely fake/generated.
    """
    img = _to_gray_array(image)

    # Manual Laplacian convolution (deterministic, no scipy needed)
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    h, w = img.shape
    padded = np.pad(img, 1, mode='reflect')
    lap = np.zeros((h, w), dtype=np.float32)
    for i in range(3):
        for j in range(3):
            lap += kernel[i, j] * padded[i:i+h, j:j+w]

    variance = float(np.var(lap))

    # Smoothness ratio: fraction of pixels with very low |Laplacian|
    smoothness_ratio = float(np.mean(np.abs(lap) < 2.0))

    # Score from variance
    # Real photos: variance ~200-3000.  AI images: <50 or very high.
    # Map using a smooth V-shaped score centred on the "normal" range.
    log_var = np.log1p(variance)
    log_normal_centre = np.log1p(600)   # midpoint of normal range
    log_spread = 2.5                     # controls sharpness of V
    var_score = float(np.clip(abs(log_var - log_normal_centre) / log_spread, 0.0, 1.0))

    # Score from smoothness ratio
    # Typical real photo: ratio ~0.3-0.5.  AI: often >0.65 (too smooth).
    smooth_score = _sigmoid(smoothness_ratio, k=12.0, x0=0.58)

    # Combine (65% variance signature, 35% smoothness)
    score = 0.65 * var_score + 0.35 * smooth_score

    logger.debug(f"noise: var={variance:.1f}  smooth_ratio={smoothness_ratio:.3f}  score={score:.4f}")
    return float(np.clip(score, 0.0, 1.0))


# ── 2. Compression artefact detector ──────────────────────────────────────────

def detect_compression_artifacts(image: Image.Image) -> float:
    """
    Analyse 8×8 DCT block artefacts.

    Two signals:
    a) Coefficient-of-variation of block variances — inconsistent blocks
       indicate splicing / local manipulation.
    b) Block-boundary energy — the energy spike exactly at 8-pixel grid lines
       is characteristic of JPEG re-compression (deepfake pipeline artefact).

    Returns a continuous float in [0, 1].
    """
    img = _to_gray_array(image)
    h, w = img.shape

    # ── (a) Block variance CV ──────────────────────────────────────────────────
    block_variances = []
    for row in range(0, h - 8, 8):
        for col in range(0, w - 8, 8):
            block = img[row:row+8, col:col+8]
            block_variances.append(float(np.var(block)))

    if not block_variances:
        return 0.3

    variances = np.array(block_variances)
    mean_var  = float(np.mean(variances))
    cv        = float(np.std(variances) / (mean_var + 1e-6))
    cv_score  = _sigmoid(cv, k=3.0, x0=1.8)   # high CV → high score

    # ── (b) Grid-line boundary energy ─────────────────────────────────────────
    # Compare mean absolute difference at 8-pixel boundaries vs interior lines.
    row_diffs_boundary = []
    row_diffs_interior = []
    for r in range(1, h):
        diff = float(np.mean(np.abs(img[r] - img[r-1])))
        if r % 8 == 0:
            row_diffs_boundary.append(diff)
        else:
            row_diffs_interior.append(diff)

    if row_diffs_boundary and row_diffs_interior:
        boundary_ratio = float(np.mean(row_diffs_boundary)) / (float(np.mean(row_diffs_interior)) + 1e-6)
        # Legitimate JPEG: ratio ~1.0.  Re-compressed fakes: often >1.3.
        grid_score = _sigmoid(boundary_ratio, k=5.0, x0=1.25)
    else:
        grid_score = 0.3

    score = 0.55 * cv_score + 0.45 * grid_score
    logger.debug(f"compression: cv={cv:.3f}  boundary_ratio={boundary_ratio if row_diffs_boundary and row_diffs_interior else 'N/A'}  score={score:.4f}")
    return float(np.clip(score, 0.0, 1.0))


# ── 3. Frequency domain analysis ───────────────────────────────────────────────

def frequency_analysis(image: Image.Image) -> float:
    """
    Full radial frequency energy profile using 2D FFT.

    Real photographs follow a roughly 1/f power spectrum.
    AI-generated images often deviate: too flat (over-smooth) or
    have periodic artefacts at mid-frequencies (GAN fingerprints).

    Returns a continuous float in [0, 1].
    """
    img = _to_gray_array(image).astype(np.float64)

    fft         = np.fft.fft2(img)
    fft_shifted = np.fft.fftshift(fft)
    power       = np.log(np.abs(fft_shifted) ** 2 + 1)

    h, w  = power.shape
    cy, cx = h // 2, w // 2
    max_r  = min(h, w) // 2

    y_idx, x_idx = np.ogrid[:h, :w]
    dist = np.sqrt((y_idx - cy) ** 2 + (x_idx - cx) ** 2)

    # Build radial profile with 32 bins
    n_bins   = 32
    bin_edges = np.linspace(0, max_r, n_bins + 1)
    profile  = []
    for k in range(n_bins):
        mask = (dist >= bin_edges[k]) & (dist < bin_edges[k+1])
        if mask.any():
            profile.append(float(np.mean(power[mask])))

    if len(profile) < 4:
        return 0.3

    profile = np.array(profile)

    # ── 1/f deviation: real photos have steeply falling profile ───────────────
    # Fit a line to log-profile; large residuals = anomalous spectrum.
    x_vals   = np.arange(len(profile), dtype=np.float64)
    coeffs   = np.polyfit(x_vals, profile, 1)
    fitted   = np.polyval(coeffs, x_vals)
    residuals = profile - fitted
    spectral_flatness = float(np.std(residuals) / (np.mean(np.abs(profile)) + 1e-6))

    # ── High-freq / low-freq ratio ─────────────────────────────────────────────
    mid   = len(profile) // 2
    lo_e  = float(np.mean(profile[:mid]))
    hi_e  = float(np.mean(profile[mid:]))
    ratio = hi_e / (lo_e + 1e-6)

    # Typical real ratio: 0.3-0.65.  Score rises outside that window.
    ratio_score = _sigmoid(abs(ratio - 0.48), k=6.0, x0=0.22)   # peaks at extreme ratios
    flat_score  = _sigmoid(spectral_flatness, k=8.0, x0=0.40)   # rises with flat spectrum

    score = 0.50 * ratio_score + 0.50 * flat_score
    logger.debug(f"frequency: ratio={ratio:.3f}  flatness={spectral_flatness:.3f}  score={score:.4f}")
    return float(np.clip(score, 0.0, 1.0))


# ── 4. Gradient consistency ─────────────────────────────────────────────────────

def gradient_consistency(image: Image.Image) -> float:
    """
    Measure local gradient coherence.

    AI-generated images often have unnaturally coherent gradients
    (over-smoothed textures) or incoherent gradient magnitudes at
    face boundaries (blending seams).

    Returns a continuous float in [0, 1].
    """
    img = _to_gray_array(image)
    h, w = img.shape

    # Sobel-like gradients
    gx = img[:, 1:] - img[:, :-1]    # horizontal differences
    gy = img[1:, :] - img[:-1, :]    # vertical differences

    mag_x = np.abs(gx[:h-1, :w-1])
    mag_y = np.abs(gy[:h-1, :w-1])
    mag   = np.sqrt(mag_x**2 + mag_y**2 + 1e-6)

    # Local coherence: compute 8×8 block-level std of gradient magnitude
    bsize = 8
    bh = (h - 1) // bsize
    bw = (w - 1) // bsize

    block_stds = []
    for r in range(bh):
        for c in range(bw):
            block = mag[r*bsize:(r+1)*bsize, c*bsize:(c+1)*bsize]
            block_stds.append(float(np.std(block)))

    if not block_stds:
        return 0.3

    stds = np.array(block_stds)
    # High variance of block-stds → incoherent gradients → suspicious
    coherence_var = float(np.var(stds))
    score = _sigmoid(np.log1p(coherence_var), k=0.8, x0=5.5)

    logger.debug(f"gradient: coherence_var={coherence_var:.2f}  score={score:.4f}")
    return float(np.clip(score, 0.0, 1.0))


# ── Combined analysis ─────────────────────────────────────────────────────────

def combined_signal_analysis(image: Image.Image) -> dict:
    """
    Run all signal-based analyses and return individual + combined scores.

    Returns:
        dict with noise_score, compression_score, frequency_score,
              gradient_score, combined
    """
    noise_score        = estimate_noise_level(image)
    compression_score  = detect_compression_artifacts(image)
    frequency_score    = frequency_analysis(image)
    gradient_score     = gradient_consistency(image)

    # Weighted combination
    combined = (
        0.32 * noise_score       +
        0.28 * compression_score +
        0.22 * frequency_score   +
        0.18 * gradient_score
    )

    return {
        'noise_score':       round(noise_score,       4),
        'compression_score': round(compression_score, 4),
        'frequency_score':   round(frequency_score,   4),
        'gradient_score':    round(gradient_score,    4),
        'combined':          round(combined,          4),
    }


# ── 5. AI Smoothness Score ────────────────────────────────────────────────────

def ai_smoothness_score(image: Image.Image) -> float:
    """
    Dedicated detector for the "too perfect / too clean" fingerprint of
    AI-generated, GAN, and diffusion-model images.

    Real photographs contain:
      - Sensor noise (random grain)
      - Slight colour channel misalignment
      - Non-uniform sharpness across the frame
      - Natural texture irregularities

    AI images typically LACK all of the above, making them look
    unnaturally clean. This function quantifies that "cleanliness" and
    returns a score where HIGH = more likely AI-generated.

    Five complementary signals:
      1. Micro-noise level  — real photos have visible sensor grain
      2. Colour channel correlation — AI images have perfectly aligned channels
      3. Local texture entropy — AI images have suspiciously uniform texture
      4. Sharpness uniformity — real images have variable focus across regions
      5. JPEG history absence — AI images have pristine compression history

    Returns float in [0, 1]. Higher = more likely AI / synthetic.
    """
    img_rgb = np.array(image.convert('RGB'), dtype=np.float32)
    h, w, _ = img_rgb.shape

    # ── 1. Micro-noise level ────────────────────────────────────────────────────
    # Apply a strong Gaussian-like blur (box filter) and measure the residual.
    # Real photos: residual has visible structure (grain).
    # AI images:   residual is near-zero or perfectly smooth.
    blur_size = 5
    pad = blur_size // 2
    gray = img_rgb.mean(axis=2)

    # Box blur
    kernel = np.ones((blur_size, blur_size), dtype=np.float32) / (blur_size ** 2)
    padded = np.pad(gray, pad, mode='reflect')
    blurred = np.zeros_like(gray)
    for i in range(blur_size):
        for j in range(blur_size):
            blurred += kernel[i, j] * padded[i:i+h, j:j+w]

    residual = np.abs(gray - blurred)
    micro_noise = float(np.mean(residual))

    # Real photos: micro_noise typically 2–8.  AI images: often < 1.5.
    # Score rises when noise is unnaturally low.
    noise_signal = _sigmoid(-micro_noise, k=1.5, x0=-2.5)   # high score when noise is low

    # ── 2. Colour channel correlation ──────────────────────────────────────────
    # Real photos: R, G, B channels have different noise patterns (Bayer sensor).
    # AI images:   channels are nearly perfectly correlated (synthesised together).
    r = img_rgb[:, :, 0].flatten()
    g = img_rgb[:, :, 1].flatten()
    b = img_rgb[:, :, 2].flatten()

    try:
        corr_rg = float(abs(np.corrcoef(r, g)[0, 1]))
        corr_gb = float(abs(np.corrcoef(g, b)[0, 1]))
        avg_corr = (corr_rg + corr_gb) / 2.0
    except Exception:
        avg_corr = 0.85

    # Perfect correlation (avg_corr → 1.0) = AI.  Real: typically 0.85-0.95.
    channel_signal = _sigmoid(avg_corr, k=20.0, x0=0.97)   # rises sharply near 1.0

    # ── 3. Local texture entropy ───────────────────────────────────────────────
    # Compute entropy in non-overlapping 16×16 blocks.
    # AI images: very uniform entropy (all blocks look similar = over-smooth).
    # Real images: wider range of local entropies.
    bsize = 16
    bh    = h // bsize
    bw    = w // bsize
    entropies = []

    for r_idx in range(bh):
        for c_idx in range(bw):
            block = gray[r_idx*bsize:(r_idx+1)*bsize, c_idx*bsize:(c_idx+1)*bsize]
            # Histogram-based entropy
            hist, _ = np.histogram(block.flatten(), bins=32, range=(0, 255))
            hist    = hist.astype(np.float64)
            hist   /= hist.sum() + 1e-9
            ent     = float(-np.sum(hist * np.log2(hist + 1e-9)))
            entropies.append(ent)

    if len(entropies) > 4:
        ent_std = float(np.std(entropies))
        # Low std = uniformly textured = AI. Real: typically std > 0.6.
        entropy_signal = _sigmoid(-ent_std, k=4.0, x0=-0.5)   # high when std is low
    else:
        entropy_signal = 0.4

    # ── 4. Sharpness uniformity ─────────────────────────────────────────────────
    # Real images: sharpness (Laplacian variance) varies significantly across regions.
    # AI images: uniformly sharp because the generator applies consistent attention.
    bsize2 = 32
    bh2    = h // bsize2
    bw2    = w // bsize2
    sharp_vals = []

    for r_idx in range(bh2):
        for c_idx in range(bw2):
            block = gray[r_idx*bsize2:(r_idx+1)*bsize2, c_idx*bsize2:(c_idx+1)*bsize2]
            # Simple Laplacian variance = local sharpness
            lap = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float32)
            bh_, bw_ = block.shape
            pb = np.pad(block, 1, mode='reflect')
            lb = np.zeros_like(block)
            for ii in range(3):
                for jj in range(3):
                    lb += lap[ii, jj] * pb[ii:ii+bh_, jj:jj+bw_]
            sharp_vals.append(float(np.var(lb)))

    if len(sharp_vals) > 4:
        sharp_cv = float(np.std(sharp_vals) / (np.mean(sharp_vals) + 1e-6))
        # Low CV = uniformly sharp everywhere = AI. Real: typically CV > 0.8.
        sharpness_signal = _sigmoid(-sharp_cv, k=3.0, x0=-0.7)
    else:
        sharpness_signal = 0.4

    # ── 5. JPEG history absence ────────────────────────────────────────────────
    # Real photos from cameras/phones always have JPEG history artefacts at 8px.
    # AI images rendered/saved for the first time lack this history.
    # We use the grid-line boundary energy: very LOW boundary energy → AI-fresh.
    row_diffs_boundary, row_diffs_interior = [], []
    for row in range(1, min(h, 200)):   # sample first 200 rows for speed
        diff = float(np.mean(np.abs(img_rgb[row] - img_rgb[row-1])))
        if row % 8 == 0:
            row_diffs_boundary.append(diff)
        else:
            row_diffs_interior.append(diff)

    if row_diffs_boundary and row_diffs_interior:
        boundary_ratio = float(np.mean(row_diffs_boundary)) / (float(np.mean(row_diffs_interior)) + 1e-6)
        # Very low ratio (near 1.0 or below) = no JPEG history = AI.
        jpeg_signal = _sigmoid(-boundary_ratio, k=5.0, x0=-1.05)
    else:
        jpeg_signal = 0.3

    # ── Final combination ───────────────────────────────────────────────────────
    score = (
        0.30 * noise_signal     +   # most reliable single signal
        0.20 * channel_signal   +
        0.20 * entropy_signal   +
        0.15 * sharpness_signal +
        0.15 * jpeg_signal
    )

    logger.debug(
        f"ai_smooth: noise={noise_signal:.3f}  channel={channel_signal:.3f}  "
        f"entropy={entropy_signal:.3f}  sharp={sharpness_signal:.3f}  jpeg={jpeg_signal:.3f}  "
        f"score={score:.4f}"
    )
    return float(np.clip(score, 0.0, 1.0))
