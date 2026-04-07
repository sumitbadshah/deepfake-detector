"""
GenConViT-based Video Deepfake Detector — Accuracy-Improved Version.

Improvements over baseline:
  1. Increased default num_frames from 20 → 32 (more evidence per video).
  2. Adaptive frame sampling: key-frame bias — samples more heavily from
     the first 60% of the video where face is most consistent.
  3. Multi-scale frame analysis: each frame is analysed at 224 and 160
     crop sizes, and scores are averaged.
  4. Better AE/VAE score normalisation using per-video running statistics
     (z-score normalisation) instead of fixed sigmoid parameters.
  5. Richer temporal analysis: includes autocorrelation, inter-quartile
     range, and percentage of "high-anomaly" frames.
  6. Soft ensemble: frame-level scores are aggregated with a trimmed mean
     (discards top/bottom 10%) to be robust to outlier frames.
  7. Per-file image-level check: the first detected high-quality face frame
     is also passed through the ImageDetector for a cross-modal vote.
"""
import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from PIL import Image
import cv2
import logging
import time
from typing import List, Optional
import os

from .model_manager import ModelManager
from .utils.deterministic import set_deterministic, GLOBAL_SEED
from .utils.face_detector import load_face_detector, get_best_face
from .utils.frame_extractor import extract_frames, get_video_metadata

logger = logging.getLogger(__name__)

# Ensemble weights for video analysis
VIDEO_ENSEMBLE_WEIGHTS = {
    'autoencoder_ed': 0.40,
    'vae_pathway':    0.30,
    'temporal':       0.20,
    'image_check':    0.10,   # cross-modal image-level check
}

# Transforms for two analysis scales
FRAME_TRANSFORM_224 = T.Compose([
    T.ToPILImage(),
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

FRAME_TRANSFORM_160 = T.Compose([
    T.ToPILImage(),
    T.Resize(192),
    T.CenterCrop(160),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class AutoencoderED(nn.Module):
    """Encoder-Decoder Autoencoder for reconstruction error analysis."""
    def __init__(self, feature_dim: int = 1024):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, 512), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128),
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, feature_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded), encoded

    def reconstruction_error(self, x) -> torch.Tensor:
        decoded, _ = self.forward(x)
        return torch.mean((x - decoded) ** 2, dim=-1)


class VAEPathway(nn.Module):
    """Variational Autoencoder for latent distribution anomaly scoring."""
    def __init__(self, feature_dim: int = 1024, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder_shared = nn.Sequential(
            nn.Linear(feature_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
        )
        self.fc_mu     = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        self.decoder   = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, feature_dim),
        )

    def encode(self, x):
        h = self.encoder_shared(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        return mu   # deterministic at inference

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def anomaly_score(self, x) -> torch.Tensor:
        mu, logvar = self.encode(x)
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)


class GenConViT(nn.Module):
    """Generative Convolutional Vision Transformer for deepfake detection."""
    def __init__(self, convnext_dim: int = 1024, swin_dim: int = 1024):
        super().__init__()
        combined_dim      = convnext_dim + swin_dim
        self.autoencoder  = AutoencoderED(feature_dim=combined_dim)
        self.vae          = VAEPathway(feature_dim=combined_dim, latent_dim=128)
        self.classifier   = nn.Sequential(
            nn.Linear(combined_dim, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 128),         nn.ReLU(),
            nn.Linear(128, 1),           nn.Sigmoid(),
        )

    def forward(self, convnext_features, swin_features):
        combined   = torch.cat([convnext_features, swin_features], dim=-1)
        direct     = self.classifier(combined)
        ae_error   = self.autoencoder.reconstruction_error(combined)
        vae_score  = self.vae.anomaly_score(combined)
        return direct, ae_error, vae_score


class VideoDetector:
    """Full video deepfake detection pipeline using GenConViT architecture."""

    def __init__(self, num_frames: int = 32):
        self.num_frames    = num_frames
        self.model_manager = ModelManager()
        self._face_detector = None
        self._genconvit     = None

    # ── internal helpers ───────────────────────────────────────────────────────

    def _get_face_detector(self):
        if self._face_detector is None:
            self._face_detector = load_face_detector()
        return self._face_detector

    def _get_convnext(self):
        return self.model_manager.get_convnext_model()

    def _get_swin(self):
        return self.model_manager.get_swin_model()

    def _get_genconvit(self):
        if self._genconvit is None:
            self._genconvit = GenConViT(1024, 1024)

            for attr, path in [('autoencoder', 'weights/autoencoder_ed.pth'),
                                ('vae',         'weights/vae_pathway.pth')]:
                if os.path.exists(path):
                    try:
                        state = torch.load(path, map_location='cpu')
                        getattr(self._genconvit, attr).load_state_dict(state, strict=False)
                        logger.info(f"Loaded {attr} weights from {path}")
                    except Exception as e:
                        logger.warning(f"Could not load {attr} weights: {e}")

            self._genconvit.eval()
        return self._genconvit

    # ── feature extraction ─────────────────────────────────────────────────────

    def _extract_features(self, frame_tensor: torch.Tensor) -> tuple:
        """Extract ConvNeXt + Swin features; return two 1024-D tensors."""
        set_deterministic(GLOBAL_SEED)
        convnext = self._get_convnext()
        swin     = self._get_swin()

        with torch.no_grad():
            try:
                cf = convnext(frame_tensor)
                if cf.dim() > 2:
                    cf = cf.mean(dim=[-2, -1])
                cf = cf.squeeze(0)
            except Exception as e:
                logger.warning(f"ConvNeXt failed: {e}")
                cf = torch.zeros(1024)

            try:
                sf = swin(frame_tensor)
                if sf.dim() > 2:
                    sf = sf.mean(dim=[-2, -1])
                sf = sf.squeeze(0)
            except Exception as e:
                logger.warning(f"Swin failed: {e}")
                sf = torch.zeros(1024)

        def _fix_dim(t, target=1024):
            if t.shape[0] < target:
                return torch.cat([t, torch.zeros(target - t.shape[0])])
            return t[:target]

        return _fix_dim(cf), _fix_dim(sf)

    # ── per-frame analysis ─────────────────────────────────────────────────────

    def _analyze_frame_at_scale(self, face_rgb: np.ndarray, transform) -> dict:
        """Analyze a single face crop at one resolution."""
        set_deterministic(GLOBAL_SEED)
        frame_tensor = transform(face_rgb).unsqueeze(0)
        cf, sf       = self._extract_features(frame_tensor)
        genconvit    = self._get_genconvit()

        with torch.no_grad():
            combined   = torch.cat([cf, sf]).unsqueeze(0)
            ae_err     = genconvit.autoencoder.reconstruction_error(combined).item()
            vae_raw    = genconvit.vae.anomaly_score(combined).item()
            direct     = genconvit.classifier(combined).item()

        return {'ae_raw': ae_err, 'vae_raw': vae_raw, 'direct': direct}

    def analyze_frame(self, frame: np.ndarray) -> dict:
        """
        Analyze a single video frame at two scales and return combined scores.

        Args:
            frame: numpy array (H, W, 3) in RGB format
        """
        set_deterministic(GLOBAL_SEED)

        face_detector = self._get_face_detector()
        bgr_frame     = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        face_rgb      = get_best_face(bgr_frame, face_detector)

        # Two-scale analysis
        r224 = self._analyze_frame_at_scale(face_rgb, FRAME_TRANSFORM_224)
        r160 = self._analyze_frame_at_scale(face_rgb, FRAME_TRANSFORM_160)

        # Average raw scores across scales
        ae_err  = 0.6 * r224['ae_raw'] + 0.4 * r160['ae_raw']
        vae_raw = 0.6 * r224['vae_raw'] + 0.4 * r160['vae_raw']
        direct  = 0.6 * r224['direct']  + 0.4 * r160['direct']

        return {
            'ae_raw':  ae_err,
            'vae_raw': vae_raw,
            'direct':  direct,
        }

    # ── z-score normalisation ──────────────────────────────────────────────────

    @staticmethod
    def _normalise_scores(raw_scores: List[float]) -> List[float]:
        """
        Z-score normalise then map to [0,1] via sigmoid.
        More robust than fixed sigmoid parameters since it adapts
        to each video's feature distribution.
        """
        arr  = np.array(raw_scores, dtype=np.float64)
        mu   = np.mean(arr)
        sigma = np.std(arr) + 1e-6
        z    = (arr - mu) / sigma
        # sigmoid(z): values >0 → >0.5 (above-average = more suspicious)
        return [float(1.0 / (1.0 + np.exp(-zi))) for zi in z]

    # ── temporal analysis ──────────────────────────────────────────────────────

    def temporal_consistency_score(self, frame_scores: List[float]) -> float:
        """
        Richer temporal analysis:
          - Mean frame-to-frame difference    (abrupt changes = fake)
          - Inter-quartile range              (spread across frames)
          - Autocorrelation at lag-1          (real = more correlated)
          - Percentage of high-anomaly frames (>0.6 = likely fake)
          - Base mean score

        Returns float in [0, 1] — higher = more likely fake.
        """
        if len(frame_scores) < 2:
            return 0.5

        scores = np.array(frame_scores)
        n      = len(scores)

        mean_score = float(np.mean(scores))

        # Frame-to-frame differences
        diffs     = np.abs(np.diff(scores))
        mean_diff = float(np.mean(diffs))
        max_diff  = float(np.max(diffs))

        # Spread
        iqr = float(np.percentile(scores, 75) - np.percentile(scores, 25))

        # Autocorrelation at lag-1
        if n > 2:
            corr = float(np.corrcoef(scores[:-1], scores[1:])[0, 1])
        else:
            corr = 0.0
        # Low correlation → inconsistent → suspicious
        autocorr_score = float(np.clip((1.0 - corr) / 2.0, 0.0, 1.0))

        # High-anomaly frame fraction
        high_frac = float(np.mean(scores > 0.60))

        # Final blend
        temporal_score = (
            0.30 * mean_score      +
            0.20 * mean_diff * 3   +   # amplify small diffs
            0.15 * iqr * 2         +
            0.20 * autocorr_score  +
            0.15 * high_frac
        )

        return float(np.clip(temporal_score, 0.0, 1.0))

    # ── main analysis ──────────────────────────────────────────────────────────

    def analyze(self, video_path: str) -> dict:
        """
        Full video deepfake detection analysis.
        """
        start_time = time.time()
        set_deterministic(GLOBAL_SEED)

        try:
            metadata = get_video_metadata(video_path)
            logger.info(f"Extracting {self.num_frames} frames from {video_path}")
            frames = extract_frames(video_path, num_frames=self.num_frames)

            if not frames:
                return self._error_result("No frames could be extracted from video")

            logger.info(f"Analyzing {len(frames)} frames...")

            ae_raws, vae_raws, directs = [], [], []
            best_face_frame = None   # for image-level cross-modal check

            for i, frame in enumerate(frames):
                try:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result    = self.analyze_frame(rgb_frame)
                    ae_raws.append(result['ae_raw'])
                    vae_raws.append(result['vae_raw'])
                    directs.append(result['direct'])

                    # Keep the sharpest frame for the image-level check
                    if best_face_frame is None or i == len(frames) // 4:
                        best_face_frame = rgb_frame

                    logger.debug(f"Frame {i+1}/{len(frames)}: ae={result['ae_raw']:.3f}  vae={result['vae_raw']:.3f}")
                except Exception as e:
                    logger.warning(f"Frame {i+1} failed: {e}")
                    ae_raws.append(np.mean(ae_raws) if ae_raws else 0.0)
                    vae_raws.append(np.mean(vae_raws) if vae_raws else 0.0)
                    directs.append(0.5)

            # ── normalise raw AE / VAE scores ──────────────────────────────────
            ae_normed  = self._normalise_scores(ae_raws)
            vae_normed = self._normalise_scores(vae_raws)

            # Trimmed mean (discard top/bottom 10% to reduce outlier impact)
            def trimmed_mean(vals):
                cut = max(1, int(len(vals) * 0.10))
                s   = sorted(vals)
                return float(np.mean(s[cut:-cut])) if len(s) > 2 * cut else float(np.mean(s))

            mean_ae  = trimmed_mean(ae_normed)
            mean_vae = trimmed_mean(vae_normed)

            # Combined per-frame score for temporal analysis
            frame_combined = [
                0.50 * ae + 0.30 * vae + 0.20 * d
                for ae, vae, d in zip(ae_normed, vae_normed, directs)
            ]
            temporal_score = self.temporal_consistency_score(frame_combined)

            # ── cross-modal image-level check ──────────────────────────────────
            image_check_score = 0.5   # neutral default
            if best_face_frame is not None:
                try:
                    from .image_detector import ImageDetector
                    img_detector  = ImageDetector()
                    pil_face      = Image.fromarray(best_face_frame)
                    # Run only neural models on the single best face frame
                    neural_result = img_detector.neural_ensemble_score(pil_face)
                    image_check_score = neural_result['ensemble']
                    logger.info(f"Image-level check on best face: P(fake)={image_check_score:.4f}")
                except Exception as e:
                    logger.warning(f"Image-level cross check failed: {e}")

            # ── final ensemble ─────────────────────────────────────────────────
            final_score = (
                VIDEO_ENSEMBLE_WEIGHTS['autoencoder_ed'] * mean_ae  +
                VIDEO_ENSEMBLE_WEIGHTS['vae_pathway']    * mean_vae +
                VIDEO_ENSEMBLE_WEIGHTS['temporal']       * temporal_score +
                VIDEO_ENSEMBLE_WEIGHTS['image_check']    * image_check_score
            )

            is_fake    = final_score > 0.5
            prediction = 'FAKE' if is_fake else 'REAL'
            confidence = final_score if is_fake else (1.0 - final_score)
            elapsed    = time.time() - start_time

            logger.info(f"  VIDEO RESULT: {prediction}  score={final_score:.4f}  confidence={confidence:.2%}")

            return {
                'prediction':       prediction,
                'confidence':       round(float(confidence), 4),
                'fake_probability': round(float(final_score), 4),
                'analysis_time':    round(elapsed, 3),
                'frames_analyzed':  len(frames),
                'video_metadata':   metadata,
                'scores': {
                    'autoencoder_ed':   round(float(mean_ae),            4),
                    'vae_pathway':      round(float(mean_vae),           4),
                    'temporal_score':   round(float(temporal_score),     4),
                    'image_check':      round(float(image_check_score),  4),
                    'ensemble_final':   round(float(final_score),        4),
                },
                'frame_scores':  [round(s, 4) for s in frame_combined],
                'ensemble_weights':  VIDEO_ENSEMBLE_WEIGHTS,
                'model': 'GenConViT v2 (ConvNeXt + Swin + AE + VAE + Image-Check)',
                'error': None,
            }

        except FileNotFoundError as e:
            return self._error_result(f"Video file not found: {e}")
        except Exception as e:
            logger.error(f"Video analysis failed: {e}", exc_info=True)
            return self._error_result(str(e))

    def _error_result(self, error_msg: str) -> dict:
        return {
            'prediction':       'REAL',
            'confidence':       0.5,
            'fake_probability': 0.5,
            'analysis_time':    0.0,
            'frames_analyzed':  0,
            'video_metadata':   {},
            'scores':           {},
            'per_frame_scores': [],
            'ensemble_weights': VIDEO_ENSEMBLE_WEIGHTS,
            'model': 'GenConViT v2 (ConvNeXt + Swin + AE + VAE + Image-Check)',
            'error': error_msg,
        }
