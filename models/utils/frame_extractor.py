"""
Video Frame Extraction Utilities - Deterministic Implementation.

Extracts uniformly spaced frames from video files for analysis.
Uses OpenCV for broad codec support without GPU requirements.
"""
import cv2
import numpy as np
from typing import List, Tuple
import logging
import os

logger = logging.getLogger(__name__)


def extract_frames(
    video_path: str,
    num_frames: int = 20,
    target_size: Tuple[int, int] = (224, 224)
) -> List[np.ndarray]:
    """
    Extract uniformly spaced frames from a video.
    
    Deterministic: same video + same num_frames = same frames.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract
        target_size: (width, height) for resizing frames
    
    Returns:
        List of numpy arrays (H, W, 3) in BGR format
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if total_frames <= 0:
            logger.warning("Cannot determine frame count, reading sequentially")
            return _extract_sequential(cap, num_frames, target_size)

        # Calculate uniformly spaced frame indices (deterministic)
        actual_frames = min(num_frames, total_frames)
        if actual_frames < 2:
            indices = [0]
        else:
            # Linspace gives uniform spacing - fully deterministic
            indices = [
                int(round(i * (total_frames - 1) / (actual_frames - 1)))
                for i in range(actual_frames)
            ]
            # Remove duplicates while preserving order
            seen = set()
            indices = [x for x in indices if not (x in seen or seen.add(x))]

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                if target_size:
                    frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
                frames.append(frame)

        logger.info(f"Extracted {len(frames)}/{len(indices)} frames from {video_path}")
        return frames

    finally:
        cap.release()


def _extract_sequential(cap, num_frames: int, target_size: Tuple[int, int]) -> List[np.ndarray]:
    """Fallback: read frames sequentially when total count unknown."""
    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)

    if not all_frames:
        return []

    # Uniformly sample from collected frames
    total = len(all_frames)
    indices = [int(round(i * (total - 1) / (num_frames - 1))) for i in range(min(num_frames, total))]
    seen = set()
    indices = [x for x in indices if not (x in seen or seen.add(x))]

    frames = []
    for idx in indices:
        frame = all_frames[idx]
        if target_size:
            frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
        frames.append(frame)

    return frames


def get_video_metadata(video_path: str) -> dict:
    """Get metadata about a video file."""
    if not os.path.exists(video_path):
        return {}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}

    try:
        metadata = {
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        duration = metadata['total_frames'] / metadata['fps'] if metadata['fps'] > 0 else 0
        metadata['duration_seconds'] = round(duration, 2)
        return metadata
    finally:
        cap.release()
