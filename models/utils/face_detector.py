"""
Face Detection Utilities.

Primary: OpenCV DNN-based face detection (Caffe model)
Fallback: Haar cascade (always available in opencv)
No dlib dependency required for basic functionality.
"""
import cv2
import numpy as np
from PIL import Image
import logging
import urllib.request
import os

logger = logging.getLogger(__name__)

# Haar cascade is built into OpenCV
HAAR_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# OpenCV DNN face detector models (downloaded on first use)
DNN_PROTOTXT_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
DNN_MODEL_URL = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
DNN_PROTOTXT_PATH = "weights/face_detector.prototxt"
DNN_MODEL_PATH = "weights/face_detector.caffemodel"


def load_face_detector():
    """
    Load best available face detector.
    Priority: OpenCV DNN → Haar Cascade
    """
    # Try DNN detector
    if os.path.exists(DNN_MODEL_PATH) and os.path.exists(DNN_PROTOTXT_PATH):
        try:
            net = cv2.dnn.readNetFromCaffe(DNN_PROTOTXT_PATH, DNN_MODEL_PATH)
            logger.info("Loaded OpenCV DNN face detector")
            return ('dnn', net)
        except Exception as e:
            logger.warning(f"DNN face detector failed to load: {e}")

    # Fall back to Haar cascade (always available)
    try:
        cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
        if not cascade.empty():
            logger.info("Using Haar cascade face detector")
            return ('haar', cascade)
    except Exception as e:
        logger.warning(f"Haar cascade failed: {e}")

    logger.warning("No face detector available - will use full image")
    return ('none', None)


def detect_faces(image: np.ndarray, detector=None) -> list:
    """
    Detect faces in an image.
    
    Args:
        image: numpy array (H, W, 3) BGR
        detector: tuple from load_face_detector()
    
    Returns:
        list of (x, y, w, h) tuples for detected faces
    """
    if detector is None:
        detector = load_face_detector()

    detector_type, detector_obj = detector

    if detector_type == 'dnn':
        return _detect_faces_dnn(image, detector_obj)
    elif detector_type == 'haar':
        return _detect_faces_haar(image, detector_obj)
    else:
        # No detector - return full image as single "face"
        h, w = image.shape[:2]
        return [(0, 0, w, h)]


def _detect_faces_dnn(image: np.ndarray, net) -> list:
    """Detect faces using OpenCV DNN."""
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)),
        1.0, (300, 300),
        (104.0, 177.0, 123.0)
    )
    net.setInput(blob)
    detections = net.forward()

    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                faces.append((x1, y1, x2-x1, y2-y1))

    return faces


def _detect_faces_haar(image: np.ndarray, cascade) -> list:
    """Detect faces using Haar cascade."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    if len(faces) == 0:
        return []
    return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]


def crop_face(image: np.ndarray, face_bbox: tuple, target_size: int = 224, padding: float = 0.2) -> np.ndarray:
    """
    Crop and resize a face from an image.
    
    Args:
        image: BGR numpy array
        face_bbox: (x, y, w, h)
        target_size: output square size
        padding: fraction of face size to add as padding
    
    Returns:
        numpy array (target_size, target_size, 3) RGB
    """
    x, y, w, h = face_bbox
    ih, iw = image.shape[:2]

    # Add padding
    pad_x = int(w * padding)
    pad_y = int(h * padding)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(iw, x + w + pad_x)
    y2 = min(ih, y + h + pad_y)

    face_crop = image[y1:y2, x1:x2]

    if face_crop.size == 0:
        face_crop = image

    # Convert BGR to RGB
    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    return face_resized


def get_best_face(image: np.ndarray, detector=None) -> np.ndarray:
    """
    Get the largest detected face from an image, or full image if none found.
    
    Returns:
        numpy array (224, 224, 3) RGB
    """
    faces = detect_faces(image, detector)

    if not faces:
        # No face detected - use center crop of full image
        h, w = image.shape[:2]
        size = min(h, w)
        y_start = (h - size) // 2
        x_start = (w - size) // 2
        crop = image[y_start:y_start+size, x_start:x_start+size]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)

    # Pick largest face
    largest_face = max(faces, key=lambda f: f[2] * f[3])
    return crop_face(image, largest_face)
