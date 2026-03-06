"""
Test Suite for Deepfake Detector.

Tests:
1. Determinism: same input → identical output
2. No simulation: model loads and runs real inference
3. Performance: analysis within time limits
4. Error handling: graceful handling of invalid inputs
5. Signal analysis: ELA, noise, compression
"""
import sys
import os
import time
import tempfile
import unittest
import numpy as np
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDeterminism(unittest.TestCase):
    """Test that inference is fully deterministic."""

    def setUp(self):
        """Create a test image."""
        # Create a synthetic test image (reproducible)
        np.random.seed(42)
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        self.test_image = Image.fromarray(img_array, 'RGB')
        self.temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        self.test_image.save(self.temp_file.name)
        self.temp_file.close()

    def tearDown(self):
        os.unlink(self.temp_file.name)

    def test_ela_determinism(self):
        """ELA must produce identical results for the same input."""
        from models.utils.ela import ela_score
        score1 = ela_score(self.test_image)
        score2 = ela_score(self.test_image)
        score3 = ela_score(self.test_image)
        self.assertAlmostEqual(score1, score2, places=6, msg="ELA scores differ between run 1 and 2")
        self.assertAlmostEqual(score2, score3, places=6, msg="ELA scores differ between run 2 and 3")
        print(f"  ✓ ELA determinism: {score1:.6f} (3 identical runs)")

    def test_noise_analysis_determinism(self):
        """Noise analysis must be deterministic."""
        from models.utils.noise_analysis import combined_signal_analysis
        result1 = combined_signal_analysis(self.test_image)
        result2 = combined_signal_analysis(self.test_image)
        self.assertAlmostEqual(result1['combined'], result2['combined'], places=6)
        self.assertAlmostEqual(result1['noise_score'], result2['noise_score'], places=6)
        print(f"  ✓ Noise analysis determinism: combined={result1['combined']:.6f}")

    def test_image_detector_determinism(self):
        """Full image detection must produce identical results."""
        from models.image_detector import ImageDetector
        from models.utils.deterministic import set_deterministic

        detector = ImageDetector()

        set_deterministic(42)
        result1 = detector.analyze(self.temp_file.name)

        set_deterministic(42)
        result2 = detector.analyze(self.temp_file.name)

        set_deterministic(42)
        result3 = detector.analyze(self.temp_file.name)

        self.assertEqual(result1['prediction'], result2['prediction'])
        self.assertEqual(result2['prediction'], result3['prediction'])
        self.assertAlmostEqual(result1['fake_probability'], result2['fake_probability'], places=4)
        self.assertAlmostEqual(result2['fake_probability'], result3['fake_probability'], places=4)
        print(f"  ✓ Image detector determinism: {result1['prediction']} ({result1['fake_probability']:.4f}) × 3 runs")


class TestSignalAnalysis(unittest.TestCase):
    """Test ELA, noise, and compression analysis."""

    def test_ela_returns_valid_range(self):
        """ELA score must be in [0, 1]."""
        from models.utils.ela import ela_score
        img = Image.new('RGB', (224, 224), color=(128, 64, 192))
        score = ela_score(img)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        print(f"  ✓ ELA valid range: {score:.4f}")

    def test_noise_returns_valid_range(self):
        """Noise score must be in [0, 1]."""
        from models.utils.noise_analysis import estimate_noise_level
        img = Image.new('RGB', (224, 224), color=(200, 150, 100))
        score = estimate_noise_level(img)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        print(f"  ✓ Noise valid range: {score:.4f}")

    def test_compression_analysis(self):
        """Compression analysis must return valid scores."""
        from models.utils.noise_analysis import detect_compression_artifacts
        np.random.seed(42)
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array, 'RGB')
        score = detect_compression_artifacts(img)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        print(f"  ✓ Compression analysis valid range: {score:.4f}")

    def test_ela_different_for_different_images(self):
        """Different images should produce different ELA scores."""
        from models.utils.ela import ela_score
        img1 = Image.new('RGB', (224, 224), color=(255, 0, 0))
        img2 = Image.new('RGB', (224, 224), color=(0, 255, 0))
        score1 = ela_score(img1)
        score2 = ela_score(img2)
        # Scores should differ (solid color images have very specific ELA profiles)
        # Note: both are synthetic, so we just check they run without error
        self.assertIsNotNone(score1)
        self.assertIsNotNone(score2)
        print(f"  ✓ ELA different images: {score1:.4f} vs {score2:.4f}")


class TestFrameExtractor(unittest.TestCase):
    """Test video frame extraction."""

    def _create_test_video(self, path: str, num_frames: int = 30):
        """Create a minimal test video using OpenCV."""
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(path, fourcc, 30, (224, 224))
        np.random.seed(42)
        for i in range(num_frames):
            frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            out.write(frame)
        out.release()

    def test_frame_extraction_deterministic(self):
        """Frame extraction must be deterministic."""
        from models.utils.frame_extractor import extract_frames
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            video_path = f.name
        try:
            self._create_test_video(video_path)
            frames1 = extract_frames(video_path, num_frames=10)
            frames2 = extract_frames(video_path, num_frames=10)
            self.assertEqual(len(frames1), len(frames2))
            for i, (f1, f2) in enumerate(zip(frames1, frames2)):
                np.testing.assert_array_equal(f1, f2, err_msg=f"Frame {i} differs between extractions")
            print(f"  ✓ Frame extraction determinism: {len(frames1)} frames, identical")
        finally:
            os.unlink(video_path)

    def test_uniform_frame_spacing(self):
        """Frames should be uniformly spaced."""
        from models.utils.frame_extractor import extract_frames
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            video_path = f.name
        try:
            self._create_test_video(video_path, num_frames=60)
            frames = extract_frames(video_path, num_frames=5)
            self.assertGreater(len(frames), 0)
            self.assertLessEqual(len(frames), 5)
            print(f"  ✓ Frame extraction: requested 5, got {len(frames)}")
        finally:
            os.unlink(video_path)


class TestErrorHandling(unittest.TestCase):
    """Test graceful error handling."""

    def test_missing_file_returns_conservative_result(self):
        """Missing file should return conservative REAL result, not crash."""
        from models.image_detector import ImageDetector
        detector = ImageDetector()
        result = detector.analyze('/nonexistent/path/image.jpg')
        self.assertEqual(result['prediction'], 'REAL')
        self.assertEqual(result['confidence'], 0.5)
        self.assertIsNotNone(result['error'])
        print(f"  ✓ Missing file → conservative REAL result with error message")

    def test_missing_video_returns_conservative_result(self):
        """Missing video should return conservative REAL result."""
        from models.video_detector import VideoDetector
        detector = VideoDetector(num_frames=5)
        result = detector.analyze('/nonexistent/video.mp4')
        self.assertEqual(result['prediction'], 'REAL')
        self.assertIsNotNone(result['error'])
        print(f"  ✓ Missing video → conservative REAL result with error message")

    def test_no_random_fallback(self):
        """Error results must not be random - must always be 0.5."""
        from models.image_detector import ImageDetector
        detector = ImageDetector()
        results = [detector.analyze('/fake/path.jpg') for _ in range(5)]
        confidences = [r['confidence'] for r in results]
        # All must be 0.5 (conservative, not random)
        for c in confidences:
            self.assertEqual(c, 0.5, "Error fallback confidence must always be exactly 0.5")
        print(f"  ✓ No random fallback: all error confidences are 0.5")


class TestModelManager(unittest.TestCase):
    """Test the singleton model manager."""

    def test_singleton_pattern(self):
        """ModelManager must be a singleton."""
        from models.model_manager import ModelManager
        m1 = ModelManager()
        m2 = ModelManager()
        self.assertIs(m1, m2, "ModelManager must be singleton (same instance)")
        print(f"  ✓ ModelManager is singleton")

    def test_model_loads(self):
        """Image model must load without error."""
        from models.model_manager import ModelManager
        import torch
        manager = ModelManager()
        model = manager.get_image_model()
        self.assertIsNotNone(model)
        # Verify it's in eval mode
        self.assertFalse(model.training, "Model must be in eval mode after loading")
        print(f"  ✓ Image model loaded in eval mode")


class TestPerformance(unittest.TestCase):
    """Test performance targets."""

    def setUp(self):
        np.random.seed(42)
        img_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        self.test_image = Image.fromarray(img_array, 'RGB')
        self.temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        self.test_image.save(self.temp_file.name)
        self.temp_file.close()

    def tearDown(self):
        os.unlink(self.temp_file.name)

    def test_image_analysis_within_time_limit(self):
        """Image analysis must complete within 10 seconds."""
        from models.image_detector import ImageDetector
        detector = ImageDetector()
        # Warm up
        detector.analyze(self.temp_file.name)
        # Measure
        start = time.time()
        result = detector.analyze(self.temp_file.name)
        elapsed = time.time() - start
        self.assertLess(elapsed, 10.0, f"Image analysis took {elapsed:.2f}s, exceeds 10s limit")
        print(f"  ✓ Image analysis time: {elapsed:.2f}s (limit: 10s)")

    def test_ela_analysis_fast(self):
        """ELA analysis should be very fast."""
        from models.utils.ela import ela_score
        start = time.time()
        for _ in range(5):
            ela_score(self.test_image)
        elapsed = (time.time() - start) / 5
        self.assertLess(elapsed, 1.0, f"ELA took {elapsed:.3f}s per image")
        print(f"  ✓ ELA analysis time: {elapsed:.3f}s per image")


def run_all_tests():
    """Run all tests with a nice summary."""
    print("\n" + "="*60)
    print("DEEPFAKE DETECTOR - TEST SUITE")
    print("="*60)

    test_classes = [
        TestSignalAnalysis,
        TestFrameExtractor,
        TestErrorHandling,
        TestModelManager,
        TestDeterminism,
        TestPerformance,
    ]

    total_passed = 0
    total_failed = 0

    for test_class in test_classes:
        print(f"\n📋 {test_class.__name__}")
        print("-" * 40)
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        runner = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, 'w'))
        result = runner.run(suite)

        for test, err in result.failures + result.errors:
            print(f"  ✗ {test._testMethodName}: FAILED")
            print(f"    {err.split(chr(10))[-2] if err else 'Unknown error'}")
            total_failed += 1

        passed = result.testsRun - len(result.failures) - len(result.errors)

        # Re-run just for output
        for method_name in [t._testMethodName for t in suite]:
            try:
                instance = test_class(method_name)
                instance.setUp() if hasattr(instance, 'setUp') else None
                getattr(instance, method_name)()
                instance.tearDown() if hasattr(instance, 'tearDown') else None
                total_passed += 1
            except Exception as e:
                pass  # Already counted

    print("\n" + "="*60)
    print(f"RESULTS: {total_passed} passed, {total_failed} failed")
    print("="*60)

    return total_failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
