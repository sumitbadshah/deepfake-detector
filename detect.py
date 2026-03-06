"""
Command-line inference script for deepfake detection.

Usage:
    python detect.py image.jpg
    python detect.py video.mp4
    python detect.py --batch /path/to/folder/
    python detect.py image.jpg --verify-determinism
"""
import argparse
import json
import sys
import os
import time
from pathlib import Path

# Set seeds immediately
from models.utils.deterministic import set_deterministic
set_deterministic(42)


def detect_file(path: str, file_type: str = None) -> dict:
    """Run detection on a single file."""
    ext = Path(path).suffix.lower().lstrip('.')
    image_exts = {'png', 'jpg', 'jpeg', 'webp', 'bmp', 'tiff'}
    video_exts = {'mp4', 'avi', 'mov', 'mkv', 'webm', 'flv'}

    if file_type is None:
        if ext in image_exts:
            file_type = 'image'
        elif ext in video_exts:
            file_type = 'video'
        else:
            return {'error': f'Unknown file type: .{ext}', 'prediction': 'REAL', 'confidence': 0.5}

    if file_type == 'image':
        from models.image_detector import ImageDetector
        detector = ImageDetector()
        return detector.analyze(path)
    else:
        from models.video_detector import VideoDetector
        detector = VideoDetector(num_frames=20)
        return detector.analyze(path)


def print_result(result: dict, path: str):
    """Pretty-print detection result."""
    pred = result.get('prediction', 'UNKNOWN')
    conf = result.get('confidence', 0.5)
    fake_prob = result.get('fake_probability', 0.5)
    elapsed = result.get('analysis_time', 0)
    error = result.get('error')

    # Color codes
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

    color = GREEN if pred == 'REAL' else RED
    emoji = '✅' if pred == 'REAL' else '🚨'

    print(f"\n{'='*55}")
    print(f"  File: {Path(path).name}")
    print(f"{'='*55}")
    print(f"  {emoji} Verdict: {BOLD}{color}{pred}{RESET}")
    print(f"  📊 Fake Probability: {fake_prob*100:.1f}%")
    print(f"  🎯 Confidence: {conf*100:.1f}%")
    print(f"  ⏱  Analysis Time: {elapsed:.2f}s")

    if result.get('image_size'):
        print(f"  📐 Image Size: {result['image_size']}")
    if result.get('frames_analyzed'):
        print(f"  🎞  Frames Analyzed: {result['frames_analyzed']}")

    scores = result.get('scores', {})
    if scores:
        print(f"\n  {CYAN}Score Breakdown:{RESET}")
        for key, val in scores.items():
            if key != 'ensemble_final':
                label = key.replace('_', ' ').title()
                bar_len = int(val * 20)
                bar = '█' * bar_len + '░' * (20 - bar_len)
                color_bar = RED if val > 0.6 else YELLOW if val > 0.4 else GREEN
                print(f"    {label:<28} [{color_bar}{bar}{RESET}] {val*100:.1f}%")

    if error:
        print(f"\n  ⚠️  Warning: {YELLOW}{error}{RESET}")

    print(f"{'='*55}\n")


def verify_determinism(path: str, runs: int = 3):
    """Verify that outputs are identical across multiple runs."""
    print(f"\nVerifying determinism ({runs} runs)...")
    results = []
    for i in range(runs):
        set_deterministic(42)
        result = detect_file(path)
        results.append(result)
        print(f"  Run {i+1}: {result['prediction']} ({result['fake_probability']:.6f})")

    all_same = all(
        abs(results[0]['fake_probability'] - r['fake_probability']) < 1e-5 and
        results[0]['prediction'] == r['prediction']
        for r in results[1:]
    )

    if all_same:
        print(f"\n  ✅ DETERMINISM VERIFIED: All {runs} runs produced identical results")
    else:
        print(f"\n  ❌ DETERMINISM FAILED: Results differed between runs!")

    return all_same


def batch_detect(folder: str, output_json: str = None):
    """Detect all media files in a folder."""
    image_exts = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
    video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    all_exts = image_exts | video_exts

    files = [p for p in Path(folder).rglob('*') if p.suffix.lower() in all_exts]
    print(f"\nFound {len(files)} media files in {folder}")

    results = []
    for i, file_path in enumerate(files):
        print(f"\n[{i+1}/{len(files)}] {file_path.name}")
        set_deterministic(42)
        result = detect_file(str(file_path))
        result['file'] = str(file_path)
        results.append(result)
        print_result(result, str(file_path))

    # Summary
    fakes = [r for r in results if r.get('prediction') == 'FAKE']
    reals = [r for r in results if r.get('prediction') == 'REAL']
    print(f"\n{'='*55}")
    print(f"BATCH SUMMARY: {len(files)} files")
    print(f"  ✅ REAL: {len(reals)}")
    print(f"  🚨 FAKE: {len(fakes)}")
    print(f"{'='*55}")

    if output_json:
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_json}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Deepfake Detection CLI - DenseNet121 + GenConViT',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python detect.py photo.jpg
  python detect.py deepfake_video.mp4
  python detect.py photo.jpg --verify-determinism
  python detect.py --batch ./media_folder/ --output results.json
  python detect.py photo.jpg --json
        """
    )
    parser.add_argument('file', nargs='?', help='Image or video file to analyze')
    parser.add_argument('--batch', help='Folder to analyze all media in batch')
    parser.add_argument('--output', help='Output JSON file for batch results')
    parser.add_argument('--json', action='store_true', help='Output result as JSON')
    parser.add_argument('--verify-determinism', action='store_true', help='Run 3 times to verify determinism')
    parser.add_argument('--runs', type=int, default=3, help='Number of runs for determinism check')

    args = parser.parse_args()

    if args.batch:
        batch_detect(args.batch, args.output)
        return

    if not args.file:
        parser.print_help()
        return

    if not os.path.exists(args.file):
        print(f"Error: File not found: {args.file}")
        sys.exit(1)

    if args.verify_determinism:
        verify_determinism(args.file, runs=args.runs)
        return

    set_deterministic(42)
    result = detect_file(args.file)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_result(result, args.file)


if __name__ == '__main__':
    main()
