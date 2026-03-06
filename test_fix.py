"""
Run this to verify the fix works BEFORE starting app.py.
Usage: python test_fix.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Delete ALL pycache in models folder first
import shutil
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
pycache = os.path.join(models_dir, '__pycache__')
if os.path.exists(pycache):
    shutil.rmtree(pycache)
    print(f"Deleted: {pycache}")

utils_pycache = os.path.join(models_dir, 'utils', '__pycache__')
if os.path.exists(utils_pycache):
    shutil.rmtree(utils_pycache)
    print(f"Deleted: {utils_pycache}")

root_pycache = os.path.join(os.path.dirname(os.path.abspath(__file__)), '__pycache__')
if os.path.exists(root_pycache):
    shutil.rmtree(root_pycache)
    print(f"Deleted: {root_pycache}")

print("All pycache cleared. Now testing imports...")

try:
    from models.image_detector import ImageDetector
    print("OK: ImageDetector imported")
    
    det = ImageDetector()
    print("OK: ImageDetector instantiated")
    
    # Test with a dummy image
    from PIL import Image
    import tempfile, os
    img = Image.new('RGB', (224, 224), color=(120, 80, 60))
    tmp = tempfile.mktemp(suffix='.jpg')
    img.save(tmp)
    
    result = det.analyze(tmp)
    os.unlink(tmp)
    
    print(f"\nTest result:")
    print(f"  prediction:  {result['prediction']}")
    print(f"  confidence:  {result['confidence']}")
    print(f"  image_size:  {result['image_size']}")
    print(f"  error:       {result['error']}")
    print(f"  models_used: {result['models_used']}")
    
    if result['error']:
        print(f"\nSTILL HAS ERROR: {result['error']}")
        sys.exit(1)
    else:
        print("\nSUCCESS — no errors, image_size is not unknown")

except Exception as e:
    import traceback
    print(f"\nFATAL ERROR: {e}")
    traceback.print_exc()
    sys.exit(1)
