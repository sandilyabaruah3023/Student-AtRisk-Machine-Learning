import sys

print("Python Executable:", sys.executable)
print("\nChecking libraries...\n")

try:
    import numpy as np
    print("✅ NumPy OK — version:", np.__version__)
except ImportError:
    print("❌ NumPy not installed")

try:
    import matplotlib
    import matplotlib.pyplot as plt
    print("✅ Matplotlib OK — version:", matplotlib.__version__)
except ImportError:
    print("❌ Matplotlib not installed")


try:
    import pandas as pd
    print("✅ Pandas OK — version:", pd.__version__)
except ImportError:
    print("❌ Pandas not installed")

try:
    import cv2
    print("✅ OpenCV OK — version:", cv2.__version__)
except ImportError:
    print("❌ OpenCV not installed")

print("\n✅ Environment check complete!")
