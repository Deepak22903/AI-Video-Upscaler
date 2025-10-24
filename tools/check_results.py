import cv2, glob, numpy as np
files = sorted(glob.glob('/home/deepak/data/btechProject/results/*_out.*'))
for f in files:
    img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f, 'UNREADABLE')
    else:
        print(f, 'shape', img.shape, 'dtype', img.dtype, 'min', int(img.min()), 'max', int(img.max()), 'mean', float(img.mean()))
