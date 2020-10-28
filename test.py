import cv2
import numpy as np
from scipy import stats
from skimage.measure import block_reduce
import matplotlib.pyplot as plt

# playground for testing code snippets

test_file = './CMFD/00000/00000_Mask.jpg'

img = cv2.imread(test_file, cv2.IMREAD_GRAYSCALE)
img = block_reduce(img, block_size=(8, 8), func=np.mean)
print(f"Shape after downsamping: {img.shape}")
plt.figure()
plt.imshow(img, cmap='gray')
plt.show()
img = img.reshape(1024 * 1024)
img = stats.zscore(img, axis=0)
print(f'Image Dimension: {img.shape}')
print(f'mean:{img.mean(axis=0)}\n stdev: {img.std(axis=0)}')
