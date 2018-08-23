import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
from PIL import Image
import cv2
#cmap = plt.get_cmap('tab20b')
#colors = [cmap(i) for i in np.linspace(0,1,20)]
#print(colors)

# img_path = 'test_examples/dog-cycle-car.png'
# img = np.array(Image.open(img_path))
# h, w, c = img.shape
# dim_diff = np.abs(h - w)
# # h > w, left and rigth padding
# # h < w, upper and lower padding
# pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
# if h <= w:
#     pad = ((pad1, pad2), (0, 0), (0, 0))
# else:
#     pad = ((0, 0), (pad1, pad2), (0, 0))
#
# input_img = np.pad(img, pad, 'constant', constant_values=127.5) / 255.0
# input_img = resize(input_img, (416,416, 3), mode='reflect')
# input_img = np.transpose(input_img, (2, 0, 1))
