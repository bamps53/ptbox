import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import cv2

def normalize(img):
    img = img - img.min()
    img = img/img.max()
    return img


def show_image(img):
    img = normalize(img)
    plt.imshow(np.transpose(img, [1, 2, 0]))
    plt.show()