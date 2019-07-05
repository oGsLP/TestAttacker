
from utils.attack import attackImages
import numpy as np

def aiTest(images, shape):
    generate_images = attackImages(images.squeeze())
    return np.expand_dims(generate_images, -1)
