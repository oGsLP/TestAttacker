
from utils.attack import attackImages
import numpy as np

def aiTest(images, shape):
    attacked_images = attackImages(images.squeeze())
    generate_imges = (np.expand_dims(attacked_images, -1))*255.0
    return generate_imges
