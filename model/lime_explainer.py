import numpy as np
import cv2

def generate_lime(model, img):
    """
    Simple placeholder LIME-style explanation
    (you can upgrade later to real LIME)
    """

    img_resized = cv2.resize(img, (224, 224))

    # create fake segmentation mask
    mask = np.zeros(img_resized.shape[:2], dtype=np.uint8)

    h, w = mask.shape
    mask[h//4:3*h//4, w//4:3*w//4] = 1

    # highlight region
    highlighted = img_resized.copy()
    highlighted[mask == 0] = highlighted[mask == 0] * 0.3

    return highlighted