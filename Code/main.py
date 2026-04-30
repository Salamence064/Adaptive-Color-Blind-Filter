import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import argparse


#define daltonization transform matrices
#matrices based on: https://github.com/joergdietrich/daltonize/blob/main/daltonize/daltonize.py
DALTONIZATION_MATRICES = {
    "protanopia": np.array([
        [0, 0.90822864, 0.008192],
        [0, 1, 0],
        [0, 0, 1]
    ]),
    "deuteranopia": np.array([
        [1, 0, 0], 
        [1.10104433,  0, -0.00901975], 
        [0, 0, 1]
    ]),
    "tritanopia": np.array([
        [1, 0, 0],
        [0, 1, 0],
        [-0.15773032,  1.19465634, 0]
    ])
}

def apply_transform(image, matrix):
    # apply per-pixel transform: result[x,y,:] = kernel @ image[x,y,:]
    out = np.tensordot(image, matrix.T, axes=([2], [0]))  #HxWx3
    out = np.clip(out, 0, 255).astype(np.uint8)

    return out

if __name__ == "__main__":
    outputDir = '../Outputs/'
    os.makedirs(outputDir, exist_ok=True)

    #use format 'python3 main.py {pathname}' to run code
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to the input image")
    args = parser.parse_args()

    image = cv2.imread(args.image_path)

    #select matrix to apply
    matrix_name = "tritanopia"
    applied_matrix = DALTONIZATION_MATRICES[matrix_name]

    #BGR to RGB
    image_rgb = image[:, :, ::-1]

    transformed_image = apply_transform(image_rgb, applied_matrix)

    plt.imsave(os.path.join(outputDir, f"{matrix_name}_image.png"), transformed_image)