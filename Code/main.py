import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import argparse



def apply_transform(image, matrix):
    # apply per-pixel transform: result[x,y,:] = kernel @ image[x,y,:]
    out = np.tensordot(image, matrix.T, axes=([2], [0]))  #HxWx3
    out = np.clip(out, 0, 255)

    return out

if __name__ == "__main__":
    outputDir = '../Outputs/'

    #use format 'python3 main.py {pathname}' to run code
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to the input image")
    args = parser.parse_args()

    image = cv2.imread(args.image_path)

    #BGR to RGB
    image_rgb = image[:, :, ::-1]

    plt.imsave(os.path.join(outputDir, "image.png"), image_rgb)