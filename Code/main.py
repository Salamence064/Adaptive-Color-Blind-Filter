import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to the input image")
    args = parser.parse_args()

    image = plt.imread(args.image_path)

    outputDir = '../Outputs/'

    plt.imsave(os.path.join(outputDir, "image.png"), image)