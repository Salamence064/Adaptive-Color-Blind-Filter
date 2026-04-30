import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import argparse


#define daltonization transform matrices
#matrices based on: https://github.com/joergdietrich/daltonize/blob/main/daltonize/daltonize.py
DALTONIZATION_MATRICES = {
    "protanopia": np.array([ #missing red
        [0, 0.90822864, 0.008192],
        [0, 1, 0],
        [0, 0, 1]
    ]),
    "deuteranopia": np.array([ #missing green
        [1, 0, 0], 
        [1.10104433,  0, -0.00901975], 
        [0, 0, 1]
    ]),
    "tritanopia": np.array([ #missing blue
        [1, 0, 0],
        [0, 1, 0],
        [-0.15773032,  1.19465634, 0]
    ])
}

def weigh_matrices(matrices: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    names = [n for n in weights if n in matrices]

    w = np.array([weights[n] for n in names], dtype=np.float64)
    w = w / w.sum()

    mats = np.stack([matrices[n] for n in names], axis=0)   # (K, 3, 3)
    weighted_m = np.tensordot(w, mats, axes=([0], [0]))        # (3, 3)
    return weighted_m

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
    # matrix_name = "protanopia"
    # matrix_name = "deuteranopia"
    matrix_name = "tritanopia"

    weights = {
        "protanopia": 0.6,
        "deuteranopia": 0.3,
        "tritanopia": 0.1,
    }

    applied_matrix = weigh_matrices(DALTONIZATION_MATRICES, weights)

    #BGR to RGB
    image_rgb = image[:, :, ::-1]

    transformed_image = apply_transform(image_rgb, applied_matrix)

    #save image
    p_weight = weights.get("protanopia", 0.0)
    d_weight = weights.get("deuteranopia", 0.0)
    t_weight = weights.get("tritanopia", 0.0)
    weight_tag = f"{p_weight}_{d_weight}_{t_weight}"
    input_name = os.path.splitext(os.path.basename(args.image_path))[0]
    plt.imsave(os.path.join(outputDir, f"{weight_tag}_{input_name}.png"), transformed_image)
# Generate two new filters for the user based on the weights of the selected one
# Outputs a tuple of tuples with the weights for the two new filters ((r, g, b), (r, g, b)) 
def apply_new_filters(r: float, g: float, b: float) -> tuple:
    # Play around with varying degrees of the main weakness + the other two
    # We could make use of binary search in some way here
    return ()


# Display a two images each with a different filter applied for the user to choose between
# This takes the weights in for the two filters applied
def display_filters(r1: float, g1: float, b1: float, r2: float, g2: float, b2: float) -> None:
    pass


def main() -> None:
    # Start out with no filter applied and a fully green weak filter applied
    # display_filters(NONE, GREEN-WEAK)
    # If NONE: display_filters(RED-WEAK, BLUE-WEAK)
    # while loop to keep it going
    # apply_new_filters(STORED-WEIGHTS)
    # display_filters(NEW_WEIGHTS)
    # If we reach the quit condition (maybe user says previous one was better 3 times in a row or something like that):
    #     Break out of the loop and print the final weights and output the final image of the best filter
    pass
