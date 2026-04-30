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

GRADIENT_IMAGE_RGB = cv2.imread("Images/gradient.jpg")
GRADIENT_IMAGE_RGB = GRADIENT_IMAGE_RGB[:, :, ::-1] # BGR to RGB


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


# Generate two new filters for the user based on the weights of the selected one
# Outputs a tuple of tuples with the weights for the two new filters ((r, g, b), (r, g, b)) 
def apply_new_filters(r: float, g: float, b: float) -> tuple:
    # Play around with varying degrees of the main weakness + the other two
    # We could make use of binary search in some way here
    return ()


# Display three images each with a different filter applied for the user to choose between
# This takes the weights in for the two filters applied and for the original ((r, g, b), (r, g, b), (r, g, b))
def display_filters(weights: tuple) -> None:
    # format the weights appropriately
    w1 = {
        "protanopia": weights[0][0],
        "deuteranopia": weights[0][1],
        "tritanopia": weights[0][2],
    }

    w2 = {
        "protanopia": weights[1][0],
        "deuteranopia": weights[1][1],
        "tritanopia": weights[1][2],
    }

    w3 = {
        "protanopia": weights[2][0],
        "deuteranopia": weights[2][1],
        "tritanopia": weights[2][2],
    }


    mat1 = weigh_matrices(DALTONIZATION_MATRICES, w1)
    mat2 = weigh_matrices(DALTONIZATION_MATRICES, w2)
    mat3 = weigh_matrices(DALTONIZATION_MATRICES, w3)

    im1 = apply_transform(GRADIENT_IMAGE_RGB, mat1)
    im2 = apply_transform(GRADIENT_IMAGE_RGB, mat2)
    im3 = apply_transform(GRADIENT_IMAGE_RGB, mat3)

    ims = [im1, im2, im3]

   
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    for i, (ax, img) in enumerate(zip(axes, ims), start=1):
        ax.imshow(img)

        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values(): spine.set_visible(False)

        ax.set_xlabel(f"Image {i}", fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.4)
    plt.show()


# Loops until the user is satisfied with the filter
# Outputs a tuple
def find_optimal_filter() -> tuple:
    # Start out with no filter applied and a fully green weak filter applied
    # display_filters(NONE, GREEN-WEAK)
    # If NONE: display_filters(RED-WEAK, BLUE-WEAK)
    # while loop to keep it going
    # apply_new_filters(STORED-WEIGHTS)
    # display_filters(NEW_WEIGHTS)
    # If we reach the quit condition (maybe user says previous one was better 3 times in a row or something like that):
    #     Break out of the loop and print the final weights and output the final image of the best filter
    return ()


if __name__ == "__main__":
    display_filters(((0.6, 0.3, 0.1), (0.2, 0.7, 0.1), (0.3, 0.3, 0.4)))

    # outputDir = '../Outputs/'
    # os.makedirs(outputDir, exist_ok=True)

    # #use format 'python3 main.py {pathname}' to run code
    # parser = argparse.ArgumentParser()
    # parser.add_argument("image_path", help="Path to the input image")
    # args = parser.parse_args()

    # image = cv2.imread(args.image_path)

    # #select matrix to apply
    # # matrix_name = "protanopia"
    # # matrix_name = "deuteranopia"
    # matrix_name = "tritanopia"

    # weights = {
    #     "protanopia": 0.6,
    #     "deuteranopia": 0.3,
    #     "tritanopia": 0.1,
    # }

    # applied_matrix = weigh_matrices(DALTONIZATION_MATRICES, weights)

    # #BGR to RGB
    # image_rgb = image[:, :, ::-1]

    # transformed_image = apply_transform(image_rgb, applied_matrix)

    # #save image
    # p_weight = weights.get("protanopia", 0.0)
    # d_weight = weights.get("deuteranopia", 0.0)
    # t_weight = weights.get("tritanopia", 0.0)
    # weight_tag = f"{p_weight}_{d_weight}_{t_weight}"
    # input_name = os.path.splitext(os.path.basename(args.image_path))[0]
    # plt.imsave(os.path.join(outputDir, f"{weight_tag}_{input_name}.png"), transformed_image)
