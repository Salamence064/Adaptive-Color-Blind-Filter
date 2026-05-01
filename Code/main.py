import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import argparse
import random


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

GRADIENT_IMAGE_RGB = cv2.imread("../Images/gradient.jpg")
GRADIENT_IMAGE_RGB = GRADIENT_IMAGE_RGB[:, :, ::-1] # BGR to RGB


def weigh_matrices(matrices: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    names = [n for n in weights if n in matrices]

    w = np.array([weights[n] for n in names], dtype=np.float64)

    if w.sum() == 0.0:
        return np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])

    mats = np.stack([matrices[n] for n in names], axis=0)   # (K, 3, 3)
    weighted_m = np.tensordot(w, mats, axes=([0], [0]))        # (3, 3)
    return weighted_m

def apply_transform(image, matrix):
    # apply per-pixel transform: result[x,y,:] = kernel @ image[x,y,:]
    out = np.tensordot(image, matrix.T, axes=([2], [0]))  #HxWx3
    out = np.clip(out, 0, 255).astype(np.uint8)

    return out


def clamp(n, a, b) -> float:
    return max(a, min(n, b))

def normalize(w: tuple) -> tuple:
    true_vals = list(w)

    for i in range(len(true_vals)):
        true_vals[i] = clamp(true_vals[i], 0.0, 1.0)

    arr = []

    for val in true_vals: arr.append(val / sum(true_vals))
    res: tuple = tuple(arr)

    return res

def average(n1, n2):
    return (n1 + n2) / 2.0


# Generate two new filters for the user
# Takes in a tuple with the original and selected filters ((old_r, old_g, old_b), (sel_r, sel_g, sel_b))
# If old = new, pass in the other two filters as parameters, too
# Outputs a tuple of tuples with the weights for the two new filters ((r, g, b), (r, g, b)) 
def apply_new_filters(weights: tuple) -> tuple:
    old = weights[0]
    new = weights[1]

    f1 = weights[2]
    f2 = weights[3]
    f3 = weights[4]

    ran_r = 0.0
    ran_g = 0.0
    ran_b = 0.0

    # add variance if it is the same one otherwise we can get stuck before it reaches a natural point for this to happen
    if old[0] == new[0] and old[1] == new[1] and old[2] == new[2]:
        ran_r = random.uniform(-0.1, 0.1)
        ran_g = random.uniform(-0.1, 0.1)
        ran_b = random.uniform(-0.1, 0.1)

    f1_avg = (average(f1[0], new[0]) + ran_r, average(f1[1], new[1]) + ran_g, average(f1[2], new[2]) + ran_b)
    f2_avg = (average(f2[0], new[0]) + ran_r, average(f2[1], new[1]) + ran_g, average(f2[2], new[2]) + ran_b)
    f3_avg = (average(f3[0], new[0]) + ran_r, average(f3[1], new[1]) + ran_g, average(f3[2], new[2]) + ran_b)

    return (normalize(f1_avg), normalize(f2_avg), normalize(f3_avg))


# Display three images each with a different filter applied for the user to choose between
# This takes the weights in for the two filters applied and for the original ((r, g, b), (r, g, b), (r, g, b))
# Returns the figure ID
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

    w4 = {
        "protanopia": weights[3][0],
        "deuteranopia": weights[3][1],
        "tritanopia": weights[3][2],
    }


    mat1 = weigh_matrices(DALTONIZATION_MATRICES, w1)
    mat2 = weigh_matrices(DALTONIZATION_MATRICES, w2)
    mat3 = weigh_matrices(DALTONIZATION_MATRICES, w3)
    mat4 = weigh_matrices(DALTONIZATION_MATRICES, w4)

    im1 = apply_transform(GRADIENT_IMAGE_RGB, mat1)
    im2 = apply_transform(GRADIENT_IMAGE_RGB, mat2)
    im3 = apply_transform(GRADIENT_IMAGE_RGB, mat3)
    im4 = apply_transform(GRADIENT_IMAGE_RGB, mat4)

    ims = [im1, im2, im3, im4]

   
    plt.close('all')

    fig, axes = plt.subplots(1, 4, figsize=(12, 4))

    for i, (ax, img) in enumerate(zip(axes, ims), start=1):
        ax.imshow(img)

        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values(): spine.set_visible(False)

        ax.set_xlabel(f"Image {i}", fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.4)
    
    fig.canvas.draw_idle()
    plt.pause(0.05)


# Loops until the user is satisfied with the filter
# Outputs a tuple
def find_optimal_filter() -> tuple:
    plt.ion()

    curr_filter: tuple = (0.0, 0.0, 0.0)
    filter1: tuple = (1.0, 0.0, 0.0)
    filter2: tuple = (0.0, 1.0, 0.0)
    filter3: tuple = (0.0, 0.0, 1.0)

    while True:
        display_filters((curr_filter, filter1, filter2, filter3))

        selected_img = input("Which image do you like most (1-4)? ")
        if selected_img == 'quit' or selected_img == 'q':
            print("Quitting program...")
            break

        elif selected_img == 'restart' or selected_img == 'r':
            print("Restarting...")
            curr_filter = (0.0, 0.0, 0.0)
            filter1 = (1.0, 0.0, 0.0)
            filter2 = (0.0, 1.0, 0.0)
            filter3 = (0.0, 0.0, 1.0)
            continue

        while selected_img != '1' and selected_img != '2' and selected_img != '3' and selected_img != '4':
            selected_img = input("Which image do you like most? Enter a number from 1-4: ")
            if selected_img == 'quit' or selected_img == 'q':
                print("Quitting program...")
                return curr_filter

            elif selected_img == 'restart' or selected_img == 'r':
                print("Restarting...")
                curr_filter = (0.0, 0.0, 0.0)
                filter1 = (1.0, 0.0, 0.0)
                filter2 = (0.0, 1.0, 0.0)
                filter3 = (0.0, 0.0, 1.0)
                break
        
        og_filter = curr_filter

        if selected_img == '2': curr_filter = filter1
        elif selected_img == '3': curr_filter = filter2
        elif selected_img == '4': curr_filter = filter3

        if selected_img != 'restart' and selected_img != 'r':       
            filter1, filter2, filter3 = apply_new_filters((og_filter, curr_filter, filter1, filter2, filter3))

    return curr_filter


if __name__ == "__main__":
    outputDir = '../Output'
    imagesDir = "../Images/"
    os.makedirs(outputDir, exist_ok=True)

    #use format 'python3 main.py {pathname}' to run code
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help="Path to the input image")
    args = parser.parse_args()

    if not args.i:
        final_filter = find_optimal_filter()

        with open(f"{outputDir}/filter.txt", 'w') as out:
            out.write(str(final_filter))
            print("Saved filter to file.")
            print("Rerun with a filepath to apply it!")
    
    else:
        if not os.path.exists(outputDir + "/filter.txt"):
            print("Run without a filepath first to get a filter value!")
            exit(0)

        image = cv2.imread(imagesDir + args.i)

        with open(outputDir + '/filter.txt', 'r') as f:
            weighty_weights = f.read().strip()[1:-2].split(', ')
            for i in range(len(weighty_weights)): weighty_weights[i] = float(weighty_weights[i])

        weights = {
            "protanopia": weighty_weights[0],
            "deuteranopia": weighty_weights[1],
            "tritanopia": weighty_weights[2],
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
        input_name = os.path.splitext(os.path.basename(args.i))[0]
        plt.imsave(os.path.join(outputDir, f"{weight_tag}_{input_name}.png"), transformed_image)
