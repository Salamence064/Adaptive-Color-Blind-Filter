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
