import cv2
import numpy as np
from flowMethods import (
    lucas_kanade_dense,
    horn_schunck,
    farneback,
    lucas_kanade_sparse,
    flow_to_color,
    flow_to_binary
)


def load_gray(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
def m():
    print("=== First Method ===")

def main():
    print("=== Optical Flow Test ===")
    img1 = cv2.imread("./data/frame_6300.jpg")
    img2 = cv2.imread("./data/frame_6350.jpg")

    print("Select method:\n1 = Lucas–Kanade Dense\n2 = Horn–Schunck\n3 = Farnebäck (Custom)\n4 = Sparse Lucas–Kanade")
    method = input("Method number: ")

    print("Computing optical flow...")
    if method == "1":
        print("Running Lucas–Kanade Dense...")
        u, v = lucas_kanade_dense(img1, img2)
    elif method == "2":
        print("Running Horn–Schunck Iterative Solver...")
        u, v = horn_schunck(img1, img2)
    elif method == "3":
        print("Running Custom Farnebäck Polynomial Expansion...")
        u, v = farneback(img1, img2)
    elif method == "4":
        print("Running Sparse Lucas–Kanade tracker...")
        p0, p1, status = lucas_kanade_sparse(img1, img2)
        print("Sparse flow computed.")
        print("Initial points:\n", p0)
        print("Tracked points:\n", p1)
        print("Status:\n", status)
        return
    else:
        print("Invalid method.")
        return

    print("Generating visualizations...")
    # Flow visualization
    color_vis = flow_to_color(u, v)
    binary_vis = flow_to_binary(u, v)

    cv2.imshow("Flow Color Visualization", color_vis)
    cv2.imshow("Binary Motion Map (0/1)", binary_vis * 255)

    print("Press any key to exit.")
    cv2.waitKey(0)


if __name__ == "__main__":
    m()
    main()
