import cv2
import numpy as np
from flowMethods import (
    lucas_kanade_dense,
    horn_schunck,
    farneback,
    lucas_kanade_sparse,
    flow_to_color
)

# -------------------------------
# Utility: load or generate images
# -------------------------------
def load_or_generate_images(path1=None, path2=None):
    if path1 and path2:
        img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
        if img1 is not None and img2 is not None:
            return img1, img2

    # If files not found, create simple synthetic motion frames
    print("⚠️ Using synthetic test images...")
    img1 = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(img1, (60, 80), (110, 130), 255, -1)

    img2 = np.zeros_like(img1)
    cv2.rectangle(img2, (70, 80), (120, 130), 255, -1)  # Move 10px to right

    return img1, img2


# -------------------------------
# Main comparison script
# -------------------------------
def run_all_methods():
    # Load frames
    img1, img2 = load_or_generate_images("frame000.png", "frame001.png")

    # Resize for uniform visualization
    img1 = cv2.resize(img1, (320, 240))
    img2 = cv2.resize(img2, (320, 240))

    # ----- Lucas–Kanade (Dense) -----
    print("Running Lucas–Kanade (dense)...")
    flow_lk = lucas_kanade_dense(img1, img2)
    color_lk = flow_to_color(flow_lk)

    # ----- Horn–Schunck -----
    print("Running Horn–Schunck...")
    flow_hs = horn_schunck(img1, img2, alpha=1.0, num_iter=100)
    color_hs = flow_to_color(flow_hs)

    # ----- Farnebäck -----
    print("Running Farnebäck...")
    flow_fb = farneback(img1, img2)
    color_fb = flow_to_color(flow_fb)

    # ----- Sparse Lucas–Kanade -----
    print("Running Sparse Lucas–Kanade (feature tracking)...")
    sparse_vis = lucas_kanade_sparse(img1, img2)

    # Convert grayscale base to BGR for consistency
    img2_bgr = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    # Create composite view (2x2 grid)
    top_row = np.hstack((color_lk, color_hs))
    bottom_row = np.hstack((color_fb, sparse_vis))
    combined = np.vstack((top_row, bottom_row))

    cv2.imshow("Optical Flow Comparison", combined)
    print("Press 'q' to quit.")
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_all_methods()
