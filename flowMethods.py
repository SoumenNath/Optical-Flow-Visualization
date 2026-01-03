import cv2
import numpy as np
import matplotlib.pyplot as plt
#Some resources
#LK Dense
#https://sandipanweb.wordpress.com/2018/02/25/implementing-lucas-kanade-optical-flow-algorithm-in-python/

#HS
#https://github.com/lmiz100/Optical-flow-Horn-Schunck-method/blob/master/MyHornSchunck.py
#https://datahacker.rs/013-optical-flow-using-horn-and-schunck-method/
#https://github.com/lmiz100/Optical-flow-Horn-Schunck-method/

#F
#https://www.geeksforgeeks.org/python/opencv-the-gunnar-farneback-optical-flow/
#https://medium.com/%40igorirailean/dense-optical-flow-with-python-using-opencv-cb6d9b6abcaf
#https://github.com/ericPrince/optical-flow

#LK Sparse
#https://github.com/Utkal97/Object-Tracking/blob/main/LucasKanadeOptFlow.py

def compare_flow_quiver(u_gt, v_gt, u_est, v_est, step=10):
    h, w = u_gt.shape
    Y, X = np.mgrid[0:h:step, 0:w:step]

    plt.figure(figsize=(12,5))

    # Ground Truth
    plt.subplot(1,2,1)
    plt.quiver(X, Y,
               u_gt[0:h:step, 0:w:step],
               v_gt[0:h:step, 0:w:step],
               angles='xy', scale_units='xy', scale=1)
    plt.gca().invert_yaxis()
    plt.title("Ground Truth Flow")

    # Estimated
    plt.subplot(1,2,2)
    plt.quiver(X, Y,
               u_est[0:h:step, 0:w:step],
               v_est[0:h:step, 0:w:step],
               angles='xy', scale_units='xy', scale=1)
    plt.gca().invert_yaxis()
    plt.title("Estimated Optical Flow")

    plt.tight_layout()
    plt.show()

def flow_error_map(u_gt, v_gt, u_est, v_est):
    error = np.sqrt((u_gt - u_est)**2 + (v_gt - v_est)**2)

    plt.figure(figsize=(6,5))
    plt.imshow(error, cmap='hot')
    plt.colorbar(label="Endpoint Error (pixels)")
    plt.title("Optical Flow Error Map")
    plt.axis("off")
    plt.show()

def endpoint_error(u_gt, v_gt, u_est, v_est):
    epe = np.sqrt((u_gt - u_est)**2 + (v_gt - v_est)**2)
    return np.mean(epe)

def porcupine_plot(u, v, step=10, auto_scale=True, scale=1.0):
    h, w = u.shape
    Y, X = np.mgrid[0:h:step, 0:w:step]

    U = u[0:h:step, 0:w:step]
    V = v[0:h:step, 0:w:step]

    # If the flow is tiny, boost it
    if auto_scale:
        max_mag = np.max(np.sqrt(U**2 + V**2))
        if max_mag < 0.5:   # small flow threshold
            scale = 20.0 / (max_mag + 1e-6)  # boost small flow
            print(f"[INFO] Auto-scaling porcupine arrows by {scale:.2f}")

    U *= scale
    V *= scale

    plt.figure(figsize=(8, 8))
    plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1)
    plt.gca().invert_yaxis()
    plt.title("Porcupine Optical Flow Plot")
    plt.show()


def visualize_quiver(u, v, sparse_p0=None, sparse_p1=None,
                      img=None, step=10, title="Optical Flow (Quiver)"):
    """
    Universal quiver visualization for both dense and sparse flow.

    Exception to sparse flow Sparse flow:
        visualize_quiver(None, None, p0, p1)

    Parameters
    ----------
    flow_or_points : ndarray or None
        Dense flow array of shape (H,W,2) OR u array.
    flow_v : ndarray or None
        Dense v array if u passed separately.
    sparse_p0, sparse_p1 : arrays (N,2)
        Sparse original and transformed points.
    step : int
        Arrow stride for dense flow.
    """
    plt.figure(figsize=(12, 8))
    H, W = u.shape

    # Downsample for readability
    y, x = np.mgrid[0:H:step, 0:W:step]
    u_s = u[::step, ::step]
    v_s = v[::step, ::step]

    # Compute vector magnitude for autoscaling
    mag = np.sqrt(u_s**2 + v_s**2)
    max_mag = np.max(mag) if np.max(mag) > 0 else 1

    # Autoscale so arrows are visible
    scale = 0.2 * max_mag

    if img is not None:
        plt.imshow(img, cmap='gray')

    plt.quiver(
        x, y, u_s, -v_s,
        color='red',
        angles='xy',
        scale_units='xy',
        scale=scale,
        width=0.003
    )

    plt.gca().invert_yaxis()
    plt.xlim(0, W)
    plt.ylim(H, 0)
    plt.title(title)
    plt.show()



#Colour Visualization
def flow_to_color(flow):
    #Convert optical flow to a color image using HSV color encoding
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    value = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) # Value = magnitude
    hue = (angle * 180 / np.pi / 2).astype(np.uint8)
    hue1 = (angle + np.pi) * (180 / (2*np.pi))

    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = hue  #Hue = direction
    hsv[..., 1] = 255 #(full) saturation
    hsv[..., 2] = value  #Brightness 
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

#Binary visualization
def flow_to_binary(flow, threshold=0.1):
    mag = np.sqrt(flow[...,0]**2 + flow[...,1]**2)
    #If motion at pixel > threshold then the resulting mapping is 1
    binary = (mag > threshold).astype(np.uint8)
    return binary


#Convert optical flow into a 0/1 matrix showing flow presence
#1 = significant motion (above threshold)
#0 = no motion
def flow_to_binary_matrix(u, v, threshold=1.0, print_matrix=True):
    #Magnitude of flow
    mag = np.sqrt(u**2 + v**2)

    #Threshold → 0/1 matrix
    binary = (mag > threshold).astype(np.uint8)

    if print_matrix:
        print("\n=== Binary Flow Matrix (1 = Motion) ===")
        h, w = binary.shape

        if h > 30 or w > 30:
            print("(Matrix is large — printing top-left 30×30 preview)")
            print(binary[:30, :30])
        else:
            print(binary)


# Lucas–Kanade Dense Optical Flow
def lucas_kanade_dense(img1, img2, window_size=5):
    #Compute gradients in x, y, and time using Sobel filters and simple differencing
    Ix = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=5)
    Iy = cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=5)
    It = img2.astype(float) - img1.astype(float)

    #Initialize flow to zero everywhere
    u = np.zeros(img1.shape)
    v = np.zeros(img1.shape)
    half_w = window_size // 2

    for y in range(half_w, img1.shape[0] - half_w):
        for x in range(half_w, img1.shape[1] - half_w):
            #Ix_win, Iy_win, and It_win are flattened arrays (N elements each) representing the gradients inside the local window.
            Ix_win = Ix[y-half_w:y+half_w+1, x-half_w:x+half_w+1].flatten()
            Iy_win = Iy[y-half_w:y+half_w+1, x-half_w:x+half_w+1].flatten()
            It_win = It[y-half_w:y+half_w+1, x-half_w:x+half_w+1].flatten()

            #For each small window (like 5×5 pixels):
            #Constructs the system A [ u , v ]^T = b
            #Solve for u and v using least-squares

            #stack arrarys into an N×2 matrix where each row is [Ix, Iy].
            A = np.vstack((Ix_win, Iy_win)).T
            b = -It_win

            ATA = A.T @ A
            ATb = A.T @ b

            #If the determinant of this matrix is 0, that means it’s singular — we can’t invert it (e.g. in flat regions with no texture).
            if np.linalg.det(ATA) != 0:
                uv = np.linalg.inv(ATA) @ ATb
                #Assign the computed flow vector to the pixel’s center
                u[y, x] = uv[0]
                v[y, x] = uv[1]

    #Stack the 2D arrays along a new 3rd axis so that each pixel has a 2D flow vector stored at that position
    return np.dstack((u, v))



# Horn–Schunck Global Optical Flow
def horn_schunck(img1, img2, alpha=1.0, num_iter=100):
    #Compute gradients in x, y, and time using Sobel filters and simple differencing
    Ix = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=3)
    It = img2.astype(float) - img1.astype(float)

    #Initializes flow to zero everywhere.
    u = np.zeros(img1.shape)
    v = np.zeros(img1.shape)

    kernel = np.array([[1/12, 1/6, 1/12],
                       [1/6, 0, 1/6],
                       [1/12, 1/6, 1/12]])
    
    #The loop runs for num_iter iterations to converge, as in the iterative estimates of the flow field (u,v) stop changing
    for _ in range(num_iter):
        #Compute the local average flow for each pixel — this is the smoothness term
        u_avg = cv2.filter2D(u, -1, kernel)
        v_avg = cv2.filter2D(v, -1, kernel)
        #Update each pixel’s flow estimate using Horn–Schunck’s iterative formula
        #alpha controls how much smoothing occurs — higher = smoother flow, lower = more detailed but noisier
        num = (Ix * u_avg + Iy * v_avg + It)
        den = alpha**2 + Ix**2 + Iy**2
        u = u_avg - Ix * num / den
        v = v_avg - Iy * num / den

    #Stack the 2D arrays along a new 3rd axis so that each pixel has a 2D flow vector stored at that position
    return np.dstack((u, v))



# Farnebäck Optical Flow (dense)
def farneback(img1, img2, use_cv=True):
    #Compute gradients
    Ix = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=5)
    Iy = cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=5)
    It = img2.astype(float) - img1.astype(float)

    #Local products of derivatives
    #Captures local image structure similar to Lucas–Kanade, but used here to approximate quadratic patches
    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    Ixy = Ix * Iy
    Ixt = Ix * It
    Iyt = Iy * It

    #Apply Gaussian smoothing to simulate polynomial neighborhood fitting
    #Instead of solving for each pixel independently, we blur (average) derivative products to simulate polynomial fitting over a neighborhood
    #This mimics how the real Farnebäck algorithm uses local polynomial expansions
    Ix2 = cv2.GaussianBlur(Ix2, (9, 9), 1.5)
    Iy2 = cv2.GaussianBlur(Iy2, (9, 9), 1.5)
    Ixy = cv2.GaussianBlur(Ixy, (9, 9), 1.5)
    Ixt = cv2.GaussianBlur(Ixt, (9, 9), 1.5)
    Iyt = cv2.GaussianBlur(Iyt, (9, 9), 1.5)

    # Solve the optical flow equations for each pixel
    det = Ix2 * Iy2 - Ixy * Ixy + 1e-6  # determinant for 2x2 system

    u = (-Iy2 * Ixt + Ixy * Iyt) / det
    v = (Ixy * Ixt - Ix2 * Iyt) / det

    return np.dstack((u, v))


# Sparse Lucas–Kanade 
def lucas_kanade_sparse(img1, img2, feature_params=None, window_size=15):

    # Convert to grayscale
    if img1.ndim == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if img2.ndim == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    #Must be uint8
    img1 = img1.astype(np.uint8)
    img2 = img2.astype(np.uint8)

    #Detect corners (points to track)
    if feature_params is None:
        feature_params = dict(  maxCorners=1500, qualityLevel=0.001, minDistance=2, blockSize=5,)

    p0 = cv2.goodFeaturesToTrack(img1, mask=None, **feature_params)
    if p0 is None:
        print("No features found.")
        return None, None, None, None, None

    p0 = p0.astype(np.float32)

    #Compute image gradients
    Ix = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=3)
    It = img2.astype(float) - img1.astype(float)

    #This computes half of the local window size. Lucas–Kanade, looks at a small square patch around each corner (for example, 15×15 pixels). 
    #If the window size is 15, half_w = 7. This makes it easy to index local regions around each feature.
    half_w = window_size // 2

    #p0 the original feature points (the corners found in the first frame)
    #Shape: (N, 1, 2) — where N is the number of points, and each entry has [x, y].
    #Initialize p1 with the same shape and datatype, but filled with zeros because p1 will store the new positions of each tracked feature in the second frame
    p1 = np.zeros_like(p0)

    #status keeps track of whether the flow for each point was successfully computed
    status = np.zeros((p0.shape[0], 1), dtype=np.uint8)

    # Dense flow outputs
    H, W = img1.shape
    u = np.zeros((H, W), dtype=np.float32)
    v = np.zeros((H, W), dtype=np.float32)

    #For each corner solve local flow
    for i, pt in enumerate(p0):
        x, y = pt.ravel()
        x = int(x)
        y = int(y)

        # Skip if too close to borders
        if (x - half_w < 0 or y - half_w < 0 or
            x + half_w >= W or y + half_w >= H):
            continue

        #Extract local patches
        #This ensures taking a centered window around (x, y)
        Ix_win = Ix[y-half_w:y+half_w+1, x-half_w:x+half_w+1].flatten()
        Iy_win = Iy[y-half_w:y+half_w+1, x-half_w:x+half_w+1].flatten()
        It_win = It[y-half_w:y+half_w+1, x-half_w:x+half_w+1].flatten()

        #Least-squares matrices
        A = np.vstack((Ix_win, Iy_win)).T
        b = -It_win

        #Solve (A^T A) v = A^T b
        ATA = A.T @ A
        if np.linalg.det(ATA) < 1e-6:
            continue

        flow = np.linalg.inv(ATA) @ (A.T @ b)
        dx, dy = flow[0], flow[1]

        # Save sparse flow
        p1[i, 0, 0] = x + dx
        p1[i, 0, 1] = y + dy
        status[i] = 1
        
        # Put sparse flow into dense u,v arrays
        u[y, x] = dx
        v[y, x] = dy
    return p0, p1, np.dstack((u, v))

def synthetic_translation(size=(200, 200), shift=(10, 0)):
    #apply more feature points (stripes or corners)
    #put a little image inside
    """Create two synthetic images with a white square moving."""
    h, w = size
    img1 = np.zeros((h, w), np.uint8)

    # Original square
    cv2.rectangle(img1, (50, 50), (100, 100), 255, -1)

    # Shift
    dx, dy = shift
    M = np.float32([[1, 0, dx],
                    [0, 1, dy]])
    img2 = cv2.warpAffine(img1, M, (w, h))

    return img1, img2

def gt_translation(shape, dx=10, dy=0):
    """
    Create ground-truth flow for a known translation.
    """
    h, w = shape
    u_gt = np.full((h, w), dx, dtype=np.float32)
    v_gt = np.full((h, w), dy, dtype=np.float32)
    return u_gt, v_gt

def synthetic_textured_translation(dx=10, dy=0, size=(200,200)):
    img1 = np.random.randint(0, 255, size, dtype=np.uint8)
    img2 = np.roll(img1, shift=(dy, dx), axis=(0,1))
    return img1, img2

def gt_textured_translation(shape, dx=10, dy=0):
    h, w = shape
    u_gt = np.full((h, w), dx, dtype=np.float32)
    v_gt = np.full((h, w), dy, dtype=np.float32)
    return u_gt, v_gt

# def synthetic_expansion(size=(200, 200), r1=20, r2=30):
#     h, w = size
#     img1 = np.zeros((h, w), np.uint8)
#     img2 = np.zeros((h, w), np.uint8)

#     cv2.circle(img1, (w//2, h//2), r1, 255, -1)
#     cv2.circle(img2, (w//2, h//2), r2, 255, -1)

#     return img1, img2

def synthetic_expansion(size=(200,200), scale=1.05):
    img1 = np.random.randint(0, 255, size, dtype=np.uint8)
    h, w = size

    M = cv2.getRotationMatrix2D((w//2, h//2), 0, scale)
    img2 = cv2.warpAffine(img1, M, (w, h))

    return img1, img2

def gt_expansion(shape, alpha=0.02):
    h, w = shape
    cy, cx = h // 2, w // 2

    Y, X = np.mgrid[0:h, 0:w]
    u_gt = alpha * (X - cx)
    v_gt = alpha * (Y - cy)

    return u_gt.astype(np.float32), v_gt.astype(np.float32)

# def synthetic_rotation(size=(200, 200), angle=10):
#     img1 = np.zeros(size, np.uint8)

#     # Draw vertical line
#     cv2.line(img1, (100, 50), (100, 150), 255, 4)

#     # Rotation matrix
#     M = cv2.getRotationMatrix2D((100, 100), angle, 1.0)
#     img2 = cv2.warpAffine(img1, M, size[::-1])

#     return img1, img2
def synthetic_rotation(size=(200,200), angle_deg=5):
    img1 = np.random.randint(0, 255, size, dtype=np.uint8)
    h, w = size

    M = cv2.getRotationMatrix2D((w//2, h//2), angle_deg, 1.0)
    img2 = cv2.warpAffine(img1, M, (w, h))

    return img1, img2

def gt_rotation(shape, omega=0.02):
    h, w = shape
    cy, cx = h // 2, w // 2

    Y, X = np.mgrid[0:h, 0:w]

    u_gt = -omega * (Y - cy)
    v_gt =  omega * (X - cx)

    return u_gt.astype(np.float32), v_gt.astype(np.float32)

def get_ground_truth(motion_type, img_shape):
    if motion_type == 1:
        return gt_translation(img_shape)
    elif motion_type == 2:
        return gt_textured_translation(img_shape)
    elif motion_type == 3:
        return gt_expansion(img_shape)
    elif motion_type == 4:
        return gt_rotation(img_shape)
            
def choose_synthetic_images():
    while True:
        print("\nSelect a synthetic test case:")
        print("1 - Translation")
        print("2 - Textured Translation")
        print("3 - Expansion")
        print("4 - Rotation")
        print("Q - Quit program")

        choice = input("Enter choice: ").strip().lower()

        if choice == "1":
            return *synthetic_translation(), 1
        elif choice == "2":
            return *synthetic_textured_translation(), 2
        elif choice == "3":
            return *synthetic_expansion(), 3
        elif choice == "4":
            return *synthetic_rotation(), 4
        elif choice == "q":
            return None, None, "q"

        print("Invalid choice. Try again.")

def show_synthetic_images(img1, img2, title1="Image 1", title2="Image 2"):
    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)
    plt.title(title1)
    plt.imshow(img1, cmap="gray")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.title(title2)
    plt.imshow(img2, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def main():
    print("=== Optical Flow Demo ===")
    # img1 = cv2.imread("./data/frame_32.png")
    # img2 = cv2.imread("./data/frame_33.png")
    while True:

        # --- Choose synthetic image pair ---
        img1, img2, choice = choose_synthetic_images()
        if img1 is None:
            print("Exiting program...")
            break
       #u_gt, v_gt = gt_translation(img1.shape, dx=10, dy=0)
        u_gt, v_gt = get_ground_truth(choice, img1.shape)

        # Show both images
        show_synthetic_images(img1, img2)
        while True:
            print("\n=== Select Optical Flow Method ===")
            print("1 - Lucas–Kanade Dense")
            print("2 - Horn–Schunck")
            print("3 - Farneback")
            print("4 - Lucas–Kanade Sparse")
            print("Q - Return to shape selection")

            choice = input("Enter choice: ").strip().lower()
            quiver_title = ""

            if choice == 'q':
                print("Returning to shape selection.")
                break

            if choice == '1':
                print("Running dense Lucas–Kanade...")
                flow = lucas_kanade_dense(img1, img2)
                quiver_title = "Dense LK"

            elif choice == '2':
                print("Running Horn–Schunck...")
                flow = horn_schunck(img1, img2)
                quiver_title = "Horn–Schunck"

            elif choice == '3':
                print("Running Farneback...")
                flow = farneback(img1, img2)
                quiver_title = "Farneback Dense"

            elif choice == '4':
                print("Running sparse Lucas–Kanade...")
                p0, p1, flow = lucas_kanade_sparse(img1, img2)
                quiver_title = "Sparse LK"

            else:
                print("Invalid choice.")
                continue

            u, v = flow[:, :, 0], flow[:, :, 1]
            #visualize_quiver(u, v, title=quiver_title)
            visualize_quiver(u, v, img=img1, title=quiver_title)
            porcupine_plot(u, v, step=10, scale=3.0)
            # Visual comparison
            compare_flow_quiver(u_gt, v_gt, u, v)

            # Error visualization
            flow_error_map(u_gt, v_gt, u, v)
            # Metric
            print("Mean EPE:", endpoint_error(u_gt, v_gt, u, v))
            binary_map = flow_to_binary(flow)
            #scale the 1 value so it becomes 255 (white)
            cv2.imshow("Binary Flow Visualization", binary_map * 255)


#quiveer plot
#don't use the defination
#what is the motivation
#proquipine, synthetic image see if the vector tracsk the 
#what does the literatures say, what is the evidence that it is dangerous
#comparisons between ar vs non ar
#meta analysis
#dimension of papers, plotting a grid
#a tree visualization


if __name__ == "__main__":
    main()
