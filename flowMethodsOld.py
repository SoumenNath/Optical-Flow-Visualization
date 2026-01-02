import numpy as np
import cv2
from scipy.ndimage import gaussian_filter, convolve

# --------------------------------------------------------
# Utilities
# --------------------------------------------------------
def to_gray_float(img):
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.float32)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def compute_gradients(I1, I2, sigma=1.0):
    I1s = gaussian_filter(I1, sigma=sigma)
    I2s = gaussian_filter(I2, sigma=sigma)
    kx = np.array([[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]]) / 6.0
    ky = kx.T
    kt = np.array([[1,1,1],[1,1,1],[1,1,1]]) / 9.0
    Ix = convolve((I1s + I2s) * 0.5, kx)
    Iy = convolve((I1s + I2s) * 0.5, ky)
    It = convolve(I2s, kt) - convolve(I1s, kt)
    return Ix, Iy, It

# --------------------------------------------------------
# Lucas–Kanade (dense)
# --------------------------------------------------------
def lucas_kanade_dense(I1, I2, win_size=5, sigma=1.0, eps=1e-6):
    I1 = to_gray_float(I1)
    I2 = to_gray_float(I2)
    Ix, Iy, It = compute_gradients(I1, I2, sigma=sigma)
    half = win_size // 2
    kernel = np.ones((win_size, win_size))
    Ix2, Iy2, Ixy = Ix*Ix, Iy*Iy, Ix*Iy
    Ixt, Iyt = Ix*It, Iy*It
    sum_Ix2 = convolve(Ix2, kernel)
    sum_Iy2 = convolve(Iy2, kernel)
    sum_Ixy = convolve(Ixy, kernel)
    sum_Ixt = convolve(Ixt, kernel)
    sum_Iyt = convolve(Iyt, kernel)
    det = (sum_Ix2 * sum_Iy2) - (sum_Ixy**2)
    det_safe = det + eps
    u = (-sum_Iy2 * sum_Ixt + sum_Ixy * sum_Iyt) / det_safe
    v = (sum_Ixy * sum_Ixt - sum_Ix2 * sum_Iyt) / det_safe
    u[:half,:] = v[:half,:] = 0
    u[-half:,:] = v[-half:,:] = 0
    u[:,:half] = v[:,:half] = 0
    u[:,-half:] = v[:,-half:] = 0
    return u, v

# --------------------------------------------------------
# Horn–Schunck
# --------------------------------------------------------
def horn_schunck(I1, I2, alpha=1.0, n_iter=200, sigma=1.0):
    I1 = to_gray_float(I1)
    I2 = to_gray_float(I2)
    Ix, Iy, It = compute_gradients(I1, I2, sigma=sigma)
    u = np.zeros_like(I1)
    v = np.zeros_like(I1)
    kernel = np.array([[0, 1/4, 0],
                       [1/4, 0, 1/4],
                       [0, 1/4, 0]], dtype=np.float32)
    for _ in range(n_iter):
        u_avg = convolve(u, kernel, mode='reflect')
        v_avg = convolve(v, kernel, mode='reflect')
        denom = alpha**2 + Ix**2 + Iy**2
        term = (Ix*u_avg + Iy*v_avg + It)
        u = u_avg - (Ix*term) / denom
        v = v_avg - (Iy*term) / denom
    return u, v

# --------------------------------------------------------
# Farnebäck (simplified multi-scale polynomial expansion)
# --------------------------------------------------------
def farneback_flow(I1, I2, pyr_scale=0.5, levels=3, winsize=15, iterations=2, poly_n=5, poly_sigma=1.2, use_cv=False):
    """
    Simplified wrapper.
    If use_cv=True, uses OpenCV's calcOpticalFlowFarneback (faster, optimized).
    Otherwise, performs a rough polynomial-expansion style iterative refinement.
    """
    I1 = to_gray_float(I1)
    I2 = to_gray_float(I2)

    if use_cv:
        flow = cv2.calcOpticalFlowFarneback(
            I1, I2, None,
            pyr_scale=pyr_scale,
            levels=levels,
            winsize=winsize,
            iterations=iterations,
            poly_n=poly_n,
            poly_sigma=poly_sigma,
            flags=0
        )
        u, v = flow[...,0], flow[...,1]
        return u, v

    # DIY version (simplified)
    u = np.zeros_like(I1)
    v = np.zeros_like(I1)
    for level in reversed(range(levels)):
        scale = pyr_scale ** level
        I1L = cv2.resize(I1, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        I2L = cv2.resize(I2, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        u = cv2.resize(u, I1L.shape[::-1]) * (1.0 / pyr_scale)
        v = cv2.resize(v, I1L.shape[::-1]) * (1.0 / pyr_scale)

        for _ in range(iterations):
            # warp second image
            H, W = I1L.shape
            grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
            map_x = (grid_x + u).astype(np.float32)
            map_y = (grid_y + v).astype(np.float32)
            I2_warp = cv2.remap(I2L, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

            # local polynomial expansion via Gaussian-blurred derivatives
            Ix = cv2.Sobel(I2_warp, cv2.CV_32F, 1, 0, ksize=poly_n)
            Iy = cv2.Sobel(I2_warp, cv2.CV_32F, 0, 1, ksize=poly_n)
            It = I2_warp - I1L

            # weighted least squares within window
            win = cv2.getGaussianKernel(winsize, poly_sigma)
            win2d = win @ win.T
            Ix2 = convolve(Ix*Ix, win2d)
            Iy2 = convolve(Iy*Iy, win2d)
            Ixy = convolve(Ix*Iy, win2d)
            Ixt = convolve(Ix*It, win2d)
            Iyt = convolve(Iy*It, win2d)

            det = Ix2*Iy2 - Ixy**2 + 1e-9
            du = (-Iy2*Ixt + Ixy*Iyt) / det
            dv = (Ixy*Ixt - Ix2*Iyt) / det

            u += du
            v += dv

    return u, v


# --------------------------------------------------------
# Visualization
# --------------------------------------------------------
def flow_to_color(u, v, max_flow=None):
    H, W = u.shape
    hsv = np.zeros((H, W, 3), dtype=np.float32)
    ang = np.arctan2(v, u)
    mag = np.sqrt(u**2 + v**2)
    if max_flow is None:
        max_flow = np.percentile(mag, 99)
    hsv[...,0] = (ang + np.pi) / (2*np.pi)
    hsv[...,1] = 1.0
    hsv[...,2] = np.clip(mag / (max_flow + 1e-6), 0, 1.0)
    hsv8 = (hsv * np.array([180,255,255])).astype(np.uint8)
    return cv2.cvtColor(hsv8, cv2.COLOR_HSV2BGR)

img1 = cv2.imread("./data/frame_6300.jpg")
img2 = cv2.imread("./data/frame_6350.jpg")

# Lucas-Kanade
u_lk, v_lk = lucas_kanade_dense(img1, img2)

# Horn-Schunck
u_hs, v_hs = horn_schunck(img1, img2, alpha=1.0, n_iter=200)

# Farnebäck (OpenCV optimized)
u_fb, v_fb = farneback_flow(img1, img2, use_cv=True)

# Farnebäck (simplified custom)
u_fb_s, v_fb_s = farneback_flow(img1, img2, use_cv=False)

# Visualize
cv2.imwrite("flow_flucas_kanade_dense.png", flow_to_color(u_lk, v_lk))
cv2.imwrite("flow_horn-schunck.png", flow_to_color(u_hs, v_hs))
cv2.imwrite("flow_farneback_cv.png", flow_to_color(u_fb, v_fb))
cv2.imwrite("flow_farneback_simple.png", flow_to_color(u_fb_s, v_fb_s))