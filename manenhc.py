import numpy as np  
import matplotlib.pyplot as plt
import cv2 as cv
import manimg as img

# =========================================
# NORMALIZE
# =========================================

def normalize(img):

    img = np.abs(img)

    img = (img / img.max()) * 255

    return img.astype(np.uint8)

# =========================================
# CONVOLUTION
# =========================================

def convolution(img, kernel):

    size = kernel.shape[0]

    pad = size // 2

    padded = np.pad(img, pad, mode='constant')

    h, w = img.shape

    hasil = np.zeros_like(img).astype(np.float32)

    for i in range(h):
        for j in range(w):

            region = padded[i:i+size, j:j+size]

            hasil[i, j] = np.sum(region * kernel)

    return hasil

# cara panggil / how to call
# hasil = filter(foto_asli, 3, 'mean')
def filter(img, size, mode):
    # dimensi gambar
    height, width = img.shape

    # ukuran padding
    pad = size // 2

    # tambah padding tepi / add edge padding
    padded = np.pad(img, pad, mode='edge')

    # canvas hasil / input canvas
    canvas = np.zeros_like(img, dtype=np.uint8)

    match mode:

        # filter rata-rata / mean filter
        case 'mean':
            area = size * size

            for i in range(height):
                for j in range(width):

                    # area kernel / kernel region
                    region = padded[i:i+size, j:j+size]

                    # rumus nilai mean / manual mean formula
                    canvas[i, j] = np.sum(region) / area

        case 'median':

            # filter median / median filter
            for i in range(height):
                for j in range(width):

                    # area kernel / kernel region
                    region = padded[i:i+size, j:j+size]

                    # rumus median manual / manual median formula
                    canvas[i, j] = np.median(region)

        case 'mode':

            # filter modus / mode filter
            for i in range(height):
                for j in range(width):

                    # area kernel / kernel region
                    region = padded[i:i+size, j:j+size]

                    # flatten array / flatten region
                    values = region.ravel()

                    # hitung kemunculan / count occurrences
                    count = {}

                    for val in values:
                        if val in count:
                            count[val] += 1
                        else:
                            count[val] = 1

                    # cari nilai terbanyak / find mode value
                    max_count = 0
                    mode_val = 0

                    for val, freq in count.items():
                        if freq > max_count:
                            max_count = freq
                            mode_val = val

                    # simpan hasil / save result
                    canvas[i, j] = mode_val

    # kembalikan gambar / return image
    return canvas

# =========================================
# SMOOTHING
# =========================================

def smoothing(img):

    kernel = np.array([
        [1/10, 1/10, 1/10],
        [1/10, 1/5,  1/10],
        [1/10, 1/10, 1/10]
    ])

    hasil = convolution(img, kernel)

    return np.clip(hasil, 0, 255).astype(np.uint8)

def sharpening(img):

    kernel = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
    ])

    hasil = convolution(img, kernel)

    return np.clip(hasil, 0, 255).astype(np.uint8)

def sobel(img):

    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    sobel_y = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ])

    gx = convolution(img, sobel_x)

    gy = convolution(img, sobel_y)

    hasil = np.abs(gx) + np.abs(gy)

    return normalize(hasil)

def prewitt(img):

    prewitt_x = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ])

    prewitt_y = np.array([
        [-1, -1, -1],
        [ 0,  0,  0],
        [ 1,  1,  1]
    ])

    gx = convolution(img, prewitt_x)

    gy = convolution(img, prewitt_y)

    hasil = np.abs(gx) + np.abs(gy)

    return normalize(hasil)

# =========================================
# ROBERTS
# =========================================

def roberts(img):

    roberts_x = np.array([
        [1, 0],
        [0,-1]
    ])

    roberts_y = np.array([
        [0, 1],
        [-1,0]
    ])

    gx = convolution(img, roberts_x)

    gy = convolution(img, roberts_y)

    hasil = np.abs(gx) + np.abs(gy)

    return normalize(hasil)
