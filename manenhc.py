import numpy as np  
import matplotlib.pyplot as plt
import cv2 as cv

# =========================================
# MANUAL ABS
# =========================================

def manual_abs(arr):

    h, w = arr.shape
    hasil = np.zeros((h, w), dtype=np.float32)

    for i in range(h):
        for j in range(w):

            if arr[i, j] < 0:
                hasil[i, j] = -arr[i, j]
            else:
                hasil[i, j] = arr[i, j]

    return hasil

# =========================================
# MANUAL MAX
# =========================================

def manual_max(arr):

    h, w = arr.shape
    maksimum = arr[0, 0]

    for i in range(h):
        for j in range(w):

            if arr[i, j] > maksimum:
                maksimum = arr[i, j]

    return maksimum

# =========================================
# MANUAL SUM
# =========================================

def manual_sum(arr):

    total = 0

    if len(arr.shape) == 1:

        for i in arr:
            total += i

    else:

        h, w = arr.shape

        for i in range(h):
            for j in range(w):
                total += arr[i, j]

    return total

# =========================================
# MANUAL ZEROS LIKE
# =========================================

def manual_zeros_like(img):

    h, w = img.shape

    hasil = []

    for i in range(h):

        baris = []

        for j in range(w):
            baris.append(0)

        hasil.append(baris)

    return np.array(hasil)

# =========================================
# MANUAL CLIP
# =========================================

def manual_clip(arr, min_val, max_val):

    h, w = arr.shape

    hasil = np.zeros((h, w), dtype=np.float32)

    for i in range(h):
        for j in range(w):

            nilai = arr[i, j]

            if nilai < min_val:
                hasil[i, j] = min_val

            elif nilai > max_val:
                hasil[i, j] = max_val

            else:
                hasil[i, j] = nilai

    return hasil

# =========================================
# MANUAL MEDIAN
# =========================================

def manual_median(arr):

    data = []

    h, w = arr.shape

    for i in range(h):
        for j in range(w):
            data.append(arr[i, j])

    data.sort()

    n = len(data)

    if n % 2 == 1:
        return data[n // 2]

    else:
        tengah1 = data[(n // 2) - 1]
        tengah2 = data[n // 2]

        return (tengah1 + tengah2) / 2

# =========================================
# MANUAL FLATTEN / RAVEL
# =========================================

def manual_ravel(arr):

    hasil = []

    h, w = arr.shape

    for i in range(h):
        for j in range(w):
            hasil.append(arr[i, j])

    return hasil

# =========================================
# MANUAL PAD
# =========================================

def manual_pad(img, pad):

    h, w = img.shape

    new_h = h + (2 * pad)
    new_w = w + (2 * pad)

    hasil = np.zeros((new_h, new_w), dtype=img.dtype)

    for i in range(h):
        for j in range(w):

            hasil[i + pad][j + pad] = img[i][j]

    return hasil

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
# EDGE DETECTION (kalau mau pakai kernel khusus)
# =========================================

def edge(img, kernelx, kernely):

    # konvolusi sumbu x / x-axis convolution
    gx = convolution(img, kernelx)

    # konvolusi sumbu y / y-axis convolution
    gy = convolution(img, kernely)

    # kanvas kosong / empty canvas
    canvas = np.zeros_like(img, dtype=np.float32)

    # gabung gradien absolut / combine absolute gradients
    canvas = np.abs(gx) + np.abs(gy)

    # normalisasi ke 0-255 / normalize to 0-255
    canvas = canvas * 255.0 / np.max(canvas)

    # batas nilai dan konversi / clip values and convert
    return np.clip(canvas, 0, 255).astype(np.uint8)

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
