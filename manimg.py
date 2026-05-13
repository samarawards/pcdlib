import numpy as np  
import matplotlib.pyplot as plt
import cv2 as cv

def threshold(image, t=128):
    h, w = image.shape

    result = np.zeros((h, w), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            if image[i, j] >= t:
                result[i, j] = 255
            else:
                result[i, j] = 0

    return result

def flipping(image, mode='X'):
    h, w = image.shape[:2]
    hasil = np.zeros_like(image)

    for i in range(h):
        for j in range(w):
            if mode == 'X':  # horizontal
                hasil[i, j] = image[i, w - 1 - j]
            elif mode == 'Y':  # vertical
                hasil[i, j] = image[h - 1 - i, j]

    return hasil

def merge_image(citra1, citra2, orientation='H'): 
    if orientation == 'H':   
        tinggi = max(citra1.shape[0], citra2.shape[0]) 
        lebar_total = citra1.shape[1] + citra2.shape[1] 
        if len(citra1.shape) == 2: 
            gabungan = np.zeros((tinggi, lebar_total), dtype=citra1.dtype) 
        else: 
            gabungan = np.zeros((tinggi, lebar_total, citra1.shape[2]), dtype=citra1.dtype) 

        gabungan[0:citra1.shape[0], 0:citra1.shape[1]] = citra1 
        gabungan[0:citra2.shape[0], citra1.shape[1]:citra1.shape[1]+citra2.shape[1]] = citra2 

    else:   
        tinggi_total = citra1.shape[0] + citra2.shape[0] 
        lebar = max(citra1.shape[1], citra2.shape[1]) \
        
        if len(citra1.shape) == 2: 
            gabungan = np.zeros((tinggi_total, lebar), dtype=citra1.dtype) 
        else: 
            gabungan = np.zeros((tinggi_total, lebar, citra1.shape[2]), dtype=citra1.dtype) 

        gabungan[0:citra1.shape[0], 0:citra1.shape[1]] = citra1 
        
        gabungan[citra1.shape[0]:citra1.shape[0]+citra2.shape[0], 0:citra2.shape[1]] = citra2 
            
    return gabungan

def rotate(image, angle):
    theta = np.radians(angle)

    h, w = image.shape
    cx, cy = w // 2, h // 2

    cos_t = abs(np.cos(theta))
    sin_t = abs(np.sin(theta))

    # Hitung ukuran baru
    new_w = int(w * cos_t + h * sin_t)
    new_h = int(h * cos_t + w * sin_t)

    # Pusat baru
    new_cx, new_cy = new_w // 2, new_h // 2

    # Canvas baru
    rotated = np.zeros((new_h, new_w), dtype=image.dtype)

    for y in range(new_h):
        for x in range(new_w):
            # Translasi ke pusat baru
            x_shift = x - new_cx
            y_shift = y - new_cy

            # Inverse rotation
            x_old = int(x_shift * np.cos(theta) + y_shift * np.sin(theta))
            y_old = int(-x_shift * np.sin(theta) + y_shift * np.cos(theta))

            # Balik ke koordinat gambar lama
            x_old += cx
            y_old += cy

            # Cek batas
            if 0 <= x_old < w and 0 <= y_old < h:
                rotated[y, x] = image[y_old, x_old]

    return rotated

import numpy as np

def dilatasi(image, skala):
    h, w = image.shape
    if skala < 0:
        scale = abs(skala)

        hasil = np.zeros((h * scale, w * scale)).astype(np.uint8)

        for y in range(h * scale):
            for x in range(w * scale):
                hasil[y][x] = image[int(y / scale)][int(x / scale)]
    elif skala > 0:
        scale = skala

        hasil = np.zeros((h // scale, w // scale)).astype(np.uint8)

        for y in range(h // scale):
            for x in range(w // scale):
                hasil[y][x] = image[int(y * scale)][int(x * scale)]
    else:
        hasil = image.copy()

    return hasil

def dilatasi2(image, scale):
    h, w = image.shape
    new_h = int(h * scale)
    new_w = int(w * scale)
    
    hasil = np.zeros((new_h, new_w), dtype=np.uint8)
    
    for y in range(new_h):
        for x in range(new_w):
            # mapping ke koordinat asli
            src_y = int(y / scale)
            src_x = int(x / scale)
            
            hasil[y, x] = image[src_y, src_x]
    
    return hasil

def crop(image, p1=0, p2=0, p3=0, p4=0):
    h, w = image.shape
    
    sy = max(0, p1)
    ey = min(h, h - p2)
    
    sx = max(0, p3)
    ex = min(w, w - p4)
    
    return image[sy:ey, sx:ex]

def translasi(image, geser_kolom_x=0, geser_baris_y=0): 
    image = np.array(image) 
    h, w = image.shape 
    hasil = np.zeros((h, w), dtype=image.dtype) 
    for i in range(h): 
        for j in range(w): 
            new_i = i + geser_baris_y 
            new_j = j + geser_kolom_x 
            if 0 <= new_i < h and 0 <= new_j < w: 
                hasil[new_i, new_j] = image[i, j] 
    return hasil 

def resize(image, new_width, new_height):
    h, w = image.shape[:2]

    if len(image.shape) == 3:
        result = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    else:
        result = np.zeros((new_height, new_width), dtype=np.uint8)

    x_ratio = w / new_width
    y_ratio = h / new_height

    for y in range(new_height):
        for x in range(new_width):
            src_x = int(x * x_ratio)
            src_y = int(y * y_ratio)

            result[y, x] = image[src_y, src_x]

    return result

def zero_padding(image, pad):
    h, w = image.shape
    padded = np.zeros((h + 2*pad, w + 2*pad))
    
    for i in range(h):
        for j in range(w):
            padded[i+pad][j+pad] = image[i][j]
    
    return padded

def convolution(image, kernel):    
    h, w = image.shape
    k = 4  # ukuran kernel
    pad = k // 2  # = 2
    
    # padding dulu (WAJIB sesuai soal)
    padded = zero_padding(image, pad)
    
    hasil = np.zeros((h, w))
    
    # konvolusi
    for i in range(h):
        for j in range(w):
            total = 0
            for m in range(k):
                for n in range(k):
                    total += padded[i+m][j+n] * kernel[m][n]
            
            hasil[i][j] = total
    
    # biar aman ke citra
    hasil = np.clip(hasil, 0, 255)
    return hasil.astype(np.uint8)

def normalize(img):
    min_val = np.min(img)
    max_val = np.max(img)
    norm_img = ((img - min_val) / (max_val - min_val)) * 255
    return norm_img.astype(np.uint8)

def clipping(image_raw):
    h, w = image_raw.shape
    hasil = np.zeros((h, w), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            piksel = image_raw[i, j]

            if piksel > 255:
                hasil[i, j] = 255
            elif piksel < 0:
                hasil[i, j] = 0
            else:
                hasil[i, j] = int(piksel)

    return hasil

def clipping_rgb(image_raw):

    h, w, c = image_raw.shape

    hasil = np.zeros((h, w, c), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            for k in range(c):

                piksel = image_raw[i, j, k]

                if piksel > 255:
                    hasil[i, j, k] = 255

                elif piksel < 0:
                    hasil[i, j, k] = 0

                else:
                    hasil[i, j, k] = int(piksel)

    return hasil

def histogram(image):
    hist = [0] * 256  # 0-255
    
    for row in image:
        for pixel in row:
            hist[pixel] += 1
            
    return hist

def buat_hist(citra): 
    histogram = [0] * 256 
    height = len(citra) 
    width = len(citra[0]) if height > 0 else 0 
    for i in range(height): 
        for j in range(width): 
            val = int(citra[i][j])   
            histogram[val] += 1   
    return histogram 

def to_grayscale(image):
    h, w, c = image.shape
    gray = np.zeros((h, w), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            r = image[i, j, 2]
            g = image[i, j, 1]
            b = image[i, j, 0]
            gray[i, j] = int(0.299*r + 0.587*g + 0.114*b)

    return gray

def to_gray(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return gray.astype(np.uint8)