import cv2
import numpy as np
import pandas as pd
from scipy.stats import entropy
from skimage.feature import graycomatrix, graycoprops


# =========================================================
# FUNGSI DASAR
# =========================================================

# Mengubah gambar BGR menjadi grayscale
def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Normalisasi nilai piksel ke rentang 0 - 255
def normalize(image):
    return ((image - np.min(image)) /
           (np.max(image) - np.min(image)) * 255).astype(np.uint8)


# =========================================================
# KERNEL HAAR WAVELET
# =========================================================

# Nilai konstanta Haar Wavelet
nilai = 1 / np.sqrt(2)

# Low Pass Filter
# Digunakan untuk mengambil informasi frekuensi rendah
LPF = np.array([nilai, nilai])

# High Pass Filter
# Digunakan untuk mengambil detail frekuensi tinggi
HPF = np.array([nilai, -nilai])


# =========================================================
# KONVOLUSI
# =========================================================

# Konvolusi horizontal
def convo_h(image, kernel):

    # Panjang kernel
    ukuran_kernel = len(kernel)

    # Padding horizontal
    padd = np.pad(
        image,
        ((0, 0), (0, ukuran_kernel)),
        mode='constant'
    )

    # Menyimpan hasil konvolusi
    conv = np.zeros_like(image, dtype=float)

    # Proses konvolusi
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(ukuran_kernel):

                # Mengalikan kernel dengan piksel
                conv[i, j] += padd[i, j+k] * kernel[k]

    return conv


# Konvolusi vertical
def convo_v(image, kernel):

    ukuran_kernel = len(kernel)

    # Padding vertical
    padd = np.pad(
        image,
        ((0, ukuran_kernel), (0, 0)),
        mode='constant'
    )

    # Array hasil konvolusi
    conv = np.zeros_like(image, dtype=float)

    # Proses konvolusi
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(ukuran_kernel):

                # Mengalikan kernel dengan piksel
                conv[i, j] += padd[i+k, j] * kernel[k]

    return conv


# =========================================================
# DOWNSAMPLING
# =========================================================

# Downsampling horizontal
# Mengambil setiap 2 kolom sekali
def downsampling_h(image):
    return image[:, ::2]


# Downsampling vertical
# Mengambil setiap 2 baris sekali
def downsampling_v(image):
    return image[::2, :]


# =========================================================
# UPSAMPLING
# =========================================================

# Upsampling horizontal
# Menambahkan kolom kosong di antara piksel
def upsampling_h(image):

    h, w = image.shape

    # Membuat array baru dengan lebar 2x lipat
    upsampled = np.zeros((h, w*2))

    # Mengisi piksel asli pada indeks genap
    upsampled[:, ::2] = image

    return upsampled


# Upsampling vertical
# Menambahkan baris kosong di antara piksel
def upsampling_v(image):

    h, w = image.shape

    # Membuat array baru dengan tinggi 2x lipat
    upsampled = np.zeros((h*2, w))

    # Mengisi piksel asli pada indeks genap
    upsampled[::2, :] = image

    return upsampled


# =========================================================
# TRANSFORMASI WAVELET LEVEL 1
# =========================================================

def wavelet(image):

    # -----------------------------------------
    # LL (Approximation)
    # Frekuensi rendah horizontal & vertical
    # -----------------------------------------

    c1 = convo_h(image, LPF)
    ds1 = downsampling_h(c1)

    c2 = convo_v(ds1, LPF)
    LL = downsampling_v(c2)

    # -----------------------------------------
    # LH (Horizontal Detail)
    # Frekuensi rendah horizontal
    # Frekuensi tinggi vertical
    # -----------------------------------------

    c1 = convo_h(image, LPF)
    ds1 = downsampling_h(c1)

    c2 = convo_v(ds1, HPF)
    LH = downsampling_v(c2)

    # -----------------------------------------
    # HL (Vertical Detail)
    # Frekuensi tinggi horizontal
    # Frekuensi rendah vertical
    # -----------------------------------------

    c1 = convo_h(image, HPF)
    ds1 = downsampling_h(c1)

    c2 = convo_v(ds1, LPF)
    HL = downsampling_v(c2)

    # -----------------------------------------
    # HH (Diagonal Detail)
    # Frekuensi tinggi horizontal & vertical
    # -----------------------------------------

    c1 = convo_h(image, HPF)
    ds1 = downsampling_h(c1)

    c2 = convo_v(ds1, HPF)
    HH = downsampling_v(c2)

    # Mengembalikan hasil normalisasi
    return (
        normalize(LL),
        normalize(LH),
        normalize(HL),
        normalize(HH)
    )


# =========================================================
# DEKOMPOSISI MULTI LEVEL
# =========================================================

def multi_wavelet(image, level=1):

    # Menyimpan hasil tiap level
    hasil = []

    # Copy image agar data asli aman
    current = image.copy()

    # Proses dekomposisi bertingkat
    for i in range(level):

        LL, LH, HL, HH = wavelet(current)

        # Menyimpan hasil tiap level
        hasil.append((LL, LH, HL, HH))

        # LL dipakai untuk level berikutnya
        current = LL

    return hasil


# =========================================================
# REKONSTRUKSI WAVELET
# =========================================================

def reconstruct(LL, LH, HL, HH):

    # -----------------------------------------
    # Rekonstruksi bagian low frequency
    # -----------------------------------------

    up = upsampling_v(LL)
    convL = convo_v(up, LPF)

    up = upsampling_v(LH)
    convH = convo_v(up, HPF)

    low = upsampling_h(convL + convH)
    low = convo_h(low, LPF)

    # -----------------------------------------
    # Rekonstruksi bagian high frequency
    # -----------------------------------------

    up = upsampling_v(HL)
    convL = convo_v(up, LPF)

    up = upsampling_v(HH)
    convH = convo_v(up, HPF)

    high = upsampling_h(convL + convH)
    high = convo_h(high, HPF)

    # Menggabungkan hasil rekonstruksi
    hasil = low + high

    return normalize(hasil)


# =========================================================
# GLCM (GRAY LEVEL CO-OCCURRENCE MATRIX)
# =========================================================

def glcm(image, angle=0):

    # Dictionary sudut GLCM
    sudut = {
        0: 0,
        45: np.pi/4,
        90: np.pi/2,
        135: 3*np.pi/4
    }

    # Membuat matriks GLCM
    matriks = graycomatrix(
        image,
        [1],                 # jarak antar piksel
        [sudut[angle]],      # sudut
        256,                 # level grayscale
        symmetric=True,
        normed=True
    )

    return matriks


# =========================================================
# EKSTRAKSI FITUR GLCM
# =========================================================

def glcm_features(matriks):

    # Mengambil fitur statistik dari GLCM
    fitur = {

        # Mengukur kontras tekstur
        'contrast':
            graycoprops(matriks, 'contrast')[0,0],

        # Mengukur homogenitas tekstur
        'homogeneity':
            graycoprops(matriks, 'homogeneity')[0,0],

        # Mengukur keteraturan tekstur
        'energy':
            graycoprops(matriks, 'energy')[0,0],

        # Mengukur hubungan antar piksel
        'correlation':
            graycoprops(matriks, 'correlation')[0,0],

        # Mengukur ketidaksamaan tekstur
        'dissimilarity':
            graycoprops(matriks, 'dissimilarity')[0,0],

        # Angular Second Moment
        'ASM':
            graycoprops(matriks, 'ASM')[0,0],

        # Mengukur tingkat keacakan tekstur
        'entropy':
            entropy(matriks.ravel())
    }

    return fitur