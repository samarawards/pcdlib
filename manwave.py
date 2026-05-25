import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
import os
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

# Ubah ukuran gambar jadi genap
def even_size(image):

    h, w = image.shape

    # Jika tinggi ganjil
    if h % 2 != 0:
        image = image[:-1, :]

    # Jika lebar ganjil
    if w % 2 != 0:
        image = image[:, :-1]

    return image

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

    image = even_size(image)
    
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
    return (LL, LH, HL, HH)
    # return (
    #     normalize(LL),
    #     normalize(LH),
    #     normalize(HL),
    #     normalize(HH)
    # )

# =========================================================
# DISPLAY TRANSFORMASI WAVELET LEVEL 1
# =========================================================

def display_wavelet(subband):
    return normalize(subband)

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

def dynamic_decomposition(image, level=5):

    image = even_size(image)

    canvas = image.copy().astype(np.float32)

    current_x = 0
    current_y = 0

    current = image.copy()

    for i in range(level):

        LL, LH, HL, HH = wavelet(current)

        h, w = LL.shape

        canvas[current_y:current_y+h, current_x:current_x+w] = normalize(LL)

        canvas[current_y:current_y+h, current_x+w:current_x+(w*2)] = normalize(LH)

        canvas[current_y+h:current_y+(h*2), current_x:current_x+w] = normalize(HL)

        canvas[current_y+h:current_y+(h*2), current_x+w:current_x+(w*2)] = normalize(HH)

        current = LL

    return normalize(canvas)

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
# GLCM (GRAY LEVEL CO-OCCURRENCE MATRIX) MULTI ANGLES
# =========================================================

def glcm_all_angles(image):

    hasil = []

    angles = [0, 45, 90, 135]

    for angle in angles:

        matriks = glcm(
            normalize(image),
            angle
        )

        fitur = glcm_features(matriks)

        fitur["angle"] = angle

        hasil.append(fitur)

    return pd.DataFrame(hasil)

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

# =========================================================
# DISPLAY FITUR GLCM
# =========================================================

def glcm_table(title, subbands, angle=0):

    data = []

    for nama, gambar in subbands.items():

        matriks = glcm(
            normalize(gambar),
            angle
        )

        fitur = glcm_features(matriks)

        fitur["subband"] = nama

        data.append(fitur)

    print(title)

    return pd.DataFrame(data)

"""
CARA PAKAI
df = wave.glcm_table(

    "FITUR GLCM",

    {
        "LL": LL,
        "LH": LH,
        "HL": HL,
        "HH": HH
    },

    angle=0
)

biar bisa
{
    "LL level 3": LL3
}
"""

# =========================================================
# SAVE FITUR
# =========================================================

def save_features(df, filename):

    df.to_csv(filename, index=False)

# =========================================================
# DATASET GLCM
# =========================================================

def dataset_glcm(folder, label, angle=0):

    data = []

    files = os.listdir(folder)

    for file in files:

        path = os.path.join(folder, file)

        img = cv2.imread(path)

        gray = grayscale(img)

        LL, LH, HL, HH = wavelet(gray)

        matriks = glcm(
            normalize(LL),
            angle
        )

        fitur = glcm_features(matriks)

        fitur["filename"] = file
        fitur["label"] = label

        data.append(fitur)

    return pd.DataFrame(data)


# =========================================================
# COMPARE IMAGE
# =========================================================

def compare_images(image1, image2):

    diff = cv2.absdiff(
        normalize(image1),
        normalize(image2)
    )

    return np.mean(diff)

"""
Cara pakai
score = compare_images(
    hasil_LL,
    reference_LL
)
"""

# =========================================================
# DISPLAY WAVE
# =========================================================

def show_subbands(LL, LH, HL, HH):

    fig, ax = plt.subplots(2, 2, figsize=(8,8))

    ax[0,0].imshow(normalize(LL), cmap='gray')
    ax[0,0].set_title("LL")

    ax[0,1].imshow(normalize(LH), cmap='gray')
    ax[0,1].set_title("LH")

    ax[1,0].imshow(normalize(HL), cmap='gray')
    ax[1,0].set_title("HL")

    ax[1,1].imshow(normalize(HH), cmap='gray')
    ax[1,1].set_title("HH")

    for a in ax.ravel():
        a.axis('on')

    plt.tight_layout()
    plt.show()

# =========================================================
# SAVE WAVE
# =========================================================

def save_subbands(LL, LH, HL, HH, path="hasil"):

    cv2.imwrite(f"{path}_LL.jpg", normalize(LL))
    cv2.imwrite(f"{path}_LH.jpg", normalize(LH))
    cv2.imwrite(f"{path}_HL.jpg", normalize(HL))
    cv2.imwrite(f"{path}_HH.jpg", normalize(HH))

"""
CARA PAKAI
import cv2
from pcdlib import manwave

img = cv2.imread('rinjani.jpg')

gray = manwave.grayscale(img)

LL, LH, HL, HH = manwave.wavelet(gray)

hasil = manwave.reconstruct(LL, LH, HL, HH)

mat = manwave.glcm(LL, 0)

fitur = manwave.glcm_features(mat)

print(fitur)
"""