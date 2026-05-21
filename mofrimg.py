import numpy as np


# =========================================================
# KERNEL
# =========================================================

kernel_cross = np.array([
    [0,1,0],
    [1,1,1],
    [0,1,0]
], dtype=np.uint8)

kernel_diamond = np.array([
    [0,0,1,0,0],
    [0,1,1,1,0],
    [1,1,1,1,1],
    [0,1,1,1,0],
    [0,0,1,0,0]
], dtype=np.uint8)

kernel_x = np.array([
    [1,0,0,0,1],
    [0,1,0,1,0],
    [0,0,1,0,0],
    [0,1,0,1,0],
    [1,0,0,0,1]
], dtype=np.uint8)


# =========================================================
# THRESHOLDING
# =========================================================

def threshold(image, batas):
    """
    Mengubah citra grayscale menjadi biner.
    """

    baris, kolom = image.shape
    hasil = np.zeros_like(image, dtype=np.uint8)

    for i in range(baris):
        for j in range(kolom):

            if image[i, j] > batas:
                hasil[i, j] = 255
            else:
                hasil[i, j] = 0

    return hasil


# =========================================================
# RESIZE
# =========================================================

def resize(image, new_width, new_height):
    """
    Resize manual nearest neighbor.
    """

    old_height, old_width = image.shape[:2]

    hasil = np.zeros((new_height, new_width), dtype=np.uint8)

    for i in range(new_height):
        for j in range(new_width):

            x = int(j * old_width / new_width)
            y = int(i * old_height / new_height)

            hasil[i, j] = image[y, x]

    return hasil


# =========================================================
# DILASI
# =========================================================

def dilasi(image, kernel):
    """
    Memperbesar area objek putih.
    """

    height, width = image.shape
    k_height, k_width = kernel.shape

    center = k_height // 2

    hasil = np.zeros((height, width), dtype=np.uint8)

    for i in range(center, height - center):
        for j in range(center, width - center):

            if image[i, j] == 255:

                for k in range(k_height):
                    for l in range(k_width):

                        if kernel[k, l] == 1:
                            hasil[i + k - center,j + l - center] = 255

    return hasil


# =========================================================
# EROSI
# =========================================================

def erosi(image, kernel):
    """
    Menipiskan area objek putih.
    """

    height, width = image.shape
    k_height, k_width = kernel.shape

    center = k_height // 2

    hasil = np.zeros((height, width), dtype=np.uint8)

    for i in range(center, height - center):
        for j in range(center, width - center):

            cocok = True

            for k in range(k_height):
                for l in range(k_width):

                    if (
                        kernel[k, l] == 1 and
                        image[i + k - center, j + l - center] == 0
                    ):
                        cocok = False
                        break

                if not cocok:
                    break

            if cocok:
                hasil[i, j] = 255

    return hasil


# =========================================================
# OPENING
# =========================================================

def opening(image, kernel):
    """
    Erosi lalu dilasi.
    Menghilangkan noise kecil.
    """

    hasil_erosi = erosi(image, kernel)
    hasil = dilasi(hasil_erosi, kernel)

    return hasil


# =========================================================
# CLOSING
# =========================================================

def closing(image, kernel):
    """
    Dilasi lalu erosi.
    Menutup lubang kecil pada objek.
    """

    hasil_dilasi = dilasi(image, kernel)
    hasil = erosi(hasil_dilasi, kernel)

    return hasil


# =========================================================
# THINNING
# =========================================================

def thinning(img):
    """
    Menipiskan objek hingga skeleton.
    """

    binary = (img > 0).astype(np.uint8)

    height, width = binary.shape

    changed = True

    while changed:

        changed = False
        hapus = []

        # STEP 1
        for i in range(1, height - 1):
            for j in range(1, width - 1):

                P1 = binary[i, j]

                if P1 != 1:
                    continue

                P2 = binary[i-1, j]
                P3 = binary[i-1, j+1]
                P4 = binary[i, j+1]
                P5 = binary[i+1, j+1]
                P6 = binary[i+1, j]
                P7 = binary[i+1, j-1]
                P8 = binary[i, j-1]
                P9 = binary[i-1, j-1]

                tetangga = [P2,P3,P4,P5,P6,P7,P8,P9]

                jumlah = np.sum(tetangga)

                transisi = 0

                urutan = tetangga + [P2]

                for k in range(8):
                    if urutan[k] == 0 and urutan[k+1] == 1:
                        transisi += 1

                if (
                    2 <= jumlah <= 6 and
                    transisi == 1 and
                    P2 * P4 * P6 == 0 and
                    P4 * P6 * P8 == 0
                ):
                    hapus.append((i, j))

        if hapus:
            changed = True

            for i, j in hapus:
                binary[i, j] = 0

        hapus = []

        # STEP 2
        for i in range(1, height - 1):
            for j in range(1, width - 1):

                P1 = binary[i, j]

                if P1 != 1:
                    continue

                P2 = binary[i-1, j]
                P3 = binary[i-1, j+1]
                P4 = binary[i, j+1]
                P5 = binary[i+1, j+1]
                P6 = binary[i+1, j]
                P7 = binary[i+1, j-1]
                P8 = binary[i, j-1]
                P9 = binary[i-1, j-1]

                tetangga = [P2,P3,P4,P5,P6,P7,P8,P9]

                jumlah = np.sum(tetangga)

                transisi = 0

                urutan = tetangga + [P2]

                for k in range(8):
                    if urutan[k] == 0 and urutan[k+1] == 1:
                        transisi += 1

                if (
                    2 <= jumlah <= 6 and
                    transisi == 1 and
                    P2 * P4 * P8 == 0 and
                    P2 * P6 * P8 == 0
                ):
                    hapus.append((i, j))

        if hapus:
            changed = True

            for i, j in hapus:
                binary[i, j] = 0

    return binary * 255


# =========================================================
# THICKENING
# =========================================================

def thickening(image, kernel, iterasi=1):
    """
    Menebalkan objek.
    """

    hasil = image.copy()

    for _ in range(iterasi):
        hasil = dilasi(hasil, kernel)

    return hasil


# =========================================================
# MORPHOLOGICAL GRADIENT
# =========================================================

def gradient_morph(image, kernel):
    """
    Selisih dilasi dan erosi.
    Menonjolkan tepi objek.
    """

    hasil_dilasi = dilasi(image, kernel)
    hasil_erosi = erosi(image, kernel)

    hasil = hasil_dilasi.astype(np.int16) - hasil_erosi.astype(np.int16)

    hasil = np.clip(hasil, 0, 255)

    return hasil.astype(np.uint8)