# PCDLib

PCDLib is a Python-based digital image processing library created for learning, experimentation, and implementation of fundamental image processing techniques. This project provides various image manipulation and analysis functions such as histogram processing, masking, thresholding, filtering, color transformation, image compositing, and more.

Built with simplicity in mind, PCDLib is suitable for students, beginners in image processing, and academic projects related to Digital Image Processing (Pengolahan Citra Digital).

---

## 🚧 Development Status

PCDLib is actively being developed and expanded. More image processing libraries, algorithms, and advanced computer vision features will be added in future updates.

Planned additions include:

* Advanced filtering techniques
* Morphological transformations
* Feature extraction
* Object detection utilities
* Image segmentation methods
* Machine learning integration for image analysis
* Additional helper modules and utilities

The library is designed to grow progressively alongside ongoing learning and experimentation in digital image processing.

---

## ✨ Features

* Image loading and visualization
* RGB and grayscale image processing
* Histogram calculation and visualization
* Histogram equalization
* Histogram specification
* Thresholding techniques
* Image masking
* Cropping and image merging
* Noise reduction filters
* Edge detection operations
* Pixel manipulation using NumPy
* Visualization with Matplotlib

---

## 🛠️ Technologies Used

* Python
* NumPy
* OpenCV (cv2)
* Matplotlib
* Jupyter Notebook

---

## 📂 Project Structure

```bash
PCDLib/
│
├── notebooks/          # Jupyter notebooks for experiments and testing
├── images/             # Sample images used in processing
├── pcdlib/             # Main library source code
├── examples/           # Example implementations
├── requirements.txt    # Python dependencies
└── README.md
```

---

## 🚀 Installation

Clone this repository:

```bash
git clone https://github.com/samarawards/pcdlib.git
cd pcdlib
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Example Usage

### Display an Image

```python
import matplotlib.pyplot as plt
import cv2 as cv

image = cv.imread('image.jpg')
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

plt.imshow(image)
plt.axis('off')
plt.show()
```

### Convert RGB Image to Grayscale

```python
gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

plt.imshow(gray, cmap='gray')
plt.axis('off')
plt.show()
```

### Histogram Visualization

```python
import matplotlib.pyplot as plt

plt.hist(gray.ravel(), bins=256, range=[0,256])
plt.title('Histogram')
plt.show()
```

---

## 📚 Learning Goals

This project was developed to:

* Understand fundamental image processing concepts
* Practice implementing algorithms manually
* Explore OpenCV and NumPy operations
* Support digital image processing coursework and experiments
* Build reusable image processing utilities

---

## 🧠 Topics Covered

* Point Operations
* Histogram Processing
* Spatial Filtering
* Thresholding
* Morphological Operations
* Edge Detection
* Image Enhancement
* Image Transformation
* Masking and Compositing

---

## 📸 Sample Processing Ideas

* Detect edges using Sobel or Prewitt operators
* Improve image contrast using histogram equalization
* Create image masks and overlays
* Reduce image noise using filtering techniques
* Perform binary threshold segmentation

---

## 🤝 Contributing

Contributions, suggestions, and improvements are welcome.

1. Fork the repository
2. Create a new branch
3. Commit your changes
4. Push to your branch
5. Open a pull request

---

## 👩‍💻 Author

Developed by Ara.

GitHub: [samarawards GitHub Profile](https://github.com/samarawards?utm_source=chatgpt.com)
