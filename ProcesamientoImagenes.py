import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar la imagen
img = cv2.imread('panel_solar.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Mostrar la imagen original
plt.figure(figsize=(12, 8))
plt.subplot(231), plt.imshow(img_rgb), plt.title('Original')

# Redimensionar
img_resized = cv2.resize(img_rgb, (300, 200))
plt.subplot(232), plt.imshow(img_resized), plt.title('Redimensionada')

# Rotar
rows, cols, _ = img_rgb.shape
M = cv2.getRotationMatrix2D((cols/2, rows/2), 15, 1)
img_rotated = cv2.warpAffine(img_rgb, M, (cols, rows))
plt.subplot(233), plt.imshow(img_rotated), plt.title('Rotada')

# Detectar bordes (Canny)
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
img_edges = cv2.Canny(img_gray, 100, 200)
plt.subplot(234), plt.imshow(img_edges, cmap='gray'), plt.title('Bordes')

# Segmentación simple
_, img_threshold = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
plt.subplot(235), plt.imshow(img_threshold, cmap='gray'), plt.title('Segmentación')

# Encontrar contornos
contours, _ = cv2.findContours(img_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
img_contours = img_rgb.copy()
cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)
plt.subplot(236), plt.imshow(img_contours), plt.title('Contornos')

plt.tight_layout()
plt.show()

# Información sobre los contornos encontrados
print(f"Número de contornos encontrados: {len(contours)}")
for i, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    print(f"Área del contorno {i+1}: {area:.2f} píxeles cuadrados")