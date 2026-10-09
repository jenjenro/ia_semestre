import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

os.makedirs('data/semana10', exist_ok=True)

# 1. Imagen de Texto (Factura / Contrato) con ruido
ruido1 = np.random.normal(240, 10, (400, 400)).astype(np.uint8)
img_texto = Image.fromarray(ruido1)
draw_texto = ImageDraw.Draw(img_texto)
# Simulamos lineas de texto
for y in range(40, 360, 20):
    for x in range(40, 360, 10):
        if np.random.rand() > 0.3:
            draw_texto.rectangle([x, y, x+8, y+10], fill=np.random.randint(20, 80))
img_texto.save('data/semana10/img_texto.png')

# 2. Imagen de Firma con ruido
ruido2 = np.random.normal(240, 10, (400, 400)).astype(np.uint8)
img_firma = Image.fromarray(ruido2)
draw_firma = ImageDraw.Draw(img_firma)
# Simulamos una firma con curvas
points = []
cx, cy = 100, 200
for i in range(50):
    cx += np.random.randint(2, 10)
    cy += np.random.randint(-15, 15)
    points.append((cx, cy))
draw_firma.line(points, fill=40, width=5)
# Alguna otra parte de la firma
points2 = []
cx, cy = 150, 180
for i in range(30):
    cx += np.random.randint(1, 15)
    cy += np.random.randint(-25, 25)
    points2.append((cx, cy))
draw_firma.line(points2, fill=60, width=3)
img_firma.save('data/semana10/img_firma.png')

print("Imágenes generadas correctamente.")
