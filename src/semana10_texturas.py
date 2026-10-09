import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, measure, color
from skimage.feature import local_binary_pattern
import os

def procesar_imagen(ruta_imagen, radio_lbp=3, puntos_lbp=24):
    print(f"\n--- Procesando: {ruta_imagen} ---")
    
    # 1. Cargar imagen en escala de grises
    img = io.imread(ruta_imagen, as_gray=True)
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
        
    # 2. Segmentación mediante histograma y Otsu
    hist_intensidad, bins = np.histogram(img.flatten(), bins=256, range=(0, 256))
    
    umbral = filters.threshold_otsu(img)
    # Suponiendo fondo blanco y texto/firma oscuro:
    mascara = img < umbral 
    
    # 3. Etiquetado de regiones
    # Eliminamos ruido pequeño antes de etiquetar (opcional, pero útil)
    etiquetas = measure.label(mascara, connectivity=2)
    propiedades = measure.regionprops(etiquetas)
    
    # Filtrar regiones pequeñas
    area_minima = 10
    regiones_validas = [prop for prop in propiedades if prop.area >= area_minima]
    
    num_regiones = len(regiones_validas)
    areas = [prop.area for prop in regiones_validas]
    
    if num_regiones > 0:
        area_promedio = np.mean(areas)
        area_std = np.std(areas)
    else:
        area_promedio = 0.0
        area_std = 0.0
        
    print(f"Umbral Otsu: {umbral:.2f}")
    print(f"Regiones encontradas (Area >= {area_minima}): {num_regiones}")
    print(f"Área promedio: {area_promedio:.2f}, Desviación: {area_std:.2f}")
    
    # 4. Análisis de Texturas LBP
    lbp = local_binary_pattern(img, puntos_lbp, radio_lbp, method="uniform")
    # Número de bins para el método 'uniform' = puntos_lbp + 2
    n_bins = puntos_lbp + 2
    hist_lbp, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    
    # 5. Vector de características
    # Combinamos: [num_regiones, area_promedio, area_std] + hist_intensidad (normalizado) + hist_lbp
    hist_intensidad_norm = hist_intensidad / hist_intensidad.sum()
    vector_caracteristicas = np.concatenate([
        [float(num_regiones), area_promedio, area_std],
        hist_intensidad_norm,
        hist_lbp
    ])
    
    return {
        'img': img,
        'hist_intensidad': hist_intensidad_norm,
        'umbral': umbral,
        'mascara': mascara,
        'lbp': lbp,
        'hist_lbp': hist_lbp,
        'vector': vector_caracteristicas
    }

def main():
    os.makedirs('artifacts', exist_ok=True)
    
    # Procesar ambas imágenes
    res_texto = procesar_imagen('data/semana10/img_texto.png')
    res_firma = procesar_imagen('data/semana10/img_firma.png')
    
    # Guardar vectores de características
    vector_final = np.vstack([res_texto['vector'], res_firma['vector']])
    np.save('artifacts/semana10_features.npy', vector_final)
    print("\nVector de características guardado en 'artifacts/semana10_features.npy'")
    
    # Visualización y guardado de histogramas (artifacts/semana10_histograma.png)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Fila 1: Imagen Texto
    axes[0,0].imshow(res_texto['img'], cmap='gray')
    axes[0,0].set_title("Imagen Texto (Documento)")
    axes[0,0].axis('off')
    
    axes[0,1].plot(res_texto['hist_intensidad'])
    axes[0,1].axvline(res_texto['umbral'], color='r', linestyle='--')
    axes[0,1].set_title(f"Hist. y Umbral Otsu: {res_texto['umbral']:.1f}")
    
    axes[0,2].imshow(res_texto['mascara'], cmap='gray')
    axes[0,2].set_title("Máscara Binaria")
    axes[0,2].axis('off')
    
    axes[0,3].bar(range(len(res_texto['hist_lbp'])), res_texto['hist_lbp'])
    axes[0,3].set_title("Histograma LBP Uniforme")
    
    # Fila 2: Imagen Firma
    axes[1,0].imshow(res_firma['img'], cmap='gray')
    axes[1,0].set_title("Imagen Firma")
    axes[1,0].axis('off')
    
    axes[1,1].plot(res_firma['hist_intensidad'])
    axes[1,1].axvline(res_firma['umbral'], color='r', linestyle='--')
    axes[1,1].set_title(f"Hist. y Umbral Otsu: {res_firma['umbral']:.1f}")
    
    axes[1,2].imshow(res_firma['mascara'], cmap='gray')
    axes[1,2].set_title("Máscara Binaria")
    axes[1,2].axis('off')
    
    axes[1,3].bar(range(len(res_firma['hist_lbp'])), res_firma['hist_lbp'])
    axes[1,3].set_title("Histograma LBP Uniforme")
    
    plt.tight_layout()
    plt.savefig('artifacts/semana10_histograma.png')
    plt.close()
    print("Gráficas guardadas en 'artifacts/semana10_histograma.png'")

if __name__ == "__main__":
    main()
