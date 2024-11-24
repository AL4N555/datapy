import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def box_counting(series, n_box_sizes=50):
    """
    Calcula la dimensión fractal usando el método de box-counting.
    """
    # Convertir la serie a un array de NumPy
    series_array = series.to_numpy()
    # Normalizar la serie entre 0 y 1
    series_normalizada = (series_array - np.min(series_array)) / (np.max(series_array) - np.min(series_array))
    
    # Tamaños de las cajas (escala logarítmica)
    box_sizes = np.logspace(np.log10(1), np.log10(len(series_normalizada) // 2), n_box_sizes, dtype=int)
    box_sizes = np.unique(box_sizes)  # Eliminar duplicados
    
    counts = []
    for size in box_sizes:
        # Dividir en cajas y contar cuántas contienen datos
        n_boxes = len(series_normalizada) // size
        reshaped = series_normalizada[:n_boxes * size].reshape(n_boxes, size)
        max_vals = np.max(reshaped, axis=1)
        min_vals = np.min(reshaped, axis=1)
        counts.append(np.sum(max_vals - min_vals > 0))  # Contar cajas con datos
    
    return box_sizes, counts

# Cargar datos desde Excel
archivo_excel = 'datos_fractales.xlsx'  # Nombre del archivo Excel
df = pd.read_excel(archivo_excel)

# Suponemos que la columna que queremos analizar se llama 'Serie'
serie = df['Serie']

# Calcular dimensión fractal
box_sizes, counts = box_counting(serie)

# Ajustar una línea para calcular la dimensión fractal
log_box_sizes = np.log(box_sizes)
log_counts = np.log(counts)
slope, _ = np.polyfit(log_box_sizes, log_counts, 1)  # Pendiente de la recta

# Mostrar la dimensión fractal
dim_fractal = -slope
print(f"Dimensión fractal estimada: {dim_fractal:.4f}")

# Graficar resultados
plt.figure(figsize=(10, 6))

# Serie de datos
plt.subplot(2, 1, 1)
plt.plot(serie, label="Serie", color="blue")
plt.title("Serie Temporal")
plt.xlabel("Indice")
plt.ylabel("Valor")
plt.legend()

# Gráfico de box-counting
plt.subplot(2, 1, 2)
plt.plot(log_box_sizes, log_counts, 'o-', label="Datos")
plt.plot(log_box_sizes, np.polyval([slope, _], log_box_sizes), 'r--', label="Ajuste lineal")
plt.title(f"Análisis Fractal (Dimensión: {dim_fractal:.4f})")
plt.xlabel("log(Tamaño de caja)")
plt.ylabel("log(Cantidad de cajas)")
plt.legend()

plt.tight_layout()
plt.savefig('analisis_fractal.png')  # Guardar la gráfica
plt.show()

