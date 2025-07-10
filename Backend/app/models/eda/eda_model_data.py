import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

output_dir = os.path.join('Backend', 'app', 'models', 'eda')
os.makedirs(output_dir, exist_ok=True)

# --- 1. Cargar y Preparar los Datos ---
try:
    # Lee el archivo Excel especificado
    file_path = 'Datos_Arduino.xlsx'
    df = pd.read_excel('C:\Entorno\Datos_Arduino.xlsx')
except FileNotFoundError:
    print("Error: No se encontró el archivo")
    exit()

# Asegurarse de que ambas columnas sean de tipo texto (string)
df['Fecha'] = df['Fecha'].astype(str)
df['Hora'] = df['Hora'].astype(str)

# Combina 'Fecha' y 'Hora' en una sola columna 'timestamp' y la convierte al formato datetime
df['timestamp'] = pd.to_datetime(df['Fecha'].str.split(' ').str[0] + ' ' + df['Hora'])

# Establece la columna 'timestamp' como el índice del DataFrame
df.set_index('timestamp', inplace=True)

# Elimina las columnas originales de 'Fecha' y 'Hora' que ya no son necesarias
df.drop(['Fecha', 'Hora'], axis=1, inplace=True)

print(df)

print("\n--- Información y Tipos de Datos ---")
df.info()

print("\n--- Revisión de Valores Nulos ---")
print(df.isnull().sum())

# --- 2. Análisis Descriptivo ---
print("\n--- Resumen Estadístico de las Variables ---")
# Muestra estadísticas clave como media, desviación, mínimo, máximo, etc.
print(df.describe())

# --- 3. Visualización de los Datos ---
print("\nGenerando visualizaciones...")

# Gráfico 1: Evolución de la Temperatura y Humedad a lo largo del día
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax1 = plt.subplots(figsize=(15, 7))

ax1.set_xlabel('Hora del día')
ax1.set_ylabel('Temperatura (°C)', color='red')
ax1.plot(df.index, df['Temperatura'], color='red', label='Temperatura')
ax1.tick_params(axis='y', labelcolor='red')

# Crea un segundo eje Y para la humedad
ax2 = ax1.twinx()
ax2.set_ylabel('Humedad (%)', color='blue')
ax2.plot(df.index, df['Humedad'], color='blue', label='Humedad')
ax2.tick_params(axis='y', labelcolor='blue')

plt.title('Temperatura y Humedad a lo largo del Día (20/05/2025)')
fig.tight_layout()

# Guardamos
save_path_1 = os.path.join(output_dir, 'evolucion_temperatura_humedad.png')
plt.savefig(save_path_1, dpi=300)
plt.show()

# Gráfico 2: Evolución de las lecturas del LDR (Luz)
fig, ax = plt.subplots(figsize=(15, 7))
ax.plot(df.index, df['LdrValorAnalog'], label='Valor LDR (Luz)', color='orange')
ax.set_xlabel('Hora del día')
ax.set_ylabel('Valor Analógico LDR (0-1023)')
ax.set_title('Nivel de Luz Detectado a lo largo del Día')
ax.legend()

# Guardamos
save_path_2 = os.path.join(output_dir, 'evolucion_nivel_luz.png')
plt.savefig(save_path_2, dpi=300)
plt.show()

# Gráfico 3: Correlación entre todas las variables
plt.figure(figsize=(10, 8))
# Calcula la matriz de correlación
correlation_matrix = df.corr()
# Dibuja el mapa de calor
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Mapa de Calor de Correlación de Variables')

# Guardamos
save_path_3 = os.path.join(output_dir, 'mapa_correlacion_variables.png')
plt.savefig(save_path_3, dpi=300)
plt.show()

# --- 4. Guardar el DataFrame Procesado ---
output_csv_path = os.path.join(output_dir, 'datos_procesados.csv')
df.to_csv(output_csv_path)

print("\n✅ EDA finalizado.")
print("El DataFrame final preprocesado es:")
print(df.head())