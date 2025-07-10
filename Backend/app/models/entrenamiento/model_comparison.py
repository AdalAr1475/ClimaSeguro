import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import joblib

# Ignorar advertencias para una salida más limpia
import warnings
warnings.filterwarnings('ignore')

# --- Importación de Modelos y Métricas ---
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Modelos de Scikit-learn
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# Modelos de Boosting
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

# Modelos de Series de Tiempo
from statsmodels.tsa.arima.model import ARIMA

# Modelo de Red Neuronal (LSTM)
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def main():
    """
    Función principal que ejecuta todo el proceso de carga, entrenamiento,
    evaluación y visualización de los modelos de Machine Learning.
    """
    # --- 1. Carga y Preparación de Datos ---
    # Ruta al archivo procesado y a la carpeta de salida para los gráficos
    data_path = os.path.join('Backend', 'app', 'models', 'eda', 'datos_procesados.csv')
    output_dir = os.path.join('Backend', 'app', 'models','entrenamiento')
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Cargar los datos, parsear la columna de fecha y establecerla como índice
        df = pd.read_csv(data_path, parse_dates=['timestamp'], index_col='timestamp')
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo '{data_path}'. Asegúrate de haber ejecutado el script de EDA primero.")
        exit()

    # Feature Engineering: Crear características basadas en el tiempo desde el índice
    df['hora'] = df.index.hour
    df['dia_semana'] = df.index.dayofweek # Lunes=0, Domingo=6
    df['mes'] = df.index.month

    # Definir características (X) y variable objetivo (y)
    # Usaremos las otras variables y las características de tiempo para predecir la Temperatura
    features = ['Humedad', 'LdrValorAnalog', 'hora', 'dia_semana', 'mes']
    target = 'Temperatura'

    X = df[features]
    y = df[target]

    # División de datos en entrenamiento y prueba (80/20)
    # Para series de tiempo, es crucial NO barajar los datos (shuffle=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # --- 2. Implementación y Evaluación de Modelos ---
    print("\n--- 🚀 Iniciando entrenamiento y evaluación de 10 modelos ---")

    # Diccionario para almacenar los resultados de cada modelo
    results = {}
    predictions = {}

    # Función para evaluar y almacenar resultados
    def evaluate_model(name, y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        results[name] = {'RMSE': rmse, 'MAE': mae, 'R²': r2}
        predictions[name] = y_pred
        print(f"✅ {name:25} -> RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

    # Modelos estandar
    models_to_run = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(),
        "Random Forest Regressor": RandomForestRegressor(random_state=42),
        "SVR": SVR(),
        "XGBoost": xgb.XGBRegressor(random_state=42),
        "LightGBM": lgb.LGBMRegressor(random_state=42),
        "CatBoost": CatBoostRegressor(verbose=0, random_state=42)
    }

    # Entrenar y evaluar modelos estándar
    for name, model in models_to_run.items():
        start_time = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        evaluate_model(name, y_test, y_pred)
        print(f"   (Tiempo: {time.time() - start_time:.2f}s)")

        # Guardar modelo
        joblib.dump(model, os.path.join(output_dir, f'{name.replace(" ", "_")}.joblib'))
        
    # --- Modelos especiales de Series de Tiempo ---

    # 8. ARIMA
    start_time = time.time()
    # ARIMA solo usa la historia de la variable objetivo (univariado)
    # El orden (p,d,q) es un hiperparámetro. (5,1,0) es un punto de partida común.
    try:
        arima_model = ARIMA(y_train, order=(5, 1, 0)).fit()
        arima_pred = arima_model.forecast(steps=len(y_test))
        evaluate_model("ARIMA", y_test, arima_pred)
        arima_model.save(os.path.join(output_dir, 'ARIMA.pkl'))
        print(f"   (Tiempo: {time.time() - start_time:.2f}s)")
    except Exception as e:
        print(f"❌ ARIMA falló: {e}")

    # 10. LSTM
    start_time = time.time()
    # LSTM requiere escalar los datos y reformatearlos en secuencias
    scaler = MinMaxScaler()
    y_train_scaled = scaler.fit_transform(y_train.values.reshape(-1, 1))

    def create_sequences(data, seq_length):
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            xs.append(data[i:i+seq_length])
            ys.append(data[i+seq_length])
        return np.array(xs), np.array(ys)

    SEQ_LENGTH = 10 # Usar 10 puntos de datos anteriores para predecir el siguiente
    X_train_lstm, y_train_lstm = create_sequences(y_train_scaled, SEQ_LENGTH)

    lstm_model = Sequential([
        LSTM(50, activation='relu', input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
        Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    lstm_model.fit(X_train_lstm, y_train_lstm, epochs=20, batch_size=32, verbose=0)

    # Predecir con LSTM
    inputs = scaler.transform(y[-len(y_test) - SEQ_LENGTH:].values.reshape(-1, 1))
    X_test_lstm, _ = create_sequences(inputs, SEQ_LENGTH)
    lstm_pred_scaled = lstm_model.predict(X_test_lstm)
    lstm_pred = scaler.inverse_transform(lstm_pred_scaled)
    evaluate_model("LSTM", y_test, lstm_pred.flatten())

    # Guardar modelo y escalador
    lstm_model.save(os.path.join(output_dir, 'LSTM.h5'))
    joblib.dump(scaler, os.path.join(output_dir, 'LSTM_scaler.joblib'))
    print(f"   (Tiempo: {time.time() - start_time:.2f}s)")


    # --- 3. Comparación y Visualización de Resultados ---
    # Convertir el diccionario de resultados a un DataFrame
    results_df = pd.DataFrame(results).T.sort_values(by='RMSE')
    results_df.to_csv(os.path.join(output_dir, 'model_metrics.csv'))

    # Gráfico de Barras: Comparación de RMSE
    plt.figure(figsize=(12, 7))
    sns.barplot(x=results_df.index, y=results_df['RMSE'])
    plt.title('Comparación de Modelos por RMSE (Menor es Mejor)', fontsize=16)
    plt.xlabel('Modelo', fontsize=12)
    plt.ylabel('Root Mean Squared Error (RMSE)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    rmse_chart_path = os.path.join(output_dir, 'comparacion_rmse.png')
    plt.savefig(rmse_chart_path, dpi=300)
    plt.show()

    # Gráfico de Barras: Comparación de R²
    plt.figure(figsize=(12, 7))
    sns.barplot(x=results_df.sort_values(by='R²', ascending=False).index, y=results_df.sort_values(by='R²', ascending=False)['R²'])
    plt.title('Comparación de Modelos por R² (Mayor es Mejor)', fontsize=16)
    plt.xlabel('Modelo', fontsize=12)
    plt.ylabel('Coeficiente de Determinación (R²)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    r2_chart_path = os.path.join(output_dir, 'comparacion_r2.png')
    plt.savefig(r2_chart_path, dpi=300)
    plt.show()

    # Gráfico de Líneas: Predicciones del Mejor Modelo vs. Valores Reales
    best_model_name = results_df.index[0] # El mejor modelo es el primero (ordenado por RMSE)
    best_predictions = predictions[best_model_name]

    plt.figure(figsize=(18, 8))
    plt.plot(y_test.index, y_test, label='Valores Reales', color='blue', alpha=0.7)
    plt.plot(y_test.index, best_predictions, label=f'Predicciones de {best_model_name}', color='red', linestyle='--')
    plt.title(f'Predicciones del Mejor Modelo ({best_model_name}) vs. Valores Reales', fontsize=16)
    plt.xlabel('Fecha y Hora', fontsize=12)
    plt.ylabel('Temperatura (°C)', fontsize=12)
    plt.legend()
    plt.grid(True)
    best_model_chart_path = os.path.join(output_dir, 'mejor_modelo_predicciones.png')
    plt.savefig(best_model_chart_path, dpi=300)
    plt.show()

    # --- 4. Imprimir Resumen Final ---
    print("\n" + "="*60)
    print("✅ PROCESO DE COMPARACIÓN FINALIZADO ✅")
    print("="*60)
    print("\n--- Tabla de Resultados (ordenada por el mejor RMSE) ---")
    
    if results_df.empty:
        print("No se pudieron obtener resultados para ningún modelo.")
    else:
        print(results_df)
        print(f"\n🏆 El modelo más adecuado para ClimaSeguro, basado en el menor RMSE, es: {best_model_name}")

# --- PUNTO DE ENTRADA DEL SCRIPT ---
if __name__ == "__main__":
    main()