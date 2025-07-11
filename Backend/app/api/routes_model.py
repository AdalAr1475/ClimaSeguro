# app/api/routes_model.py

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from statsmodels.iolib.smpickle import load_pickle
import matplotlib.pyplot as plt
import io
import base64

router = APIRouter()

# Directorio donde se guardaron los modelos
MODEL_DIR = os.path.join('app', 'models', 'entrenamiento')
MODELS = {}
METRICS_DF = None
SCALERS = {}
X_TEST, Y_TEST, Y_TRAIN = None, None, None

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:i+seq_length])
        ys.append(data[i+seq_length])
    return np.array(xs), np.array(ys)

# --- 1. Inicialización de la Aplicación y Carga de Modelos ---
@router.on_event("startup")
def load_models_and_metrics():
    """Carga todos los modelos y métricas al iniciar la API."""
    global METRICS_DF, X_TEST, Y_TEST, Y_TRAIN

    print("--- Cargando modelos y métricas... ---")
    # Cargar métricas
    metrics_path = os.path.join(MODEL_DIR, 'model_metrics.csv')
    if os.path.exists(metrics_path):
        METRICS_DF = pd.read_csv(metrics_path, index_col=0)
        print("✅ Métricas cargadas.")
    else:
        print("⚠️  Archivo de métricas no encontrado.")

    # Cargar modelos
    model_files = {
        "Linear Regression": "Linear_Regression.joblib", "Ridge": "Ridge.joblib",
        "Random Forest Regressor": "Random_Forest_Regressor.joblib", "SVR": "SVR.joblib",
        "XGBoost": "XGBoost.joblib", "LightGBM": "LightGBM.joblib",
        "CatBoost": "CatBoost.joblib", "ARIMA": "ARIMA.pkl", "LSTM": "LSTM.h5"
    }

    for name, filename in model_files.items():
        path = os.path.join(MODEL_DIR, filename)
        if os.path.exists(path):
            try:
                if name == "LSTM":
                    MODELS[name] = load_model(path)
                    # Cargar su escalador
                    scaler_path = os.path.join(MODEL_DIR, 'LSTM_scaler.joblib')
                    SCALERS['LSTM'] = joblib.load(scaler_path)
                elif name == "ARIMA":
                    MODELS[name] = load_pickle(path)
                else:
                    MODELS[name] = joblib.load(path)
                print(f"  -> Modelo '{name}' cargado.")
            except Exception as e:
                print(f"❌ Error cargando '{name}': {e}")

    print("--- Cargando y preparando datos de prueba... ---")
    try:
        data_path = os.path.join('app', 'models', 'eda', 'datos_procesados.csv')
        df = pd.read_csv(data_path, parse_dates=['timestamp'], index_col='timestamp')
        df['hora'] = df.index.hour
        df['dia_semana'] = df.index.dayofweek
        df['mes'] = df.index.month
        
        X = df[['Humedad', 'LdrValorAnalog', 'hora', 'dia_semana', 'mes']]
        y = df['Temperatura']
        # Usamos los mismos parámetros para asegurar consistencia
        X_train, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X, y, test_size=0.2, shuffle=False)
        print("✅ Datos de prueba cargados en memoria.")
    except Exception as e:
        print(f"❌ Error cargando los datos de prueba: {e}")
    
    print("--- Carga completa. API lista. ---")

# --- 2. Definición de los Datos de Entrada (Request Body) ---
class PredictionInput(BaseModel):
    Humedad: float
    LdrValorAnalog: float
    hora: int
    dia_semana: int
    mes: int
    # Para LSTM, necesitamos los últimos 10 valores de temperatura
    historical_temp: list[float] = [] 

# --- 3. Endpoints de la API ---
@router.get("/metrics", summary="Obtener Métricas de Todos los Modelos")
def get_metrics():
    """
    Devuelve un JSON con las métricas (RMSE, MAE, R²) de todos los modelos,
    ordenadas por el mejor RMSE.
    """
    if METRICS_DF is None:
        raise HTTPException(status_code=404, detail="Métricas no encontradas.")
    
    # Ordenar por RMSE y convertir a JSON
    sorted_metrics = METRICS_DF.sort_values(by="RMSE")
    return JSONResponse(content=sorted_metrics.to_dict('index'))

@router.post("/predict", summary="Obtener Predicciones de Todos los Modelos")
def predict_all(input_data: PredictionInput):
    """
    Recibe los datos de entrada y devuelve una predicción de temperatura
    para cada uno de los modelos cargados.
    """
    if not MODELS:
        raise HTTPException(status_code=503, detail="Modelos no están disponibles.")

    predictions = {}
    
    # Preparar datos para modelos estándar
    input_df = pd.DataFrame([input_data.dict(exclude={'historical_temp'})])

    for name, model in MODELS.items():
        try:
            if name == "LSTM":
                if len(input_data.historical_temp) != 10:
                    predictions[name] = "Error: Se requieren 10 valores históricos de temperatura."
                    continue
                scaler = SCALERS['LSTM']
                input_seq = np.array(input_data.historical_temp).reshape(-1, 1)
                scaled_seq = scaler.transform(input_seq)
                pred_scaled = model.predict(np.array([scaled_seq]))
                pred = scaler.inverse_transform(pred_scaled)
                predictions[name] = float(pred[0][0])
            
            elif name == "ARIMA":
                # ARIMA predice el siguiente paso, no usa input directo aquí
                pred = model.forecast(steps=1)
                predictions[name] = float(pred.iloc[0])

            else: # Modelos de Scikit-learn, XGBoost, etc.
                pred = model.predict(input_df)
                predictions[name] = float(pred[0])
        except Exception as e:
            predictions[name] = f"Error en la predicción: {e}"
            
    return predictions

# Para probar este endpoint
# {
#   "Humedad": 85.5,
#   "LdrValorAnalog": 512.0,
#   "hora": 14,
#   "dia_semana": 3,
#   "mes": 7,
#   "historical_temp": [
#     18.5,
#     18.6,
#     18.5,
#     18.7,
#     18.8,
#     19.0,
#     19.1,
#     19.0,
#     18.9,
#     18.8
#   ]
# }

# La opciones del model_name: 
@router.get("/compare/{model_name}", summary="Comparar Predicciones con Reales y Obtener Similitud")
def compare_model(model_name: str):
    """
    Usa el modelo y los datos de prueba YA CARGADOS para generar el gráfico
    de comparación.
    """
    if model_name not in MODELS:
        raise HTTPException(status_code=404, detail=f"Modelo '{model_name}' no encontrado.")
    
    if X_TEST is None or Y_TEST is None or Y_TRAIN is None:
        raise HTTPException(status_code=503, detail="Datos de prueba no están disponibles. Revisa los logs de inicio.")

    model = MODELS[model_name]
    x_test = X_TEST
    y_test = Y_TEST
    
    # Generar predicciones para el conjunto de prueba
    if model_name == "LSTM":
        # --- INICIO DE LA LÓGICA CORRECTA PARA LSTM ---
        SEQ_LENGTH = 10 # Debe ser el mismo valor que en el entrenamiento
        scaler = SCALERS['LSTM']
        
        # 1. Tomar los últimos 10 valores del set de entrenamiento
        last_points_of_train = Y_TRAIN.values[-SEQ_LENGTH:]
        
        # 2. Crear el input para la predicción
        # No necesitamos concatenar con Y_TEST, el script de entrenamiento lo hace de forma más directa.
        # Replicamos la lógica del script de entrenamiento:
        y_for_lstm_test = pd.concat([Y_TRAIN, Y_TEST])[-len(Y_TEST) - SEQ_LENGTH:]

        # 3. Escalar los datos de entrada
        inputs = scaler.transform(y_for_lstm_test.values.reshape(-1, 1))
        
        # 4. Crear las secuencias de prueba
        X_test_lstm, _ = create_sequences(inputs, SEQ_LENGTH)
        
        # 5. Predecir
        lstm_pred_scaled = model.predict(X_test_lstm)
        
        # 6. Des-escalar el resultado para obtener la predicción real
        y_pred = scaler.inverse_transform(lstm_pred_scaled).flatten()
        # --- FIN DE LA LÓGICA CORRECTA PARA LSTM ---
    elif model_name == "ARIMA":
        y_pred = model.forecast(steps=len(y_test))
    else:
        y_pred = model.predict(x_test)

    # Calcular "porcentaje de similitud" (usando 1 - NRMSE como métrica)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    nrmse = rmse / (y_test.max() - y_test.min())
    similarity = (1 - nrmse) * 100

    # Generar gráfico
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_test.index, y_test, label='Valores Reales', color='blue')
    ax.plot(y_test.index, y_pred, label=f'Predicciones de {model_name}', color='red', linestyle='--')
    ax.set_title(f'Comparación: Reales vs. {model_name}\nSimilitud (1-NRMSE): {similarity:.2f}%')
    ax.set_xlabel('Fecha y Hora')
    ax.set_ylabel('Temperatura (°C)')
    ax.legend()
    ax.grid(True)
    
    # Guardar el gráfico en un buffer de memoria
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)

    # 1. Leer los bytes de la imagen del buffer
    image_bytes = buf.read()
    
    # 2. Codificar los bytes en Base64 y convertirlos a un string
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')

    # 3. Crear el diccionario que se devolverá como JSON
    response_data = {
        "similarity_percentage": round(similarity, 2),
        "image_base64": image_base64
    }
    
    return JSONResponse(content=response_data)

# Las opciones para model_name son: 
# - Linear Regression
# - Ridge
# - Random Forest Regressor
# - SVR
# - XGBoost
# - LightGBM
# - CatBoost
# - ARIMA
# - LSTM