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
from statsmodels.tsa.base.wrapper import ResultsWrapper
from prophet import Prophet
import matplotlib.pyplot as plt
import io
import base64

router = APIRouter()

# Directorio donde se guardaron los modelos
MODEL_DIR = os.path.join('Backend', 'app', 'models', 'entrenamiento')
MODELS = {}
METRICS_DF = None
SCALERS = {}

# --- 1. Inicialización de la Aplicación y Carga de Modelos ---
@router.on_event("startup")
def load_models_and_metrics():
    """Carga todos los modelos y métricas al iniciar la API."""
    global METRICS_DF
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
                    MODELS[name] = ResultsWrapper.load(path)
                else:
                    MODELS[name] = joblib.load(path)
                print(f"  -> Modelo '{name}' cargado.")
            except Exception as e:
                print(f"❌ Error cargando '{name}': {e}")
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

# La opciones del model_name: 
@router.get("/compare/{model_name}", summary="Comparar Predicciones con Reales y Obtener Similitud")
def compare_model(model_name: str):
    """
    Genera un gráfico comparando los valores reales con las predicciones
    de un modelo específico y devuelve tanto el gráfico (en Base64) como
    el porcentaje de similitud.
    """
    if model_name not in MODELS:
        raise HTTPException(status_code=404, detail=f"Modelo '{model_name}' no encontrado.")

    # Cargar datos de prueba (esto es para demostración, en un caso real los recibirías)
    data_path = os.path.join('Backend', 'app', 'models', 'eda', 'datos_procesados.csv')
    df = pd.read_csv(data_path, parse_dates=['timestamp'], index_col='timestamp')
    df['hora'] = df.index.hour
    df['dia_semana'] = df.index.dayofweek
    df['mes'] = df.index.month
    
    # Simular la división de datos para obtener el conjunto de prueba
    X = df[['Humedad', 'LdrValorAnalog', 'hora', 'dia_semana', 'mes']]
    y = df['Temperatura']
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = MODELS[model_name]
    
    # Generar predicciones para el conjunto de prueba
    if model_name == "LSTM":
        # Lógica compleja para predecir toda la secuencia de prueba con LSTM
        # (Simplificado para la demo, puede ser lento)
        y_pred = []
        temp_hist = list(y.iloc[:-len(y_test)][-10:])
        for i in range(len(y_test)):
            pred = predict_all(PredictionInput(
                Humedad=X_test.iloc[i]['Humedad'], LdrValorAnalog=X_test.iloc[i]['LdrValorAnalog'],
                hora=X_test.iloc[i]['hora'], dia_semana=X_test.iloc[i]['dia_semana'], mes=X_test.iloc[i]['mes'],
                historical_temp=temp_hist
            ))['LSTM']
            y_pred.append(pred)
            temp_hist = temp_hist[1:] + [pred]
    elif model_name == "ARIMA":
        y_pred = model.forecast(steps=len(y_test))
    else:
        y_pred = model.predict(X_test)

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

    # Devolver el gráfico como una imagen
    return StreamingResponse(buf, media_type="image/png")