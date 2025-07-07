# Estructura del proyecto
```plaintext
Backend/
│
├── app/
│   ├── main.py                  # Punto de entrada de la API
│   ├── models/                  # Modelos ML entrenados (.pkl)
│   │   └── modelo_xgb.pkl
│   ├── data/                    # Dataset CSV con datos históricos
│   │   └── clima_huancavelica.csv
│   ├── core/                    # Configuración y autenticación
│   │   ├── config.py
│   │   └── auth.py
│   ├── api/                     # Endpoints de la API REST
│   │   ├── routes_data.py
│   │   └── routes_model.py
│   ├── services/                # Lógica de predicción y evaluación
│   │   ├── predictor.py
│   │   └── evaluator.py
│   ├── utils/                   # Funciones auxiliares
│   │   └── file_loader.py
│   └── logs/                    # Registros (opcional)
│       └── api.log
│
├── requirements.txt
├── README.md
```

# Notas adicionales:
1. De ser necesario, instalar las librerías desde Backend/ con:
```shell
pip install requirements.txt
```

2. Iniciar uvicorn desde Backend/ con 
```shell
uvicorn app.main:app --reload
```
