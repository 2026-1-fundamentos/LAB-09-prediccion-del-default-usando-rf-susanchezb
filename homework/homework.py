# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import gzip
import json
import os
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix


def clean_campaign_data():
    """
    Paso 1: Limpiar los datasets
    """
    # Load data
    train_data = pd.read_csv("files/input/train_data.csv.zip")
    test_data = pd.read_csv("files/input/test_data.csv.zip")

    for idx, data in enumerate([train_data, test_data]):
        # Rename target column
        data = data.rename(columns={"default payment next month": "default"})

        # Remove ID column
        data = data.drop("ID", axis=1)

        # Remove rows with missing values (N/A in MARRIAGE, EDUCATION)
        data = data[(data["MARRIAGE"] != 0) & (data["EDUCATION"] != 0)]

        # Group EDUCATION values > 4 as "others" (4 -> 4)
        data = data.copy()
        data.loc[data["EDUCATION"] > 4, "EDUCATION"] = 4

        if idx == 0:
            train_data = data
        else:
            test_data = data

    return train_data, test_data


def build_pipeline():
    """
    Paso 3: Create pipeline with OneHotEncoder and RandomForest
    """
    # Define categorical and numerical columns
    categorical_cols = ["SEX", "EDUCATION", "MARRIAGE"]
    numerical_cols = [col for col in ["LIMIT_BAL", "AGE", "PAY_0", "PAY_2", "PAY_3",
                                       "PAY_4", "PAY_5", "PAY_6", "BILL_AMT1", "BILL_AMT2",
                                       "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
                                       "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4",
                                       "PAY_AMT5", "PAY_AMT6"]]

    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(sparse_output=False), categorical_cols),
            ("num", StandardScaler(), numerical_cols)
        ])

    # Create pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42))
    ])

    return pipeline


def optimize_hyperparameters(pipeline, x_train, y_train):
    """
    Paso 4: Optimize hyperparameters using GridSearchCV with balanced_accuracy
    """
    param_grid = {
        "classifier__n_estimators": [150, 250],
        "classifier__max_depth": [18, 22],
        "classifier__min_samples_split": [2, 4]
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=10,
        scoring="balanced_accuracy",
        n_jobs=1
    )

    grid_search.fit(x_train, y_train)
    return grid_search


def save_model(model, filename="files/models/model.pkl.gz"):
    """
    Paso 5: Save model compressed with gzip
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with gzip.open(filename, "wb") as f:
        pickle.dump(model, f)


def calculate_metrics(model, x_train, y_train, x_test, y_test):
    """
    Paso 6 y 7: Calculate metrics and confusion matrix, save to JSON
    """
    metrics_list = []

    # Predictions
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # Train metrics
    train_metrics = {
        "type": "metrics",
        "dataset": "train",
        "precision": precision_score(y_train, y_train_pred),
        "balanced_accuracy": balanced_accuracy_score(y_train, y_train_pred),
        "recall": recall_score(y_train, y_train_pred),
        "f1_score": f1_score(y_train, y_train_pred)
    }
    metrics_list.append(train_metrics)

    # Test metrics
    test_metrics = {
        "type": "metrics",
        "dataset": "test",
        "precision": precision_score(y_test, y_test_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_test_pred),
        "recall": recall_score(y_test, y_test_pred),
        "f1_score": f1_score(y_test, y_test_pred)
    }
    metrics_list.append(test_metrics)

    # Confusion matrices
    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)

    train_cm = {
        "type": "cm_matrix",
        "dataset": "train",
        "true_0": {"predicted_0": int(cm_train[0, 0]), "predicted_1": None},
        "true_1": {"predicted_0": None, "predicted_1": int(cm_train[1, 1])}
    }
    metrics_list.append(train_cm)

    test_cm = {
        "type": "cm_matrix",
        "dataset": "test",
        "true_0": {"predicted_0": int(cm_test[0, 0]), "predicted_1": None},
        "true_1": {"predicted_0": None, "predicted_1": int(cm_test[1, 1])}
    }
    metrics_list.append(test_cm)

    # Save to JSON
    os.makedirs("files/output", exist_ok=True)
    with open("files/output/metrics.json", "w") as f:
        for metric in metrics_list:
            f.write(json.dumps(metric) + "\n")


def run():
    """Execute the full pipeline"""
    # Clean data
    train_data, test_data = clean_campaign_data()

    # Split into X and y
    x_train = train_data.drop("default", axis=1)
    y_train = train_data["default"]
    x_test = test_data.drop("default", axis=1)
    y_test = test_data["default"]

    # Build and optimize pipeline
    pipeline = build_pipeline()
    model = optimize_hyperparameters(pipeline, x_train, y_train)

    # Save model
    save_model(model)

    # Calculate and save metrics
    calculate_metrics(model, x_train, y_train, x_test, y_test)


# Execute on import
try:
    run()
except Exception as e:
    import traceback
    print(f"Error: {e}")
    traceback.print_exc()
