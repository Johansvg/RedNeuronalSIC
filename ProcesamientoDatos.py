import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Cargar el conjunto de datos
df = pd.read_csv("csv/data.csv")

# Separar las características (X) y la variable objetivo (y)
features = df.drop("Target", axis=1)
target = df["Target"]

# Normalizar las características numéricas
numeric_features = features.select_dtypes(include=['float64']).columns
scaler = StandardScaler()
features[numeric_features] = scaler.fit_transform(features[numeric_features])

# Codificar la variable objetivo
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(target)

# Agregar las características normalizadas y la variable objetivo al DataFrame
df_processed = pd.concat([features, pd.Series(y, name="Target")], axis=1)

# Guardar los datos procesados en un archivo CSV
df_processed.to_csv("csv/datosProcesados.csv", index=False)