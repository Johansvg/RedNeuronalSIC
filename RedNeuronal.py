import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
import xgboost as xgb

# Cargar el conjunto de datos
df = pd.read_csv("csv/datosProcesados.csv")

# Separar las características y la variable objetivo
x = df.drop("Target", axis=1)
y = df["Target"]

# Aplicar SMOTE / No se está aplicando SMOTE porque no se está obteniendo una buena precisión
smote = SMOTE(random_state=42)
x_resampled, y_resampled = smote.fit_resample(x, y)

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=8)
# X_train, X_test, y_train, y_test = train_test_split(x_resampled, y_resampled, test_size=0.2, random_state=8)


# Crear un modelo simple
model = keras.Sequential([
    tf.keras.layers.Dense(36, activation=tf.keras.layers.LeakyReLU(alpha=0.005), input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(72, activation=tf.keras.layers.ELU(alpha=1.0)),
    tf.keras.layers.Dense(3, activation='softmax')  # 3 salidas: Dropout, Enrolled, Graduate
])
'''
activation=tf.keras.layers.LeakyReLU(alpha=0.005)
activation='swish'
activation=tf.keras.layers.ELU(alpha=1.0)
'''
# Compilar el modelo
custom_optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=custom_optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# imprimir la precisión del modelo
accuracy = history.history['accuracy'][-1]
print(f'Accuracy después de entrenar: {accuracy * 100:.2f}%')

# Guardar el modelo
model.save("modelo_entrenado.h5")

# Modelo utilizando XGBoost
'''
model = xgb.XGBClassifier(objective='multi:softmax', num_class=3, random_state=8)

# Entrenar el modelo
model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Calcular la precisión
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
'''