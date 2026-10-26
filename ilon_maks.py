import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


x_train_weight = tf.constant([5, 10, 15, 20, 25], dtype=tf.float32)
x_train_distance = tf.constant([50, 100, 150, 200, 250], dtype=tf.float32)
y_train_fuel = tf.constant([500, 1000, 1500, 2000, 2500], dtype=tf.float32)

x_train = tf.stack([x_train_weight, x_train_distance], axis=1)  # Объединяем данные в один массив


model = keras.Sequential([
    layers.Dense(20, activation='relu', input_shape=(2,)),  # Скрытый слой с 10 нейронами и функцией активации ReLU
    layers.Dense(10, activation='relu'),
    layers.Dense(5, activation='relu'),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

model.fit(x_train, y_train_fuel, epochs=500, verbose=0)
print('obuchenie zaverscheno')

x_test_weight = tf.constant([8, 12, 18], dtype=tf.float32)
x_test_distance = tf.constant([60, 120, 180], dtype=tf.float32)
x_test = tf.stack([x_test_weight, x_test_distance], axis=1)
3
predictions = model.predict(x_test)
print('Toplivo:', predictions.flatten())