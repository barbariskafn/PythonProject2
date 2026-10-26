import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

x_train = tf.constant([3, 4, 6, 9, 13], dtype=tf.float32)
y_train = tf.constant([9, 16, 36, 81, 169], dtype=tf.float32)

model = keras.Sequential([
    layers.Dense(200, activation='relu', input_shape=(1,)),
    layers.Dense(100),
    layers.Dense(1)])
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
model.fit(x_train, y_train, epochs=600, verbose=0)
print("Обучение завершено!")

x_test = tf.constant([6, 8, 11, 12, 13], dtype=tf.float32)
predictions = model.predict(x_test)

print('vyhodnye danye', x_test)
tf.print('protnosi:', predictions.flatten())