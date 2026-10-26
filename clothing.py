import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input, Dropout

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train[..., None]
x_test = x_test[..., None]

# Сравнение с исходной полносвязной моделью
baseline = Sequential([
    Input(shape=(28, 28)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
baseline.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
baseline.fit(x_train[..., 0], y_train, epochs=5, batch_size=32, validation_split=0.1, verbose=2)
_, baseline_accuracy = baseline.evaluate(x_test[..., 0], y_test, verbose=0)
print(f'Исходная модель — точность на тесте: {baseline_accuracy:.4f}')

# Улучшенная CNN: дополнительные Conv2D + MaxPooling2D и более широкие Dense-слои
model = Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.1, verbose=2)
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f'Улучшенная модель — test loss: {test_loss:.4f}, test accuracy: {test_accuracy:.4f}')
print(f'Изменение точности: {test_accuracy - baseline_accuracy:+.4f}')
model.save('fashion_mnist_model.h5')
