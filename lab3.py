import tensorflow as tf
from tensorflow.keras import layers, models, Input
import time

# Загрузка данных
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Нормализация изображений
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# Нормализация метки класса
train_images, test_images = train_images / 255.0, test_images / 255.0

# Создание модели
model = models.Sequential([
    Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (5, 5), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (5, 5), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# Компиляция модели
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Обучение модели
start_time = time.time()
model.fit(train_images, train_labels, epochs=5, batch_size=50, validation_data=(test_images, test_labels))

# Оценка модели
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)
print('Time taken:', time.time() - start_time)
