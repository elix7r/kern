import os
import time

import keras.callbacks
import tensorflow
from tensorflow.keras import layers


def create_model() -> keras.models.Sequential:
    _model = keras.models.Sequential([
        keras.Input(shape=(32, 32, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    _model.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])

    return _model


# Загрузка и нормализация данных CIFAR-10
cifar = keras.datasets.cifar10
(train_images, train_labels), (test_images, test_labels) = cifar.load_data()

# Нормализация изображений
train_images, test_images = train_images / 255.0, test_images / 255.0

# Create a basic model instance
model = create_model()

# Display the model's architecture
model.summary()

# Настройка TensorBoard
log_dir = "logs/fit/" + time.strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Обучение модели с сохранением данных для TensorBoard
start_time = time.time()
model.fit(train_images, train_labels, epochs=5, batch_size=50,
          validation_data=(test_images, test_labels), callbacks=[tensorboard_callback])

# Оценка модели
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)
print('Time taken:', time.time() - start_time)

# Сохранение модели
# model.save('my_model.keras')
# print("Model saved")

# Создание объекта Checkpoint
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tensorflow.train.Checkpoint(model=model)

# Сохранение модели
checkpoint.save(file_prefix=checkpoint_prefix)
print("Модель сохранена в формате .ckpt")
