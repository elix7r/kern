import os

import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Input

# Загрузка и нормализация данных CIFAR-10
cifar = datasets.cifar10
(train_images, train_labels), (test_images, test_labels) = cifar.load_data()

# Нормализация изображений
train_images, test_images = train_images / 255.0, test_images / 255.0

# Создание модели
model = models.Sequential([
    Input(shape=(32, 32, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Компиляция модели
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Загрузка весов из файла .ckpt
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(model=model)

# Загрузка последнего сохраненного состояния
status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# Проверка статуса загрузки
if status.expect_partial():
    print("Загружено частично")
else:
    print("Загружено полностью")

# Оценка модели
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)
