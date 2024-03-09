import time

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
    # layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    # layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    # layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    # layers.Dropout(0.5),  # Добавление Dropout для регуляризации
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
