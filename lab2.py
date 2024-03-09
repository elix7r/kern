import tensorflow as tf
import tensorflow_datasets as tfds

# Загрузка датасета MNIST с использованием TensorFlow Datasets
(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

# Определение модели с использованием tf.keras
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Компиляция модели
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Обучение модели
model.fit(ds_train.batch(100), epochs=1000)

# Вычисление точности на тестовом наборе данных
test_loss, test_acc = model.evaluate(ds_test.batch(100))
print("Test Accuracy: {:.4}%".format(test_acc * 100))
