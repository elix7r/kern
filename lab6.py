import ssl
from collections import Counter

import keras
import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_20newsgroups
from tensorflow.keras import layers

# Отключение проверки SSL
ssl._create_default_https_context = ssl._create_unverified_context

# categories = ["comp.graphics", "sci.space", "rec.sport.baseball"]
# categories = ["comp.windows.x", "misc.forsale", "comp.windows.x", "sci.electronics"]  # 4 вариант
categories = ["rec.autos", "rec.sport.hockey", "sci.crypt", "sci.med", "talk.religion.misc"]  # 5 вариант
# categories = ["comp.sys.ibm.pc.hardware", "rec.motorcycles", "sci.electronics", "alt.atheism"]  # 14 вариант
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)

vocab = Counter()
for text in newsgroups_train.data:
    for word in text.split(' '):
        vocab[word.lower()] += 1
for text in newsgroups_test.data:
    for word in text.split(' '):
        vocab[word.lower()] += 1

total_words = len(vocab)


def get_word_to_index(vocab_):
    _word_to_index = {}
    for i, _word in enumerate(vocab_):
        _word_to_index[_word.lower()] = i
    return _word_to_index


word_to_index = get_word_to_index(vocab)


def get_batch(df, i, batch_size_):
    batches = []
    results = []
    texts = df.data[i * batch_size_:i * batch_size_ + batch_size_]
    _categories = df.target[i * batch_size_:i * batch_size_ + batch_size_]
    for _text in texts:
        layer = np.zeros(total_words, dtype=float)
        for _word in _text.split(' '):
            layer[word_to_index[_word.lower()]] += 1
        batches.append(layer)
    for category in _categories:
        y = np.zeros(3, dtype=float)
        if category == 0:
            y[0] = 1.
        elif category == 1:
            y[1] = 1.
        else:
            y[2] = 1.
        results.append(y)
    return np.array(batches), np.array(results)


# Параметры обучения
learning_rate = 0.01
training_epochs = 10
batch_size = 150
display_step = 1

# Network Parameters
n_hidden_1 = 20  # скрытый слой
n_hidden_2 = 10  # скрытый слой
n_hidden_3 = 5  # добавлен 3-й скрытый слой
n_input = total_words  # количество уникальных слов в наших текстах
n_classes = 3  # 3 класса

# Создание модели
model = keras.models.Sequential()
model.add(keras.Input(shape=(n_input,)))
model.add(layers.Dense(n_hidden_1, activation='relu'))
model.add(layers.Dense(n_hidden_2, activation='relu'))
model.add(layers.Dense(n_hidden_3, activation='relu'))
model.add(layers.Dense(n_classes, activation='softmax'))

# Компиляция модели
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              metrics=['accuracy'])

# Подготовка данных
train_data = tf.data.Dataset.from_tensor_slices(
    (get_batch(newsgroups_train, 0, len(newsgroups_train.data))[0],
     get_batch(newsgroups_train, 0, len(newsgroups_train.data))[1]))

train_data = train_data.batch(batch_size)

# Обучение модели
model.fit(train_data, epochs=training_epochs)

# Тестирование
test_data = tf.data.Dataset.from_tensor_slices(
    (get_batch(newsgroups_test, 0, len(newsgroups_test.data))[0],
     get_batch(newsgroups_test, 0, len(newsgroups_test.data))[1]))

test_data = test_data.batch(batch_size)

# Вычисление точности
test_loss, test_accuracy = model.evaluate(test_data)
print("Точность:", test_accuracy)
