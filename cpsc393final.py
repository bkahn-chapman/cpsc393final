import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd

players = pd.read_csv("allsituations.csv")
players
players["position"].replace({"C" : 0, "R" : 1, "L" : 2, "D" : 3}, inplace=True)

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import svm

X_train, X_test, y_train, y_test = train_test_split(
    players[["I_F_xOnGoal", "I_F_scoreVenueAdjustedxGoals", "I_F_primaryAssists", "I_F_secondaryAssists", "possdiff"]], players.position, test_size=0.25, random_state=489)  # 75% training and 25% test

from keras.utils.np_utils import to_categorical
one_hot_train_labels = to_categorical(y_train, 5)
one_hot_test_labels = to_categorical(y_test, 5)

from tensorflow.keras import models
from tensorflow.keras import layers

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(5,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(5, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

x_val = X_train[:2500]
partial_x_train = X_train[2500:]
y_val = one_hot_train_labels[:2500]
partial_y_train = one_hot_train_labels[2500:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=25,
                    batch_size=512,
                    validation_data=(x_val, y_val))

results = model.evaluate(X_test, one_hot_test_labels)

import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()

history_dict = history.history

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(5,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(5, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(partial_x_train,
          partial_y_train,
          epochs=8,
          batch_size=512,
          validation_data=(x_val, y_val))
results = model.evaluate(X_test, one_hot_test_labels)

print(results)
