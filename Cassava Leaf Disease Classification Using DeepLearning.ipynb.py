import zipfile
zip_ref = zipfile.ZipFile('/content/Cassavaleaf.zip', 'r')
zip_ref.extractall('/content')
zip_ref.close()

import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,BatchNormalization,Dropout
train_ds = keras.utils.image_dataset_from_directory(
                                                    directory = '/content/train',
                                                    labels='inferred',
                                                    label_mode = 'int',
                                                    batch_size=32,
                                                    image_size=(256,256)
                                                  )

validation_ds = keras.utils.image_dataset_from_directory(
                                                    directory = '/content/test',
                                                    labels='inferred',
                                                    label_mode = 'int',
                                                    batch_size=32,
                                                    image_size=(256,256)
                                                  )

def process(image,label):
  '''image = tf.cast(image/255. ,tf.float32)
  return image,label'''
  one_hot_label = tf.one_hot(label, depth=5)
  return image, one_hot_label
train_ds = train_ds.map(process)
validation_ds = validation_ds.map(process)

# VGG-16
model = Sequential([
    Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(256,256,3)),
    Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(256,256,3)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(64,kernel_size=(3,3),activation='relu'),
    Conv2D(64,kernel_size=(3,3),activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(128,kernel_size=(3,3),activation='relu'),
    Conv2D(128,kernel_size=(3,3),activation='relu'),
    Conv2D(128,kernel_size=(3,3),activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(128,kernel_size=(3,3),activation='relu'),
    Conv2D(128,kernel_size=(3,3),activation='relu'),
    Conv2D(128,kernel_size=(3,3),activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(128,kernel_size=(3,3),activation='relu'),
    Conv2D(128,kernel_size=(3,3),activation='relu'),
    Conv2D(128,kernel_size=(3,3),activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),

    Flatten(),
    Dense(128,activation='relu'),
    Dropout(0.4),
    Dense(64,activation='relu'),
    Dropout(0.5),
    Dense(5,activation='softmax')
])

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

history = model.fit(train_ds,epochs=150,validation_data=validation_ds)

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'],color='red',label='train')
plt.plot(history.history['val_accuracy'],color='blue',label='validation')
plt.legend()
plt.show()

plt.plot(history.history['loss'],color='red',label='train')
plt.plot(history.history['val_loss'],color='blue',label='validation')
plt.legend()
plt.show()

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Assume test dataset as 'validation_ds'
y_true = []
y_pred = []

for images, labels in validation_ds:
  predictions = model.predict(images)
  y_pred.extend(np.argmax(predictions, axis=1))
  y_true.extend(np.argmax(labels, axis=1))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
print(classification_report(y_true, y_pred))

# Alex Net
model = Sequential([
    Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(256,256,3)),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(64,kernel_size=(3,3),activation='relu'),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(128,kernel_size=(3,3),activation='relu'),
    Conv2D(128,kernel_size=(3,3),activation='relu'),
    Conv2D(128,kernel_size=(3,3),activation='relu'),
    MaxPooling2D(pool_size=(2,2)),

    Flatten(),
    Dense(128,activation='relu'),
    Dropout(0.4),
    Dense(64,activation='relu'),
    Dropout(0.5),
    Dense(5,activation='softmax')
])

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
history = model.fit(train_ds,epochs=50,validation_data=validation_ds)

y_true = []
y_pred = []
for images, labels in validation_ds:
  predictions = model.predict(images)
  y_pred.extend(np.argmax(predictions, axis=1))
  y_true.extend(np.argmax(labels, axis=1))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'],color='red',label='train')
plt.plot(history.history['val_accuracy'],color='blue',label='validation')
plt.legend()
plt.show()

# LeNet-5
model = Sequential([
    Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(256,256,3)),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(64,kernel_size=(3,3),activation='relu'),
    MaxPooling2D(pool_size=(2,2)),

    Flatten(),
    Dense(128,activation='relu'),
    Dropout(0.4),
    Dense(64,activation='relu'),
    Dropout(0.5),
    Dense(5,activation='softmax')
])

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
history = model.fit(train_ds,epochs=50,validation_data=validation_ds)

y_true = []
y_pred = []
for images, labels in validation_ds:
  predictions = model.predict(images)
  y_pred.extend(np.argmax(predictions, axis=1))
  y_true.extend(np.argmax(labels, axis=1))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'],color='red',label='train')
plt.plot(history.history['val_accuracy'],color='blue',label='validation')
plt.legend()
plt.show()

#ZFNet
model = Sequential([
    Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(256,256,3)),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(64,kernel_size=(3,3),activation='relu'),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(128,kernel_size=(3,3),activation='relu'),
    Conv2D(128,kernel_size=(3,3),activation='relu'),
    Conv2D(128,kernel_size=(3,3),activation='relu'),
    MaxPooling2D(pool_size=(2,2)),

    Flatten(),
    Dense(128,activation='relu'),
    Dropout(0.4),
    Dense(64,activation='relu'),
    Dropout(0.5),
    Dense(5,activation='softmax')
])

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
history = model.fit(train_ds,epochs=25,validation_data=validation_ds)

y_true = []
y_pred = []
for images, labels in validation_ds:
  predictions = model.predict(images)
  y_pred.extend(np.argmax(predictions, axis=1))
  y_true.extend(np.argmax(labels, axis=1))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
print(classification_report(y_true, y_pred))

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'],color='red',label='train')
plt.plot(history.history['val_accuracy'],color='blue',label='validation')
plt.legend()
plt.show()

#Validation on new image
import cv2
import numpy as np
import matplotlib.pyplot as plt

test_img = cv2.imread('/content/images.jpeg')
plt.imshow(test_img)
test_img.shape
test_img = cv2.resize(test_img,(256,256))
test_input = test_img.reshape((1,256,256,3))
m = model.predict(test_input)
np.argmax(m)