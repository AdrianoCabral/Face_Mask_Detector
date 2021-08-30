# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D,Dropout,Flatten, Dense, Input,Conv2D,MaxPooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array,load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os


INIT_LR = 1e-4
EPOCHS = 20
BS = 32

print("[INFO] loading images...")
imagePaths = list(paths.list_images('./dataset'))
data = []
labels = []

# loop para ler o dataset
for imagePath in imagePaths:
    label = imagePath.split(os.path.sep)[-2]
    # setando as dimensões da imagem para 224,224 para serem compativeis com o modelo mobilenet
    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)
    data.append(image)
    labels.append(label)
data = np.array(data, dtype="float32")
labels = np.array(labels)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

#train/test/val split
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.30, stratify=labels, random_state=39)
print(len(testX))
print(len(testY))
(testX, valX, testY, valY) = train_test_split(testX, testY,
	test_size=0.50, stratify=testY, random_state=42)

#data augmentation
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

#estrutura de rede do nosso modelo 1
model =Sequential([
    Conv2D(100, (3,3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2,2),
    
    Conv2D(100, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),

    Dense(2, activation='softmax')
])
optim = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(optimizer=optim, loss='binary_crossentropy', metrics=['acc'])

#treinamento
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(valX, valY),
	validation_steps=len(valX) // BS,
	epochs=EPOCHS)

#teste
predIdxs = model.predict(testX, batch_size=BS)
predIdxs = np.argmax(predIdxs, axis=1)

print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

print("[INFO] saving mask detector model...")
model.save('./mask_detector.model', save_format="h5")

#fine tuning pipeline

#lendo modelo da mobilenetV2 treinado com o imagenet,removendo as camadas fully connected
baseModel = MobileNetV2(weights="imagenet", include_top=False,
    input_tensor=Input(shape=(224, 224, 3)))

#adicionando nossas camadas fully connected
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

fine_tuned_model = Model(inputs=baseModel.input, outputs=headModel)

#congelando as camadas escondidas
for layer in baseModel.layers:
    layer.trainable= False

fine_tuned_model.compile(optimizer=optim, loss='binary_crossentropy', metrics=['acc'])

#treinamento
fine_tuned_H = fine_tuned_model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(valX, valY),
	validation_steps=len(valX) // BS,
	epochs=EPOCHS)
#teste
predIdxs = fine_tuned_model.predict(testX, batch_size=BS)
predIdxs = np.argmax(predIdxs, axis=1)

print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

print("[INFO] saving fine tuned mask detector model...")
fine_tuned_model.save('./fine_tuned_mask_detector.model', save_format="h5")