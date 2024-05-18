import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

epochs = 150

# O dataset utilizado foi o Cat and Dog do Kaggle
# https://www.kaggle.com/datasets/tongpython/cat-and-dog
class_names = ["cats", "dogs"]
dataset_dir = './dataset'
height = 128
width = 128
batch_size = 128

# Carrega o dataset de treinamento
training_images = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_dir + "/training",
    labels='inferred',
    class_names=class_names,
    seed=123,
    image_size=(height, width),
    batch_size=batch_size,
    color_mode='grayscale'
)

# Carrega o dataset de validação
validation_images = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_dir + "/validation",
    labels='inferred',
    class_names=class_names,
    seed=123,
    image_size=(height, width),
    batch_size=batch_size,
    color_mode='grayscale'
)

# Visualiza algumas imagens do dataset
plt.figure(figsize=(10,10))
for images, labels in training_images.take(1):  # Pega apenas o primeiro batch
    for i in range(3*5):
        plt.subplot(3,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(tf.squeeze(images[i]), cmap='gray')  # Remove a dimensão do canal para visualização
        plt.xlabel(class_names[labels[i]])
plt.show()

# Rede neural
model = models.Sequential()
model.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=(height, width, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (5, 5), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (5, 5), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (5, 5), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l2=0.001)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(2, activation='softmax'))

# Compila o modelo
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

# Treina o modelo
history = model.fit(
    training_images,
    epochs=epochs,
    validation_data=validation_images
)

# Avalia o modelo
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(validation_images, verbose=2)
print(test_acc)

# Salva o modelo
model.save('modelo.keras')