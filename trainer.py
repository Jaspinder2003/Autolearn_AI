import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt



training_images_path = ".\\training_images"
test_images_path = ".\\test_images"
num_classes = 2
BATCH_SIZE = 20

def load_data():
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        training_images_path,
        image_size=(128,128), #resize all images
        batch_size = BATCH_SIZE
    )

    test_dataset = tf.keras.utils.image_dataset_from_directory(
        test_images_path,
        image_size=(128,128),
        batch_size= 10
    )

    return train_dataset, test_dataset


def normalize(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.one_hot(label, depth=num_classes)  # Convert to one-hot
    return image, label


def augment(image, label):
    image = tf.image.random_flip_left_right(image)          # Randomly flip horizontally
    image = tf.image.random_brightness(image, max_delta=0.2) # Adjust brightness randomly
    image = tf.image.random_contrast(image, 0.8, 1.2)       # Adjust contrast randomly
    """This next line will probably mess things up so testing is required"""
    # image = tf.image.random_crop(image, size=[100, 100, 3]) # Randomly crop image
    return image, label

def create_model():
    global num_classes

    model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')  # Adjust num_classes
    ])
    return model


def main():
    #load and preprocess data
    train_data, test_data = load_data()

    #agument the training dataset
    train_data = train_data.map(augment)

    #normalize dataset
    train_data = train_data.map(normalize)
    test_data = test_data.map(normalize)

    #create model
    model = create_model()

    #compile model
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


    #train the model
    history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=10
    )
    
    #evaluate using test images
    test_loss, test_acc = model.evaluate(test_data)
    print(f"Test Accuracy: {test_acc:.2f}")

    #save the model
    model.save('./models/cnn_model.h5')

    #visualize and save progress data
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()

main()