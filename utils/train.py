import tensorflow as tf;
from tensorflow import keras;
import numpy as np;
from utils.read import getData;
import matplotlib.pyplot as plt

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def trainModel():
    def showPlt():
        predictions = model.predict(test_images)
        print("The first picture's prediction is:{},so the result is:{}".format(predictions[0],
                                                                                np.argmax(predictions[0])))
        print("The first picture is ", test_labels[0])
        plt.figure(figsize=(10, 10))
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid('off')
            plt.imshow(test_images[i], cmap=plt.cm.binary)
            predicted_label = np.argmax(predictions[i])
            true_label = test_labels[i]
            if predicted_label == true_label:
                color = 'green'
            else:
                color = 'red'
            plt.xlabel("{} ({})".format(class_names[predicted_label],
                                        class_names[true_label]),
                       color=color)
        plt.show()

    (train_images, train_labels, test_images, test_labels) = getData()

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=10)

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('Test Acc:', test_acc)

    showPlt()

    return model
