from tensorflow import keras;


def getData():
    test_data = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = test_data.load_data()

    print("The shape of train_images is ", train_images.shape)
    print("The shape of train_labels is ", train_labels.shape)
    print("The shape of test_images is ", test_images.shape)
    print("The length of test_labels is ", len(test_labels))

    return (train_images / 255.0, train_labels, test_images / 255.0, test_labels)
