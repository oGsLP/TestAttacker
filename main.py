import tensorflow as tf


def aiTest(images, shape):
    print(len(images))
    hello = tf.constant('Hello, TensorFlow!')
    session = tf.Session()

    generate_images = session.run(hello)
    return generate_images


# print(aiTest([], []))
