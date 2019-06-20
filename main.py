import tensorflow as tf


def aiTest(images, shape):
    hello = tf.constant('Hello, TensorFlow!')
    session = tf.Session()
    generate_images = session.run(hello)
    return generate_images


print(aiTest([], []))
