import numpy as np
from keras import backend as K
import random
from utils.read import getData
from utils.train import trainModel
# from skimage.measure import compare_ssim as ssim

e = 0.007
(i_train, l_train, i_test, l_test) = getData()
model = trainModel(i_train, l_train, i_test, l_test, 5)
attack_input_layer = model.layers[0].input
attack_output_layer = model.layers[-1].output
# pre_predicts = None


def attackImages(images):
    attack_images = np.copy(images)
    # num = len(images)
    # global pre_predicts
    # pre_predicts = model.predict(images)
    for index, image in enumerate(images):
        attack_images[index] = attackImage(image, index)
    # imgs = np.copy(attack_images);
    # ssim_sum = 0.0
    # count = 0
    # predicts = model.predict(imgs)
    # for i in range(num):
    #     ssim_one = ssim(imgs[i], i_test[i])
    #     ssim_sum += ssim_one
    #     if np.argmax(predicts[i]) != l_test[i]:
    #         count += 1
    #     #     print("Img[%d] Attack succeed, ssim: %f" % (i, ssim_one))
    #     # else:
    #     #     print("Img[%d] Attack failed, ssim: %f" % (i, ssim_one))
    # ssim_sum /= num
    # ratio = count / num
    # print(ssim_sum)
    # print(ratio)
    return attack_images


def attackImage(image, index):
    # pre = image.copy()
    img = np.expand_dims(image, 0)
    fake_img = random.randint(0, 9)
    # fake_img = (index + 1) % 10
    # predict_label = np.argmax(pre_predicts[index])
    # fake_img = (predict_label + 1) % 10
    # print("p : %f f: %f" % (predict_label, fake_img))

    cost_function = attack_output_layer[0, fake_img]
    gradient_function = K.gradients(cost_function, attack_input_layer)[0]
    grab_function = K.function([attack_input_layer, K.learning_phase()], [cost_function, gradient_function])

    cost = 0.0
    while cost < 0.60:
        cost, gradients = grab_function([img, 0])
        n = np.sign(gradients)
        img += n * e
        img = np.clip(img, -1.0, 1.0)

    result = img[0]
    print("Img[%d] attacked" % index)
    return result
