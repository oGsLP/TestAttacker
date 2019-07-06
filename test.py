from main import aiTest;
from utils.read import getData;
from skimage.measure import compare_ssim as ssim
from utils.train import trainModel

import numpy as np

(i_train, l_train, i_test, l_test) = getData()

# print(i_test.shape)
# print(np.expand_dims(i_test, 0).shape)
# print(np.expand_dims(i_test, 1).shape)
# print(np.expand_dims(i_test, -1).shape)
# print(np.expand_dims(i_test, -1).squeeze().shape)


num = 100
model = trainModel(i_train, l_train, i_test, l_test, 5)
imgs = aiTest(np.expand_dims(i_test[0:100],-1), None).squeeze()/255.0;

ssim_sum = 0.0
count = 0
predicts = model.predict(imgs)
for i in range(num):
    ssim_one = ssim(imgs[i], i_test[i])
    ssim_sum += ssim_one
    if np.argmax(predicts[i]) != l_test[i]:
        count += 1
    #     print("Img[%d] Attack succeed, ssim: %f" %(i , ssim_one))
    # else:
    #     print("Img[%d] Attack failed, ssim: %f" % (i , ssim_one))
ssim_sum /= num
ratio = count / num
print(ssim_sum)
print(ratio)
