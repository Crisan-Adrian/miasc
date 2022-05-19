import matplotlib.pyplot as plt
import numpy as np
import skimage.io as skio
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')


def gen_image_predict(image, radius):
    for i_ in range(radius, len(image) - radius + 1):
        for j_ in range(radius, len(image[0]) - radius + 1):
            # if the patch center coordinates have the value 1 in the mask array select that position as a test sample
            arr = np.array([z[j_ - radius:j_ + radius + 1] for z in image[i_ - radius:i_ + radius + 1]])
            arr = np.expand_dims(arr, axis=0)
            arr = np.expand_dims(arr, axis=-1)
            # print("SHAPE")
            # print(arr.shape)
            # print(arr)
            yield arr


RADIUS = 6
SHAPE = RADIUS * 2 + 1
CLASSES = 8

model = tf.keras.models.load_model('saved_model/my_model_2')
print(model.summary())

imstack_test = skio.imread("Data/Downscaled2/2010EU_d.tif")

plt.imshow(imstack_test)
plt.show()


imstack_t = skio.imread("Data/Downscaled2/2013EU_d.tif")
predicted = imstack_test.copy()
values = model.predict(gen_image_predict(imstack_test, RADIUS), verbose=1)

print(values.shape)
print(predicted.shape)
values = values.tolist()
print(len(values))
print(len(predicted) - 2 * RADIUS)
print(len(predicted[0]) - 2 * RADIUS)
for i in range(RADIUS, len(predicted) - RADIUS + 1):
    for j in range(RADIUS, len(predicted[0]) - RADIUS + 1):
        values_i_j = values.pop(0)
        index_min = np.argmax(values_i_j)
        predicted[i][j] = index_min * CLASSES

for i in range(len(imstack_t)):
    for j in range(len(imstack_t[0])):
        imstack_t[i][j] = (imstack_t[i][j] // 8) * 8
#
plt.imshow(imstack_t)
plt.show()
skio.imsave("Data/Results/2013EU_true2.png", imstack_t, check_contrast=False)

plt.imshow(predicted)
plt.show()
skio.imsave("Data/Results/2013EU_pred2.png", predicted, check_contrast=False)
