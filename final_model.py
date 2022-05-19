import skimage.io as skio
import matplotlib.pyplot as plt
import numpy as np
import sys
import tensorflow as tf
from keras import layers, models, mixed_precision

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print("GPUs Available: ", tf.config.experimental.list_physical_devices('GPU'))

tf.config.set_visible_devices([], 'GPU')

TRAIN = False
RADIUS = 6
SHAPE = RADIUS * 2 + 1
CLASSES = 8

# policy = mixed_precision.policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Generator that splits image into patches of radius pixels around a center pixel from image1 and returns a tuple (patch, center)
# where center is the pixel with the same coordinates as the patch center, taken from image2;
# This generator is used for training
def gen_image(image1, image2, radius, mask):
    i = radius
    j = radius
    while i + radius + 1 < len(image1):
        while j + radius + 1 < len(image1[0]):
            # if the patch center coordinates have the value 0 in the mask array select that position as a train sample
            if mask[i + 1][j + 1] == 0:
                arr = np.array([z[j - radius:j + radius + 1] for z in image1[i - radius:i + radius + 1]])
                arr = np.expand_dims(arr, axis=0)
                arr = np.expand_dims(arr, axis=-1)
                label = np.array(image2[i + 1][j + 1] // CLASSES)
                label = np.expand_dims(label, axis=0)
                label = np.expand_dims(label, axis=-1)
                yield arr, label
            j += 1
        i += 1
        j = radius

# Generator that splits image into patches of radius pixels around a center pixel from image1 and returns a tuple (patch, center)
# where center is the pixel with the same coordinates as the patch center, taken from image2;
# This generator is used for testing
def gen_image_test(image1, image2, radius, mask):
    i = radius
    j = radius
    while i + radius < len(image1):
        while j + radius < len(image1[0]):
            # if the patch center coordinates have the value 1 in the mask array select that position as a test sample
            if mask[i + 1][j + 1] == 1:
                arr = np.array([z[j - radius:j + radius + 1] for z in image1[i - radius:i + radius + 1]])
                arr = np.expand_dims(arr, axis=0)
                arr = np.expand_dims(arr, axis=-1)
                label = np.array(image2[i + 1][j + 1] // CLASSES)
                label = np.expand_dims(label, axis=0)
                label = np.expand_dims(label, axis=-1)
                yield arr, label
            j += 1
        i += 1
        j = radius


def gen_image_predict(image, radius):
    for i in range(radius, len(image) - radius + 1):
        for j in range(radius, len(image[0]) - radius + 1):
            # if the patch center coordinates have the value 1 in the mask array select that position as a test sample
            arr = np.array([z[j - radius:j + radius + 1] for z in image[i - radius:i + radius + 1]])
            arr = np.expand_dims(arr, axis=0)
            arr = np.expand_dims(arr, axis=-1)
            yield arr


# Generate a mask that splits the image pixel array into test and train
def gen_mask(image, split=0.8, ignore=0.5):
    image_mask = np.random.rand(len(image), len(image[0]))
    image_mask = np.where(image_mask <= ignore, -1, image_mask)
    image_mask = np.where(np.logical_and((image_mask - ignore) / (1 - ignore) <= split, image_mask != -1), 0,
                          image_mask)
    image_mask = np.where(np.logical_and((image_mask - ignore) / (1 - ignore) > split, image_mask != -1), 1, image_mask)
    return image_mask


# Read both images
imstack1 = skio.imread("Data/Downscaled2/2011EU_d.tif", plugin="tifffile")
imstack2 = skio.imread("Data/Downscaled2/2012EU_d.tif", plugin="tifffile")
imstack3 = skio.imread("Data/Downscaled2/2013EU_d.tif", plugin="tifffile")

imstack_test = skio.imread("Data/Downscaled2/2011EU_d.tif", plugin="tifffile")

# Transform images in numpy arrays
imstack1 = np.array(imstack1)
imstack2 = np.array(imstack2)

# Create mask and train/test data generators
split_mask_train = gen_mask(imstack1, ignore=0, split=1)
split_mask_test = gen_mask(imstack1, ignore=0, split=0)

train_x = gen_image(imstack1, imstack2, RADIUS, split_mask_train)
# train_y = gen_label(imstack2, 15, split_mask)

test_x = gen_image_test(imstack2, imstack3, RADIUS, split_mask_test)
# test_y = gen_label_test(imstack2, 15, split_mask)

# for x in train_x:
#     if x[0].shape != (31, 31, 1):
#         break


print("Done preprocessing")

# Create the model

n = len(imstack1)
m = len(imstack1[0])

print(n, m)

model = None

# Train the model, input is a 31x31 array of grayscale values, label is an integer from 0 63
if TRAIN:
    model = models.Sequential()
    model.add(layers.Conv2D(filters=16, kernel_size=3, activation='relu', input_shape=(SHAPE, SHAPE, 1)))
    model.add(layers.Conv2D(32, 2, activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, 2, activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, 2, activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(16))
    model.add(layers.Dense(CLASSES))
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=tf.optimizers.Adam(),
                  metrics=['accuracy'])

    print(model.summary())
    model.fit(train_x, epochs=1, verbose=1, batch_size=512)
    model.save('saved_model/my_model3')

else:
    model = tf.keras.models.load_model('saved_model/my_model3')
    print(model.summary())
# model.evaluate(test_x, verbose=2)

predicted = imstack_test.copy()
values = model.predict(gen_image_predict(imstack_test, RADIUS), batch_size=100)

print(values.shape)
values = values.tolist()
for i in range(RADIUS, len(predicted) - RADIUS + 1):
    for j in range(RADIUS, len(predicted[0]) - RADIUS + 1):
        values_i_j = values.pop(0)
        index_min = np.argmax(values_i_j)
        predicted[i][j] = index_min * CLASSES

# plt.imshow(imstack2)
# plt.show()

# plt.imshow(predicted)
# plt.show()
skio.imsave("Predicted/root.png", imstack_test, check_contrast=False)
skio.imsave("Predicted/predicted.png", predicted, check_contrast=False)

