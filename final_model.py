import matplotlib.pyplot as plt
import numpy as np
import skimage.io as skio
import tensorflow as tf
from keras import layers, models

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print("GPUs Available: ", tf.config.experimental.list_physical_devices('GPU'))

tf.config.set_visible_devices([], 'GPU')

TRAIN = True
RADIUS = 5
SHAPE = RADIUS * 2 + 1
CLASSES = 8

tf.keras.mixed_precision.set_global_policy('mixed_float16')


# Generator that splits image into patches of radius pixels around a center pixel from image1 and returns a tuple (patch, center)
# where center is the pixel with the same coordinates as the patch center, taken from image2;
# This generator is used for training
def gen_image(image1, image2, radius):
    for path in zip(image1, image2):
        im_x = skio.imread(path[0])
        im_y = skio.imread(path[1])
        im_x = np.array(im_x)
        im_y = np.array(im_y)
        i_ = radius
        j_ = radius
        while i_ + radius + 1 < len(im_x):
            while j_ + radius + 1 < len(im_x[0]):
                arr = np.array([z[j_ - radius:j_ + radius + 1] for z in im_x[i_ - radius:i_ + radius + 1]])
                arr = np.expand_dims(arr, axis=0)
                arr = np.expand_dims(arr, axis=-1)
                label = np.array(im_y[i_ + 1][j_ + 1] // CLASSES)
                label = np.expand_dims(label, axis=0)
                label = np.expand_dims(label, axis=-1)
                yield arr, label
                j_ += 1
            i_ += 1
            j_ = radius


# Generator that splits image into patches of radius pixels around a center pixel from image1 and returns a tuple (patch, center)
# where center is the pixel with the same coordinates as the patch center, taken from image2;
# This generator is used for testing
def gen_image_test(image1, image2, radius, mask):
    i_ = radius
    j_ = radius
    while i_ + radius < len(image1):
        while j_ + radius < len(image1[0]):
            # if the patch center coordinates have the value 1 in the mask array select that position as a test sample
            if mask[i_ + 1][j_ + 1] == 1:
                arr = np.array([z[j_ - radius:j_ + radius + 1] for z in image1[i_ - radius:i_ + radius + 1]])
                arr = np.expand_dims(arr, axis=0)
                arr = np.expand_dims(arr, axis=-1)
                # print("SHAPE")
                # print(arr.shape)
                # print(arr)
                label = np.array(image2[i_ + 1][j_ + 1] // CLASSES)
                label = np.expand_dims(label, axis=0)
                label = np.expand_dims(label, axis=-1)
                # print(label)
                yield arr, label
            j_ += 1
        i_ += 1
        j_ = radius


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


batch_end_loss = list()


class SaveBatchLoss(tf.keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        batch_end_loss.append(logs['loss'])


# Generate a mask that splits the image pixel array into test and train
def gen_mask(image, split=0.8, ignore=0.5):
    image_mask = np.random.rand(len(image), len(image[0]))
    image_mask = np.where(image_mask <= ignore, -1, image_mask)
    image_mask = np.where(np.logical_and((image_mask - ignore) / (1 - ignore) <= split, image_mask != -1), 0,
                          image_mask)
    image_mask = np.where(np.logical_and((image_mask - ignore) / (1 - ignore) > split, image_mask != -1), 1, image_mask)
    return image_mask


# Read both images
imstack1 = "Data/Downscaled2/2001NA_d.png"
imstack2 = "Data/Downscaled2/2004NA_d.png"
imstack3 = "Data/Downscaled2/2007NA_d.png"
imstack4 = "Data/Downscaled2/2004EU_d.png"
imstack5 = "Data/Downscaled2/2007EU_d.png"

train_x = gen_image([imstack1, imstack2, imstack4],
                    [imstack2, imstack3, imstack5], RADIUS)

print("Begin")

model = models.Sequential()
model.add(layers.Conv2D(filters=16, kernel_size=3, activation='relu', input_shape=(SHAPE, SHAPE, 1)))
model.add(layers.Conv2D(32, 2, activation='relu'))
model.add(layers.MaxPooling2D((3, 3)))
model.add(layers.Conv2D(32, 2, activation='relu'))
model.add(layers.MaxPooling2D((3, 3)))
model.add(layers.Conv2D(32, 2, activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(16))
model.add(layers.Dense(CLASSES))
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=tf.optimizers.Adam(),
              metrics=['accuracy'])

print(model.summary())

history = model.fit(train_x, epochs=1, verbose=1, callbacks=SaveBatchLoss())

filename = "train_loss_3.csv"
f = open(filename, mode="w")
for x in batch_end_loss:
    f.write(str(x) + "\n")
f.close()
plt.plot(batch_end_loss)
plt.show()

model.save('saved_model/my_model_3')



