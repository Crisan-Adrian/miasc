import random
import numpy as np
import csv
import pandas as pd
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import seaborn as sns

# Make numpy values easier to read.
from matplotlib import pyplot as plt

print("TensorFlow version:", tf.__version__)

dataSet = pd.read_csv("dataSet.csv")

class_0, class_1 = dataSet.result.value_counts()
c0 = dataSet[dataSet['result'] == 0]
c1 = dataSet[dataSet['result'] == 1]
df_1 = c1.sample(class_0, replace=True)

oversampled_df = pd.concat([c0,df_1], axis=0)
print(dataSet.result.value_counts())
print(oversampled_df.result.value_counts())

train = oversampled_df.sample(frac=0.8, random_state=12)
test = oversampled_df.loc[~oversampled_df.index.isin(train.index)]

# train = dataSet.sample(frac=0.8, random_state=12)
# test = dataSet.loc[~dataSet.index.isin(train.index)]

train_x = train.copy()
train_y = train_x.pop("result")

test_x = test.copy()
test_y = test_x.pop("result")

model = tf.keras.Sequential([
    tf.keras.Input(shape=(6,)),
    tf.keras.layers.Dense(4),
    tf.keras.layers.Dense(20),
    tf.keras.layers.Dense(1)
])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.optimizers.Adam(),
              metrics=['accuracy'])

model.fit(train_x, train_y, epochs=100)

positive_x = dataSet[dataSet['result'] == 1]
positive_y = positive_x.pop('result')

negative_x = dataSet[dataSet['result'] == 0]
negative_y = negative_x.pop('result')

model.evaluate(test_x, test_y, verbose=2)

model.evaluate(positive_x, positive_y, verbose=2)

model.evaluate(negative_x, negative_y, verbose=2)

