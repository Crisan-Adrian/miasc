import matplotlib.pyplot as plt
import numpy as np
import skimage.io as skio
import tensorflow as tf

pred = skio.imread("Data/Results/2013EU_pred.png")
true = skio.imread("Data/Results/2013EU_true.png")

total = len(pred) * len(pred[0])
totalH = 0
correct = 0
correctH = 0

for i in range(len(pred)):
    for j in range(len(pred[0])):
        if true[i][j] >= 16:
            totalH += 1
            if pred[i][j] == true[i][j]:
                correctH += 1
        if pred[i][j] == true[i][j]:
            correct += 1

print("Accuracy:", correct / total)
print("Accuracy (adjusted):", correctH / totalH)
