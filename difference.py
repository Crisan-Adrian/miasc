import skimage.io as skio
import matplotlib.pyplot as plt
import numpy as np

# files = [["Data/1992EU.tif", "Data/2013EU.tif"]]
#
# for f in files:
#     im1 = skio.imread(f[0], plugin="tifffile")
#     im2 = skio.imread(f[1], plugin="tifffile")
#
#     n = len(im1)
#     m = len(im1[0])
#
#     im = [[] for i in range(n)]
#     for i in range(n):
#         im[i] = [0 for x in range(m)]
#         for j in range(m):
#             if im2[i][j] < im1[i][j]:
#                 im[i][j] = 0
#             else:
#                 im[i][j] = im2[i][j] - im1[i][j]
#
#     im = np.array(im)
#
#     plt.imshow(im)
#     plt.show()
#
#     plt.hist([im1.flatten(), im2.flatten()], 63, label=[1992, 2013])
#     plt.legend([1992, 2013])
#     plt.title("Visibility values")
#     plt.savefig("Visibility_difference_EU.png")
#
#     plt.imshow(np.where(im1 > 16, 63, 0))
#     plt.show()
#
#     print("Done")


# filesTrends = ["Data/1992EU.tif",
#                "Data/1993EU.tif",
#                "Data/1994EU.tif",
#                "Data/1995EU.tif",
#                "Data/1996EU.tif",
#                "Data/1997EU.tif",
#                "Data/1998EU.tif",
#                "Data/1999EU.tif",
#                "Data/2000EU.tif",
#                "Data/2001EU.tif",
#                "Data/2002EU.tif",
#                "Data/2003EU.tif",
#                "Data/2004EU.tif",
#                "Data/2005EU.tif",
#                "Data/2006EU.tif",
#                "Data/2007EU.tif",
#                "Data/2008EU.tif",
#                "Data/2009EU.tif",
#                "Data/2010EU.tif",
#                "Data/2011EU.tif",
#                "Data/2012EU.tif",
#                "Data/2013EU.tif"]
# year = 1992
#
# avgs = {}
# for f in filesTrends:
#     im = skio.imread(f, plugin="tifffile")
#     n = len(im)
#     m = len(im[0])
#     size = n * m
#     sumVis = np.sum(im)
#     print(sumVis)
#     print(sumVis / size)
#
#     avgs[year] = sumVis / size
#     year += 1
#     print("Done")
#
# keys = list(avgs.keys())
# values = list(avgs.values())
#
# valueDiffs = [values[i] - values[0] for i in range(len(values))]

# plt.bar(keys, values)
# plt.title("Average visibility")
# plt.savefig("Average_Visibility_EU.png")
# plt.bar(keys, valueDiffs)
# plt.title("Average visibility difference")
# plt.savefig("Average_Visibility_difference_EU.png")


# filesTrends = ["Data/Downscaled/1992AS_d.tif",
#                "Data/Downscaled/2012AS_d.tif"]
# year = 1992
#
# avgs = {}
# for f in filesTrends:
#     im = skio.imread(f, plugin="tifffile")
#     n = len(im)
#     m = len(im[0])
#     size = n * m
#     sumVis = np.sum(im)
#     print(sumVis)
#     print(sumVis / size)
#
#     avgs[year] = sumVis / size
#     year += 20
#     print("Done")
#
# keys = list(avgs.keys())
# values = list(avgs.values())
#
# valueDiffs = [values[i] - values[0] for i in range(len(values))]
#
# plt.bar(keys, values)
# plt.title("Average visibility d")
# plt.show()
# plt.bar(keys, valueDiffs)
# plt.title("Average visibility difference d")
# plt.show()

filesTrends = ["Data/Downscaled2/1992EU_d.png",
               "Data/Downscaled2/1993EU_d.png",
               "Data/Downscaled2/1994EU_d.png",
               "Data/Downscaled2/1995EU_d.png",
               "Data/Downscaled2/1996EU_d.png",
               "Data/Downscaled2/1997EU_d.png",
               "Data/Downscaled2/1998EU_d.png",
               "Data/Downscaled2/1999EU_d.png",
               "Data/Downscaled2/2000EU_d.png",
               "Data/Downscaled2/2001EU_d.png",
               "Data/Downscaled2/2002EU_d.png",
               "Data/Downscaled2/2003EU_d.png",
               "Data/Downscaled2/2004EU_d.png",
               "Data/Downscaled2/2005EU_d.png",
               "Data/Downscaled2/2006EU_d.png",
               "Data/Downscaled2/2007EU_d.png",
               "Data/Downscaled2/2008EU_d.png",
               "Data/Downscaled2/2009EU_d.png",
               "Data/Downscaled2/2010EU_d.png",
               "Data/Downscaled2/2011EU_d.png",
               "Data/Downscaled2/2012EU_d.png",
               "Data/Downscaled2/2013EU_d.png"]
year = 1992

avgs = {}
for f in filesTrends:
    im = skio.imread(f)
    print(year)
    im = np.where(im <= 7, 0, im)
    im = np.where(np.logical_and(15 >= im, im > 7), 8, im)
    im = np.where(np.logical_and(23 >= im, im > 15), 16, im)
    im = np.where(np.logical_and(31 >= im, im > 15), 24, im)
    im = np.where(np.logical_and(39 >= im, im > 31), 32, im)
    im = np.where(np.logical_and(47 >= im, im > 39), 40, im)
    im = np.where(np.logical_and(55 >= im, im > 47), 48, im)
    im = np.where(im > 55, 56, im)

    im = im/8

    flat = im.flatten()

    # flat = [flat[x] for x in range(len(flat)) if flat[x] != 7]

    plt.hist(flat)
    plt.savefig("Visibility_classes_EU"+str(year)+".png")
    plt.clf()
    year += 1
