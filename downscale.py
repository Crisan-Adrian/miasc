import numpy
import skimage.io as skio
import matplotlib.pyplot as plt
import numpy as np

files = [
    # "1992AS.tif",
    # "1992EU.tif",
    "1992NA.tif",
    # "1993AS.tif",
    # "1993EU.tif",
    "1993NA.tif",
    # "1994AS.tif",
    # "1994EU.tif",
    "1994NA.tif",
    # "1995AS.tif",
    # "1995EU.tif",
    "1995NA.tif",
    # "1996AS.tif",
    # "1996EU.tif",
    "1996NA.tif",
    # "1997AS.tif",
    # "1997EU.tif",
    "1997NA.tif",
    # "1998AS.tif",
    # "1998EU.tif",
    "1998NA.tif",
    # "1999AS.tif",
    # "1999EU.tif",
    "1999NA.tif",
    # "2000AS.tif",
    # "2000EU.tif",
    "2000NA.tif",
    # "2001AS.tif",
    # "2001EU.tif",
    "2001NA.tif",
    # "2002AS.tif",
    # "2002EU.tif",
    "2002NA.tif",
    # "2003AS.tif",
    # "2003EU.tif",
    "2003NA.tif",
    # "2004AS.tif",
    # "2004EU.tif",
    "2004NA.tif",
    # "2005AS.tif",
    # "2005EU.tif",
    "2005NA.tif",
    # "2006AS.tif",
    # "2006EU.tif",
    "2006NA.tif",
    # "2007AS.tif",
    # "2007EU.tif",
    "2007NA.tif",
    # "2008AS.tif",
    # "2008EU.tif",
    "2008NA.tif",
    # "2009AS.tif",
    # "2009EU.tif",
    "2009NA.tif",
    # "2010AS.tif",
    # "2010EU.tif",
    "2010NA.tif",
    # "2011AS.tif",
    # "2011EU.tif",
    "2011NA.tif",
    # "2012AS.tif",
    # "2012EU.tif",
    "2012NA.tif",
    # "2013AS.tif",
    # "2013EU.tif",
    "2013NA.tif"
]

for f in files:
    n = f.split(".")
    name = n[0]
    print(name)
    imstack1 = skio.imread("Data/" + f, plugin="tifffile")

    BLOCK = 10
    n = len(imstack1)
    m = len(imstack1[0])

    print(n, m)

    downscaled = np.zeros([n // BLOCK, m // BLOCK], dtype=numpy.uint8)

    print(n // BLOCK, m // BLOCK)

    for i in range(0, n, BLOCK):
        for j in range(0, m, BLOCK):
            pixelValue = 0
            for k in range(i, i + BLOCK):
                for l in range(j, j + BLOCK):
                    pixelValue += imstack1[k][l]
            downscaled[i // BLOCK][j // BLOCK] = pixelValue // (BLOCK ** 2)
        # print(i)

    # plt.rcParams["figure.dpi"] = 200

    # plt.imshow(downscaled)
    # plt.show()
    skio.imsave("Data/Downscaled2/" + name + "_d.png", downscaled, check_contrast=False)
