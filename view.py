import skimage.io as skio
import matplotlib.pyplot as plt

files = [
    # "1992AS_d.tif",
    # "1992EU_d.tif",
    # "1992NA_d.tif",
    # "1993AS_d.tif",
    "1993EU.tif",
    # "1993NA_d.tif",
    # "1994AS_d.tif",
    # "1994EU_d.tif",
    # "1994NA_d.tif",
    # "1995AS_d.tif",
    # "1995EU_d.tif",
    # "1995NA_d.tif",
    # "1996AS_d.tif",
    # "1996EU_d.tif",
    # "1996NA_d.tif",
    # "1997AS_d.tif",
    # "1997EU_d.tif",
    # "1997NA_d.tif",
    # "1998AS_d.tif",
    # "1998EU_d.tif",
    # "1998NA_d.tif",
    # "1999AS_d.tif",
    # "1999EU_d.tif",
    # "1999NA_d.tif",
    # "2000AS_d.tif",
    # "2000EU_d.tif",
    # "2000NA_d.tif",
    # "2001AS_d.tif",
    # "2001EU_d.tif",
    # "2001NA_d.tif",
    # "2002AS_d.tif",
    # "2002EU_d.tif",
    # "2002NA_d.tif",
    # "2003AS_d.tif",
    # "2003EU_d.tif",
    # "2003NA_d.tif",
    # "2004AS_d.tif",
    # "2004EU_d.tif",
    # "2004NA_d.tif",
    # "2005AS_d.tif",
    # "2005EU_d.tif",
    # "2005NA_d.tif",
    # "2006AS_d.tif",
    # "2006EU_d.tif",
    # "2006NA_d.tif",
    # "2007AS_d.tif",
    # "2007EU_d.tif",
    # "2007NA_d.tif",
    # "2008AS_d.tif",
    # "2008EU_d.tif",
    # "2008NA_d.tif",
    # "2009AS_d.tif",
    # "2009EU_d.tif",
    # "2009NA_d.tif",
    # "2010AS_d.tif",
    # "2010EU_d.tif",
    # "2010NA_d.tif",
    # "2011AS_d.tif",
    # "2011EU_d.tif",
    # "2011NA_d.tif",
    # "2012AS_d.tif",
    # "2012EU_d.tif",
    # "2012NA_d.tif",
    # "2013AS_d.tif",
    # "2013EU_d.tif",
    # "2013NA_d.tif"
]

for f in files:
    n = f.split(".")
    name = n[0]
    print(name)
    imstack1 = skio.imread("Data/" + f, plugin="tifffile")
    for i in range(len(imstack1)):
        for j in range(len(imstack1[0])):
            imstack1[i][j] = (imstack1[i][j] // 8) * 8
        # print(i)
    plt.imshow(imstack1)
    plt.show()
    skio.imsave("Data/Results/1993EU_true.png", imstack1, check_contrast=False)
