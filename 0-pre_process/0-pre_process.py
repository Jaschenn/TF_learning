import numpy as np
from PIL import ImageFilter,Image

image = Image.open("/Users/jaschen/Desktop/22.png")
image = image.convert("L")
pixdata = image.getdata()
width = image.size[0]
height = image.size[1]
pixdata = np.reshape(pixdata,[width,height])
