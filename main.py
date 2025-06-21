import os

import Identification
import segmentation


def wait(path):
    while True:
        if os.path.exists(path):
            break


img = "img16"

len = segmentation.seg(img + ".png", img)


result = ""
for i in range(len):
    path = "data/output/" + img + "/" + str(i) + ".png"
    wait(path)
    c = Identification.identification(path)
    result = result + c
print("识别结果:" + result)
