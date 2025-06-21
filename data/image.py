import cv2
import numpy as np
from PIL import Image

# 读取图像
image = cv2.imread('material/img1.jpg')

# 转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 二值化处理
_, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

# 形态学操作（可选，用于分离粘连字母）
kernel = np.ones((2, 2), np.uint8)
eroded = cv2.erode(binary, kernel, iterations=1)
dilated = cv2.dilate(eroded, kernel, iterations=1)

# 寻找轮廓
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 根据轮廓分割字母
letter_images = []
for contour in contours:
    # 获取轮廓的边界矩形
    x, y, w, h = cv2.boundingRect(contour)

    # 裁剪出字母区域
    letter_image = image[y:y + h, x:x + w]

    # 将裁剪后的字母图像添加到列表中
    letter_images.append(letter_image)

# 保存分割后的字母图像
for i, letter_image in enumerate(letter_images):
    cv2.imwrite(f'letter_{i}.jpg', letter_image)