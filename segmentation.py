import os

from PIL import Image, ImageOps, ImageFilter, ImageEnhance


def seg(img_path, img_name):
    path = r"data/material/" + img_path
    img = Image.open(path).convert('L')
    # 对灰度图像进行二值化处理，调整阈值以更好地分离字母
    img = img.point(lambda x: 0 if x < 128 else 255, '1')
    # 将图片转换为黑底白图
    img = ImageOps.invert(img)

    # 获取图片宽度和高度
    width, height = img.size

    # 初始化裁剪的起始位置
    start_x = 0
    # 用于保存裁剪后的图片
    cropped_images = []

    # 得到所有字符左右边界的x轴坐标
    char_x = []
    # 得到切割位置
    cut_list = []

    # 从左到右进行遍历，优化边界检测逻辑
    for x in range(0, width - 1):
        # 获取当前列的像素值
        column_pixels = [img.getpixel((x, y)) for y in range(height)]
        column_pixels_left = [img.getpixel((x - 1, y)) for y in range(height)]
        column_pixels3_right = [img.getpixel((x + 1, y)) for y in range(height)]
        # 调整边界检测逻辑，连续多列像素和变化较大的位置才认为是边界
        if sum(column_pixels) != 0 and (int(bool(sum(column_pixels_left))) ^ int(bool(sum(column_pixels3_right)))):
            char_x.append(x)

    # 将图片的始末位置加入到char_x中，方便处理
    char_x.insert(0, 0)
    char_x.append(width)

    # 两两求平均，得到切割位置cut_list
    for i in range(0, len(char_x), 2):
        cut_list.append(int((char_x[i] + char_x[i + 1]) / 2))

    # 切割出单个字符
    for j in range(len(cut_list) - 1):
        cut_image = img.crop((cut_list[j], 0, cut_list[j + 1], height))
        cropped_images.append(cut_image)

    # 继续裁剪，将上下多余的部分裁剪掉，并调整裁剪逻辑
    for a in cropped_images:
        width, height = a.size
        if height >= 30:
            # 遍历上半部分，找到第一个白色像素的位置
            top = 0
            for y in range(height):
                pixel = [a.getpixel((x, y)) for x in range(width)]
                if sum(pixel) > 10:
                    top = max(0, y - 5)  # 调整偏移量
                    break

            # 遍历下半部分，找到最后一个白色像素的位置
            bottom = 0
            for y in range(height - 1, -1, -1):
                pixel = [a.getpixel((x, y)) for x in range(width)]
                if sum(pixel) > 10:
                    bottom = min(y + 5, height - 1)  # 调整偏移量
                    break
            # 裁剪图像，只保留中间白色数字的部分
            cropped_images[cropped_images.index(a)] = a.crop((0, top, width, bottom + 1))
        else:
            cropped_images[cropped_images.index(a)] = a

    os.makedirs(f'data/output/{img_name}', exist_ok=True)

    idx = 0
    # 展示分割结果并保存，调整居中处理逻辑
    for x in cropped_images:
        x = ImageOps.invert(x)
        # 转换为RGB格式
        rgb_image = x.convert('RGB')
        # 创建一个白色背景
        new_width = 128
        new_height = 128
        # 计算缩放比例，确保图像在 128x128 像素的图片中占据更大的空间
        width_ratio = new_width / rgb_image.width
        height_ratio = new_height / rgb_image.height
        scale_ratio = min(width_ratio, height_ratio) * 0.95  # 0.95 是一个调整系数，可以根据需要调整
        scaled_width = int(rgb_image.width * scale_ratio)
        scaled_height = int(rgb_image.height * scale_ratio)
        # 创建一个白色背景
        white_background = Image.new('RGB', (new_width, new_height), (255, 255, 255))
        # 缩放图像并应用平滑滤镜
        scaled_image = rgb_image.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)
        scaled_image = scaled_image.filter(ImageFilter.SMOOTH)
        # 计算图像在背景上的位置，使其居中
        # 增加居中调整逻辑
        offset_x = (new_width - scaled_width) // 2
        offset_y = (new_height - scaled_height) // 2
        # 计算字符在裁剪后的图像中的位置，调整偏移量以使字符居中
        char_width, char_height = scaled_image.size
        if char_width > 0 and char_height > 0:
            # 遍历字符图像的上下左右，找到字符的实际边界
            char_top = 0
            for y in range(char_height):
                pixel_sum = sum([scaled_image.getpixel((x, y))[0] for x in range(char_width)])
                if pixel_sum < 255 * char_width:  # 假设背景为白色，像素值为 255
                    char_top = y
                    break
            char_bottom = char_height - 1
            for y in range(char_height - 1, -1, -1):
                pixel_sum = sum([scaled_image.getpixel((x, y))[0] for x in range(char_width)])
                if pixel_sum < 255 * char_width:
                    char_bottom = y
                    break
            char_left = 0
            for x in range(char_width):
                pixel_sum = sum([scaled_image.getpixel((x, y))[0] for y in range(char_height)])
                if pixel_sum < 255 * char_height:
                    char_left = x
                    break
            char_right = char_width - 1
            for x in range(char_width - 1, -1, -1):
                pixel_sum = sum([scaled_image.getpixel((x, y))[0] for y in range(char_height)])
                if pixel_sum < 255 * char_height:
                    char_right = x
                    break
            # 计算字符的宽度和高度
            char_actual_width = char_right - char_left + 1
            char_actual_height = char_bottom - char_top + 1
            # 计算字符在背景上的居中位置
            char_center_x = offset_x + char_left + char_actual_width // 2
            char_center_y = offset_y + char_top + char_actual_height // 2
            # 计算字符在背景上的左上角位置
            char_x = char_center_x - char_actual_width // 2
            char_y = char_center_y - char_actual_height // 2
            # 确保字符在背景上不超出边界
            char_x = max(0, min(char_x, new_width - char_actual_width))
            char_y = max(0, min(char_y, new_height - char_actual_height))
            # 将字符放置在背景上的居中位置
            white_background.paste(scaled_image.crop((char_left, char_top, char_right + 1, char_bottom + 1)),
                                   (char_x, char_y))
        else:
            white_background.paste(scaled_image, ((new_width - scaled_width) // 2, (new_height - scaled_height) // 2))
        # 增强对比度
        enhancer = ImageEnhance.Contrast(white_background)
        enhanced_image = enhancer.enhance(1.5)
        # 应用平滑滤镜
        enhanced_image = enhanced_image.filter(ImageFilter.SMOOTH_MORE)
        # 直接保存处理后的图像，避免使用matplotlib
        file_path = os.path.join('data/output', img_name, f'{idx}.png')
        enhanced_image.save(file_path, dpi=(600, 600))
        idx += 1

    return idx
