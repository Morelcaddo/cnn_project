import os
import random
import shutil


def random_move_images(source_folder, target_folder, percentage=10):
    """
    从源文件夹随机抽取指定百分比的图像文件，并将它们剪切到目标文件夹

    参数:
        source_folder: 源文件夹路径
        target_folder: 目标文件夹路径
        percentage: 要抽取的图片百分比，默认为10%
    """
    # 检查源文件夹是否存在
    if not os.path.exists(source_folder):
        print(f"源文件夹 '{source_folder}' 不存在")
        return

    # 创建目标文件夹如果不存在
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 获取所有图片文件
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    image_files = []
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                image_files.append(os.path.join(root, file))

    # 计算要抽取的图片数量
    num_images = len(image_files)
    if num_images == 0:
        print("源文件夹中没有找到图像文件")
        return

    num_to_move = max(1, int(num_images * percentage / 100))

    print(f"将在 {source_folder} 中抽取 {num_to_move} 张图片（共 {num_images} 张，{percentage}%）")

    # 随机选择图片
    selected_images = random.sample(image_files, num_to_move)

    # 将选中的图片剪切到目标文件夹
    for img_path in selected_images:
        img_name = os.path.basename(img_path)
        target_path = os.path.join(target_folder, img_name)

        shutil.move(img_path, target_path)
        print(f"已移动: {img_name}")

    print(f"成功将 {len(selected_images)} 张图片从 {source_folder} 移动到 {target_folder}")


if __name__ == "__main__":
    # 设置源文件夹和目标文件夹路径
    source_folder = "D:\AI _WorkSoace\cnn_project\data\\train"  # 替换为你的源文件夹路径
    target_folder = "D:\AI _WorkSoace\cnn_project\data\\test"  # 替换为你的目标文件夹路径

    for i in range(26):
        # 调用函数
        print(source_folder + "\Sample0" + str(i + 37))
        print(target_folder + "\Sample0" + str(i + 1))
        random_move_images(source_folder + "\Sample0" + str(i + 37), target_folder + "\Sample0" + str(i + 37), 10)
