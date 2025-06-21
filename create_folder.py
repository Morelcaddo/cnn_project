import os


def batch_create_folders(root_dir, num_folders, prefix=''):
    """
    在指定目录下批量创建多个文件夹

    参数：
        root_dir：根目录路径，文件夹将在这个目录下创建
        num_folders：要创建的文件夹数量
        prefix：文件夹名称的前缀（可选）
    """
    for i in range(1, num_folders + 1):
        folder_name = f"{prefix}{i + 36}" if prefix else f"folder_{i}"
        folder_path = os.path.join(root_dir, folder_name)

        try:
            os.makedirs(folder_path, exist_ok=True)
            print(f"文件夹 '{folder_name}' 创建成功！")
        except Exception as e:
            print(f"创建文件夹 '{folder_name}' 时出错：{e}")


# 示例用法
if __name__ == "__main__":
    root_directory = "data/test"  # 替换为你的根目录路径
    number_of_folders = 26  # 想要创建的文件夹数量
    folder_prefix = "Sample0"  # 文件夹前缀（如不需要前缀，可以留空或设置为 None）

    if folder_prefix is None:
        batch_create_folders(root_directory, number_of_folders)
    else:
        batch_create_folders(root_directory, number_of_folders, folder_prefix)
