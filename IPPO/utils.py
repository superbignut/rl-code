import os
import sys


def delete_dir_file(dir_path, root_dir_rm=False):
    """
    递归删除文件夹下文件和子文件夹里的文件，不会删除空文件夹
    :param dir_path: 文件夹路径
    :return:
    """
    if not os.path.exists(dir_path):
        return
    # 判断是不是一个文件路径，并且存在
    if os.path.isfile(dir_path) and os.path.exists(dir_path):
        os.remove(dir_path)  # 删除单个文件
    else:
        file_list = os.listdir(dir_path)
        for file_name in file_list:
            delete_dir_file(os.path.join(dir_path, file_name), root_dir_rm=True)
    if root_dir_rm == True and os.path.exists(dir_path):
        os.rmdir(dir_path)