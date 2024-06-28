import os
import random


def main():
    random.seed(0)  # 设置随机种子，保证随机结果可复现

    files_path = "./VOCdevkit/VOC2012/Annotations"   # 拿到文件路径
    assert os.path.exists(files_path), "path: '{}' does not exist.".format(files_path)

    val_rate = 0.5  # 以0.5的比例来划分训练集和验证集

    files_name = sorted([file.split(".")[0] for file in os.listdir(files_path)])  # 拿到所有标记文件的名字，也就是图片的名字
    files_num = len(files_name)
    # 从所有文件中，随机找出一般的文件。比如：我有0~9十个数，我从这十个数中随机找出5个数
    val_index = random.sample(range(0, files_num), k=int(files_num * val_rate))
    train_files = []
    val_files = []
    for index, file_name in enumerate(files_name):
        # 当前遍历的下标，是我们随机获取的下标就放入到 验证集中
        if index in val_index:
            val_files.append(file_name)
        else:
            train_files.append(file_name)

    try:
        # 使用了写入模式 "x"，如果 train.txt 文件已经存在，open 函数将会抛出一个错误。如果文件不存在，它会被创建。
        train_f = open("train.txt", "x")
        eval_f = open("val.txt", "x")
        # 写入操作不会自动添加换行符，所以使用 "\n".join(...) 在每个文件名后添加一个换行符，确保每个文件名在文件中占一行。
        train_f.write("\n".join(train_files))
        eval_f.write("\n".join(val_files))
    except FileExistsError as e:
        print(e)
        exit(1)


if __name__ == '__main__':
    main()
