import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import AlexNet

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),  # 将 H,W,C 转换为了 C,H,W，并且将图片类型转为了tensor类型
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    # load image
    img_path = "./tulip.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)

    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)  # 添加一个batch维度，因为训练的时候需要batch这个维度

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    model = AlexNet(num_classes=5).to(device)

    # load model weights
    weights_path = "./AlexNet.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path))

    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()  # torch.softmax在 CPU 上执行可能更合适
        print(f"output预测值为：{output}")  # tensor([-3.0321, -2.4126,  0.1578,  0.9257,  3.1781])
        predict = torch.softmax(output, dim=0)  # output 拿到的是一个在每种类别上的预测值，需要通过softmax归一化，实现概率分布
        predict_cla = torch.argmax(predict).numpy()  # argmax获取概率最大处所对应的索引值

    print_res = "class: {}   prob: {:.3}".format(
        class_indict[str(predict_cla)],
        predict[predict_cla].numpy()
    )
    plt.title(print_res)
    for i in range(len(predict)):
        print(
            "class: {:10}   prob: {:.3}".format(
                class_indict[str(i)],
                predict[i].numpy()
            )
        )
    plt.show()


if __name__ == '__main__':
    main()
