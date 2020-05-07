import torch
from model import Model
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np
import matplotlib.pyplot as plt

'''
 functional可以提供了一些更加精细的变换，用于搭建复杂的变换流水线。
 和transforms相反，函数变换的参数不包含随机数种子生成器。这意味着你必须指定所有参数的值，但是你可以自己引入随机数。
'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)

    model_name = checkpoint['model_name']
    output_size = checkpoint['output_size']
    hidden_layers = checkpoint['hidden_layers']

    model = Model().creat_network(model_name=model_name, output_size=output_size, hidden_layers=hidden_layers)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.label_to_name = checkpoint['label_to_name']

    return model


def process_image(image):
    normalize_mean = np.array([0.485, 0.456, 0.406])
    normalize_std = np.array([0.229, 0.224, 0.225])

    image = TF.resize(image, 256)

    upper_pixel = (image.height - 244) // 2
    left_pixel = (image.width - 244) // 2
    image = TF.crop(image, upper_pixel, left_pixel, 224, 224)  # 裁剪指定PIL图像(图像，最上侧像素坐标，最左侧像素坐标，要裁剪出的高度，要裁剪出的宽度）
    # 用transforms.CenterCrop(224)不行?
    image = TF.to_tensor(image)  # 将PIL Image或numpy.ndarray转化成张量
    image = TF.normalize(image, normalize_mean, normalize_std)

    return image


def predict(image_path, model, topk=5):
    image = Image.open(image_path)
    image = process_image(image)

    with torch.no_grad():
        model.eval()

        image = image.view(1, 3, 224, 224)
        image = image.to(device)

        prediction = model.forward(image)
        print("pred:{}".format(prediction))

        prediction = torch.exp(prediction)
        print("exp(pred):{}".format(prediction))
        top_ps, top_class = prediction.topk(topk, dim=1)

    return top_ps, top_class


def inference(image, checkpoint_path):
    model = load_checkpoint(checkpoint_path)
    model.to(device)

    probs, classes = predict(image, model)
    print(probs)
    print(classes)

    probs = probs.data.cpu()
    probs = probs.numpy().squeeze()  # 从矩阵shape中，去掉维度为1的
    print("squeeze:{}".format(probs))

    classes = classes.data.cpu()
    classes = classes.numpy().squeeze()
    print("squeeze:{}".format(classes))
    classes = [model.label_to_name[cla].title() for cla in classes]
    print("name:{}".format(classes))

    label = model.class_to_idx[str(30)]
    title = model.label_to_name[label].title()

    fig = plt.figure(figsize=(8, 10))
    ax1 = fig.add_subplot(2, 1, 1, xticks=[], yticks=[])

    image = Image.open(image)
    # image = process_image(image)
    ax1.imshow(image)

    ax2 = fig.add_subplot(2, 1, 2, xticks=[], yticks=[])
    ax2.barh(np.arange(5), probs)
    ax2.set_yticks(np.arange(5))
    ax2.set_yticklabels(classes)
    ax2.set_ylim(-1, 5)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 1.1)
    ax2.set_title('Class Probability')

    fig.savefig("inference.png")


inference('/home/ruwei/MyCode/pytorch/flower_classify/flower_data/valid/2/image_05101.jpg', 'output/checkpoint.pt')

