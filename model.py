# import numpy as np
# import torch
from torch import nn
from torchvision import models
from collections import OrderedDict
import torchvision.transforms.functional as TF


class Model(object):
    @staticmethod  # 类中所声明的方法没有使用类中的变量,可以安全的声明为静态类型
    def creat_classifier(input_size, output_size, hidden_layers=[], dropout=0.5,
                         activation=nn.RReLU(), output_function=nn.LogSoftmax(dim=1)):
        dict = OrderedDict()  # 根据放入的先后顺序对dict进行排序，输出的值的排序好的

        if len(hidden_layers) == 0:
            dict['layer0'] = nn.Linear(input_size, output_size)  # nn.Linear用于设置全连接层

        else:
            dict['layer0'] = nn.Linear(input_size, hidden_layers[0])
            if activation:
                dict['activ0'] = activation
            if dropout:
                dict['drop_0'] = nn.Dropout(dropout)

            for layer, layer_in in enumerate(zip(hidden_layers[:-1], hidden_layers[1:])):
                dict['layer' + str(layer + 1)] = nn.Linear(layer_in[0], layer_in[1])
                if activation:
                    dict['activ' + str(layer + 1)] = activation
                if dropout:
                    dict['drop_' + str(layer + 1)] = nn.Dropout(dropout)
            dict['output'] = nn.Linear(hidden_layers[-1], output_size)

        if output_function:
            dict['output_function'] = output_function

        return nn.Sequential(dict)

    def creat_network(self, model_name='resnet50', output_size=102, hidden_layers=[1000]):
        if model_name == 'resnet50':
            model = models.resnet50(pretrained=True)
            model.fc = self.creat_classifier(2048, output_size, hidden_layers)

            print("Creat NetWork {} Done!".format(model_name))

            return model

        if model_name == 'resnet152':
            model = models.resnet152(pretrained=True)
            model.fc = self.creat_classifier(2048, output_size, hidden_layers)

            print("Creat NetWork {} Done!".format(model_name))

            return model

        return None
