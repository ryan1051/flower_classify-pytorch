
import numpy as np
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import time


from model import Model
from data_process import Dataset

md = Model()
dataset = Dataset()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on {}".format(str(device).upper()))

model_name = 'resnet50'
output_size = 102
hidden_layers = [1000]

model = md.creat_network(model_name, output_size, hidden_layers)
model.to(device)

save_path = "output/"


def train(epochs, model, optimizers, lr_sheduler=None,
          dataloaders=dataset.dataloaders, state_dict=None,
          checkpoint_path="checkpoint.pt", accuracy_target=None, show_graphs=True):

    if state_dict is None:
        state_dict = {
            'elapsed_time': 0,
            'trace_log': [],
            'trace_train_loss': [],
            'trace_train_lr': [],
            'valid_loss_min': np.Inf,
            'trace_valid_loss': [],
            'trace_accuracy': [],
            'epochs_trained': 0
        }
        state_dict['trace_log'].append('PHASE ONE')

    criterion = nn.NLLLoss()  # The negative log likelihood loss

    for epoch in range(1, epochs+1):
        try:
            lr_sheduler.step()  # torch.optim.lr_scheduler
        except TypeError:
            try:
                if lr_sheduler.min_lrs[0] == lr_sheduler.optimizer.param_groups[0]['lr']:
                    break
                lr_sheduler.step(valid_loss)
            except NameError:
                lr_sheduler.step(np.Inf)
        except:
            pass

        epoch_start_time = time.time()
        ###############
        #    TRAIN    #
        ###############
        train_loss = 0
        model.train()
        for images, labels in dataloaders['train_data']:
            # Move tensors to device
            images, labels = images.to(device), labels.to(device)

            # Clear optimizers
            [opt.zero_grad() for opt in optimizers]

            # Pass train batch through model feed-forward
            output = model(images)

            # Calculate loss for this train batch
            batch_loss = criterion(output, labels)
            # Do the backpropagation  ??
            batch_loss.backward()

            # Optimize parameters
            [opt.step() for opt in optimizers]
            # Track train loss
            train_loss += batch_loss.item()*len(images)

        # Track how many epochs has already run
        state_dict['elapsed_time'] += time.time() - epoch_start_time
        state_dict['epochs_trained'] += 1

        ###############
        #    VALID    #
        ###############
        valid_loss = 0
        accuracy = 0
        top_class_graph = []
        labels_graph = []

        # Set model to evaluation mode
        model.eval()
        with torch.no_grad():  # 不计算梯度，不进行反向传播
            for images, labels in dataloaders['valid_data']:
                labels_graph.extend(labels)  # 在list里追加另一个list
                # Move tensors to device
                images, labels = images.to(device), labels.to(device)

                # Get prediction for this validation batch
                output = model(images)

                # Calculate loss
                batch_loss = criterion(output, labels)
                # Track validation loss
                valid_loss += batch_loss.item()*len(images)

                # Calculate accuracy
                output = torch.exp(output)  # 取对数
                top_ps, top_class = output.topk(1, dim=1)  # 沿给定dim维度返回(最大值,下标索引)
                top_class_graph.extend((top_class.view(-1)).to('cpu').numpy())  # ?
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()*len(images)

        train_loss = train_loss/len(dataloaders['train_data'].dataset)
        valid_loss = valid_loss/len(dataloaders['valid_data'].dataset)
        accuracy = accuracy/len(dataloaders['valid_data'].dataset)

        state_dict['trace_train_loss'].append(train_loss)
        try:
            state_dict['trace_train_lr'].append(lr_sheduler.get_lr()[0])
        except:
            state_dict['trace_train_lr'].append(optimizers[0].state_dict()['param_groups'][0]['lr'])  # [0]?

        state_dict['trace_valid_loss'].append(valid_loss)
        state_dict['trace_accuracy'].append(accuracy)
        ###################
        #    PRINT LOG    #
        ###################
        log = 'Epoch:{} lr:{:.8f} Training_loss:{:.6f} Valid_loss:{:.6f} Valid_accuracy:{:.2f} Elapsed_time:{:.2f}'.format(
              state_dict['epochs_trained'], state_dict['trace_train_lr'][-1], train_loss, valid_loss, accuracy,
              state_dict['elapsed_time'])
        print(log)

        # save model if validation loss has decreased
        if valid_loss <= state_dict['valid_loss_min']:
            print('Validation loss decrease: {:.6f}->{:.6f}. Saving model...'.format(
                state_dict['valid_loss_min'], valid_loss))

            checkpoint = {'model_state_dict': model.state_dict(),  # 学习到的参数
                          'optimizer_state_dict': optimizers[0].state_dict(),
                          'training_state_dict': state_dict}
            if lr_sheduler:
                checkpoint['lr_sheduler_state_dict'] = lr_sheduler.state_dict()

            torch.save(checkpoint, checkpoint_path)  # torch.save(model, checkpoint_path)能保存更多
            state_dict['valid_loss_min'] = valid_loss

        if show_graphs:
            plt.figure(figsize=(25, 8))
            plt.plot(np.array(labels_graph), 'k.')
            plt.plot(np.array(top_class_graph), 'r.')
            # plt.show()
            plt.savefig("train1.png")

            plt.figure(figsize=(25, 5))
            plt.subplot(1, 2, 1)
            plt.plot(np.array(state_dict['trace_train_loss']), 'b', label='train_loss')
            plt.plot(np.array(state_dict['trace_valid_loss']), 'r', label='valid_loss')
            plt.plot(np.array(state_dict['trace_accuracy']), 'g', label='accuracy')

            plt.subplot(1, 2, 2)
            plt.plot(np.array(state_dict['trace_train_lr']), 'b', label='train_loss')
            plt.savefig("train2.png")

        # stop training loop if accuracy_target has been reached
        if accuracy_target and state_dict['trace_accuracy'][-1] >= accuracy_target:
            print('Get accuracy target. Stop training!')
            break

    return state_dict


def test_model(model, dataloader=dataset.dataloaders, show_graphs=True):
    ###############
    #    TEST     #
    ###############
    criterion = nn.NLLLoss()
    test_loss = 0
    accuracy = 0
    top_class_graph = []
    labels_graph = []

    model.eval()
    with torch.no_grad():
        for images, labels in dataloader['test_data']:
            labels_graph.extend(labels)

            images, labels = images.to(device), labels.to(device)

            output = model(images)
            # print("output:{}".format(output))

            batch_loss = criterion(output, labels)
            test_loss += batch_loss.item()*len(images)

            output = torch.exp(output)
            # print("exp(output):".format(output))
            top_ps, top_class = output.topk(1, dim=1)
            # print("top_ps:{} top_class:{}".format(top_ps, top_class))
            top_class_graph.extend(top_class.view(-1).to('cpu').numpy())
            equals = top_class == labels.view(*top_class.shape)  # 一个batch里结果和label是否一致的list
            # print("equals:{}".format(equals))
            accuracy += torch.sum(equals.type(torch.FloatTensor)).item()

    test_loss = test_loss/len(dataloader['test_data'].dataset)
    accuracy = accuracy/len(dataloader['test_data'].dataset)

    print("Test Loss:{:.6f} Test Accuracy:{}".format(test_loss, accuracy))
    print("output:{}".format(output))
    print("top_ps:{}, top_clasee:{}".format(top_ps, top_class))
    print("top_class_graph:{}".format(top_class_graph))
    print("equals:{}".format(equals))

    if show_graphs:
        plt.figure(figsize=(25, 13))
        plt.plot(np.array(labels_graph), 'k.')
        plt.plot(np.array(top_class_graph), 'r.')
        # plt.show()
        plt.savefig("test.png")


def freeze_parameters(root, freeze=True):
    [param.requires_grad_(not freeze) for param in root.parameters()]  # false:不计算梯度


def load_model(checkpoint_path, state_dict):
    try:
        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint['training_state_dict']
        model.load_state_dict(checkpoint['model_state_dict'])
    except:
        pass

    return state_dict


def start(model):
    PHASE_ONE = 2  # 训练全连接层
    PHASE_TWO = 2  # 微调卷积层权重
    PHASE_THREE = 2  # 继续提高

    TEST = True

    if PHASE_ONE > 0:
        freeze_parameters(model)
        freeze_parameters(model.fc, False)  # 只有fc计算梯度、反向传播

        fc_optimizer = optim.Adagrad(model.fc.parameters(), lr=0.01, weight_decay=0.001)
        optimizers = [fc_optimizer]

        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(fc_optimizer, mode='min', factor=0.1, patience=5,
                                                            threshold=0.01, min_lr=0.00001)
        checkpoint_path = save_path + "checkpoint_phase_one.pt"

        state_dict = train(PHASE_ONE, model, optimizers, lr_sheduler=lr_scheduler,
                           state_dict=None, checkpoint_path=checkpoint_path)

        print(*state_dict['trace_log'], sep="\n")

        state_dict = load_model(checkpoint_path, state_dict)

    if PHASE_TWO > 0:
        state_dict['trace_log'].append('PHASE_TWO')

        freeze_parameters(model, False)  # 整个网络都进行梯度计算、反向传播

        conv_optimizer = optim.Adagrad(model.parameters(), lr=0.0001, weight_decay=0.001)
        optimizers = [fc_optimizer, conv_optimizer]

        checkpoint_path = save_path + "checkpoint_phase_two.pt"

        state_dict = train(PHASE_TWO, model, optimizers, lr_sheduler=None, state_dict=state_dict,
                           checkpoint_path=checkpoint_path)

        print(*state_dict['trace_log'], sep="\n")

        state_dict = load_model(checkpoint_path, state_dict)

    if PHASE_THREE > 0:
        state_dict['trace_log'].append('PHASE_THREE')

        freeze_parameters(model)
        freeze_parameters(model.fc, False)

        optimizers = [fc_optimizer]

        lr_scheduler = optim.lr_scheduler.MultiStepLR(fc_optimizer, milestones=[0], gamma=0.01)

        checkpoint_path = save_path + "checkpoint_phase_three.pt"

        state_dict = train(PHASE_THREE, model, optimizers, lr_sheduler=lr_scheduler,
                           state_dict=state_dict, checkpoint_path=checkpoint_path)

        print(*state_dict['trace_log'], sep="\n")

        state_dict = load_model(checkpoint_path, state_dict)

    if TEST:
        test_model(model)


def save_checkpoint(checkpoint_path='checkpoint.pt'):
    model.to('cpu')
    dataset.get_label2name()
    checkpoint = {'model_name': model_name,
                  'output_size': output_size,
                  'hidden_layers': hidden_layers,
                  'model_state_dict': model.state_dict(),
                  'class_to_idx': dataset.class_to_idx,
                  'label_to_name': dataset.label_to_name}
    torch.save(checkpoint, checkpoint_path)


dataset.prepare_data()
start(model)
save_checkpoint(save_path + 'checkpoint.pt')
