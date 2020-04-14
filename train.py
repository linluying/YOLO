import torch.nn as nn
from torchvision import transforms, datasets
import json
import os
import torch.optim as optim
from model import vgg
import torch
import sys
sys.path.append('..')
import time

def load_data():
    batch_size = 32
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../"))  # get data root path
    image_path = data_root + "/data_set/PCB_data/"  # flower data set path

    train_dataset = datasets.ImageFolder(root=image_path+"train",
                                         transform=data_transform["train"])
    train_num = len(train_dataset)
    print('train_num: ', train_num)
    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=0)
    validate_dataset = datasets.ImageFolder(root=image_path + "val",
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    print('val_num: ', val_num)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=0)

    return train_loader, validate_loader

def train_VGG(train_loader, validate_loader, net, loss_function, optimizer, device, num_epochs):
    net = net.to(device)
    batch_out = 0
    for epoch in range(num_epochs):
        # train
        net.train()
        running_loss, train_acc_sum, n = 0.0, 0.0, 0
        for step, data in enumerate(train_loader, start=0):
            images, labels = data
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.cpu().item()
            train_acc_sum += (outputs.argmax(dim=1) == labels.to(device)).sum().cpu().item()
            # print train process
            rate = (step + 1) / len(train_loader)
            a = "*" * int(rate * 50),
            b = "." * int((1 - rate) * 50)
            n += labels.shape[0]
            batch_out += 1
            print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
        print('step: ', step)
        print('batch_out: ', batch_out)
        print('n1: ', n)
        accurate_test = evaluate_accuracy(validate_loader, net, optimizer, device)
        print('[epoch %d], train_loss: %.3f, train acc %.3f, test_accuracy: %.3f'
              % (epoch + 1, running_loss / step, train_acc_sum / n, accurate_test))
    print('Finished Training')

def evaluate_accuracy(validate_loader, net, optimizer, device):
    model_name = "vgg19"
    save_path = './{}Net.pth'.format(model_name)
    # validate
    net.eval()
    best_acc = 0.0
    acc, n= 0.0 , 0# accumulate accurate number / epoch
    with torch.no_grad():
        for data_test in validate_loader:
            test_images, test_labels = data_test
            optimizer.zero_grad()
            outputs = net(test_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += (predict_y == test_labels.to(device)).sum().item()
            n += test_labels.shape[0]
        accurate_test = acc / n
        if accurate_test > best_acc:
            best_acc = accurate_test
            torch.save(net.state_dict(), save_path)
        print('n2: ', n)
    return accurate_test

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    train_loader, validate_loader = load_data()
    model_name = "vgg19"
    net = vgg(model_name=model_name, class_num=6, init_weights=True)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    num_epochs = 1000
    train_VGG(train_loader, validate_loader, net, loss_function, optimizer, device, num_epochs)



