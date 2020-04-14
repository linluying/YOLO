import torch
from model import vgg
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json
import glob

data_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# imgs = glob.glob(r'../data_set/3/mousebite/*.jpg')

imgs = glob.glob(r'C:\Users\502\Desktop\pytorch_learning\data_set\3\missinghole\*.jpg')
# load image
for img in imgs:
    # img = Image.open("../data_set/test/1.jpg")
    img_name = img.split('\\')[-1].split('.')[0]
    img = Image.open(img)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    try:
        json_file = open('./class_indices.json', 'r')
        class_indict = json.load(json_file)
    except Exception as e:
        print(e)
        exit(-1)

    # create model
    model = vgg(model_name="vgg19", class_num=6)
    # load model weights
    model_weight_path = "./vgg19Net1000.pth"
    model.load_state_dict(torch.load(model_weight_path))
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img))
        predict = torch.softmax(output, dim=0)

        predict_cla = torch.argmax(predict).numpy()

    print(class_indict[str(predict_cla)])
    # print(str(predict.numpy()))
    # print(str(predict.max().numpy()))

    f = open('./test_missinghole.txt','a')
    f.writelines('\n'+ img_name +'\t' + class_indict[str(predict_cla)] + '\t' + str(predict.max().numpy()))
    f.close()
    # plt.show()
