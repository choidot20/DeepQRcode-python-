## 서버 소스
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import copy
from socket import *
from select import *
import threading

port = 12345  # 파이썬 포트 설정
port2 = 20001  # 자바 서버 포트

def run():
    torch.multiprocessing.freeze_support()
    print('loop')
if __name__ == '__main__':
    run()

plt.ion()  # interactive mode

data_transforms = {
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
savePath = './output/testmodel.pth'   ## 결과값 저장용  Path
data_dir = 'hymenoptera_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                              shuffle=True, num_workers=0)
               for x in ['test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['test']}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class_names = image_datasets['test'].classes

'''
def test_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['test']:

            model.eval()  # Set model to evaluate mode

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
        print()

    time_elapsed = time.time() - since
    print('testing complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    ##torch.save(model.state_dict(), savePath)
    model.load_state_dict(best_model_wts)

    return model
'''

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.5)  # pause a bit so that plots are updated


def pred_Code(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                print('predicted: {}'.format(class_names[preds[j]]))

               ##print('메시지를 전송했습니다.')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
        return format(class_names[preds[j]])

def dothing(s):
    while True:
        print('waitng for client..')

    ## host = '1.245.48.126' # 파이썬 / 자바 host

    ##s = socket(AF_INET, SOCK_STREAM)         # 소켓 만듬
    ##s.bind((host, port))        # 소켓 바인드
        f = open('./hymenoptera_data/test\Real/1.jpg','wb') # 저장시킬 경로 및 파일이름
    ##s.listen(1)                 # 한번에 몇명 connect가능 하게 할 것인가


        while True:
            c, addr = s.accept()             # connection 승인
            print("Connected by: ", addr)  # 누구에 의해서 연결됐는가 출력
            l = c.recv(4096)        ## 1024bytes 를 스트림에서 가져옴
            print ("Received: ", repr(l))
            i = 0
            while (l):
        #if i > 1:
                f.write(l)          ## 가져온 데이터를 저장시킴
        #print("Received: ", i, " ", repr(l))
        # if i == 10 :
        #    c.sendall(bytes("OK", encoding='utf8'))
        #    print('ok sent')
        #i = i+1
                l = c.recv(4096)    ## 데이터가 더있으면 가져옴
                f.flush()
            print('fin')
            break;
        f.flush()
        f.close()               # 파일 수정 끝


        model_ft = models.resnet34(pretrained=True)
    # num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(512, 2)

        '''model_ft = models.vgg16_bn(pretrainzed=True)
        #num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(2048, 2)
        '''
        model_ft.load_state_dict(torch.load(savePath))
        model_ft = model_ft.to(device)

        criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

#model_ft = test_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
#                      num_epochs=5)
        sendtext = pred_Code(model_ft, 6)
        plt.pause(0.1)
        plt.close()

        print('sendtext :', sendtext)
        c.close()

        s2 = socket(AF_INET, SOCK_STREAM)         # 소켓 만듬
        s2.connect((host, port2))        # 소켓 바인드
        s2.send(sendtext.encode('ascii'))
        # disconnect
        print('done')
        s2.close()

host = '1.245.48.126' # 파이썬 / 자바 host
s = socket(AF_INET, SOCK_STREAM)         # 소켓 만듬
s.bind((host, port))        # 소켓 바인드
s.listen(1)  # 한번에 몇명 connect가능 하게 할 것인가

t = threading.Thread(target = dothing, args=(s,))
t.start()