import InfoManagement
import BuildModel_VAE
import BuildDataLoader
import torch
from Mnist_Net import test_if_cuda_ok
import numpy as np

# 一些参数
train_index_file = './PIE_DATA/train_index.txt'
log_file_path = './model/log.txt'
info_file_path = './model/info.info'
from torch.autograd import Variable
from torchvision.utils import save_image


GPU_NUM = -1
TOTAL_SIZE = 10262
TRAIN_SIZE = 10000
VALIDATE_SIZE = TOTAL_SIZE - TRAIN_SIZE
BATCH_SIZE = 50
IMG_SIZE = 64
ClassNum = 10
EPOCH = 600
SAVE_STEP = 3

# 查看设备信息，选择是否使用GPU
torch.cuda.set_device(1)
test_if_cuda_ok.test_gpu()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
if torch.cuda.is_available():
    print('USE cuda')
else:
    print('USE CPU')

# 建立 模型文件夹、日志、其他信息
model_folder = InfoManagement.ModelFloder(rebuild=False)
log = InfoManagement.LogFile(log_file_path, renew=False)
info = InfoManagement.InfoFile(info_file_path)
PRE_EPOCH = model_folder.epoch  # 使用之前的epoch
# 继承之前对于训练集和验证集的划分
train_index = []
validata_index = []
if info.data != None:
    [train_index, validata_index] = info.data
else:
    train_index = list(np.random.choice(range(TRAIN_SIZE + VALIDATE_SIZE), TRAIN_SIZE, replace=False))
    validata_index = []
    for x in range(TRAIN_SIZE + VALIDATE_SIZE):
        if x not in train_index:
            validata_index.append(x)
        print(x)
    info.dump([train_index, validata_index])

# 建立 data loader、model、optimizer、loss_fun
[train_loader, validate_loader] = \
        BuildDataLoader.BuildTraining(BATCH_SIZE, IMG_SIZE,
                                      train_index_file ,train_index, validata_index)
[basic_model, loss_fun] = \
    BuildModel_VAE.build_model(ClassNum=ClassNum)
# print(len(validate_loader))
if model_folder.load_model():
    basic_model = model_folder.load_model()
    print('load pre_model')
else:
    print('build new model ')
basic_model = basic_model.to(device)
optimizer, scheduler = \
    BuildModel_VAE.build_optimizer(basic_model)

# 开始训练
loss_list = []

# 计算一批input的准确率
def cal_acc(basic_model, inputs, labels):
    correct = 0
    predicts = []
    with torch.no_grad():
        inputs = inputs.to(device)
        outputs = basic_model(inputs)
        # predict为每一行最大的值的下标
        _, predicts = torch.max(outputs, 1)
        correct += (predicts == labels).sum()
        acc = float(correct) / float(len(labels))
        log.write('acc: %f\n' % acc)
        del inputs, outputs, predicts, acc
    return float(correct)

def train_model(pre_epoch, total_epoch):
    for epoch in range(pre_epoch, total_epoch):
        epoch_loss: float = 0  # total loss in one epoch
        log.write('epoch: %d\n' % epoch)
        train_acc = 0          # accuracy in training set
        count = 0              # show iteration in one epoch
        log.write('lr: %lf\n' % BuildModel_VAE.get_learning_rate(optimizer)[0])
        for data1 in train_loader:
            [inputs, labels] = data1  # use zip to validate model during training
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = basic_model(inputs)
            recon_x, mu, logvar = basic_model(inputs)
            loss = loss_fun(recon_x, inputs, mu, logvar)
            loss.backward()
            optimizer.step()

            # 记录损失函数的值
            loss_list.append(loss)
            # log.write('iter: %d, loss: %f\n' % (count, loss))
            epoch_loss = float(epoch_loss + loss)
            # train_acc = train_acc + cal_acc(basic_model, inputs, labels)
            del inputs, outputs, loss, labels
            del recon_x, mu, logvar
            count = count + 1
        log.write('epoch_loss: %f\n' % epoch_loss)
        scheduler.step()
        # print(BuildModel.get_learning_rate(optimizer))
        # log.write('total_correct: %f\n' % train_acc)

        if epoch % SAVE_STEP == 0:
            basic_model.eval()
            sample = Variable(torch.randn(64, 64)).cuda()
            sample = basic_model.decoder(basic_model.fc2(sample).view(64, 256, 16, 16)).cpu()
            save_image(sample.data.view(64, 1, 64, 64),
                       'result/sample_' + str(epoch) + '.png')
            print('img saved')
            # save model
            model_folder.save_model(basic_model)



if __name__ == '__main__':
    train_model(PRE_EPOCH, EPOCH)
