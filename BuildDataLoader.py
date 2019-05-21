from torch.utils import data
from torchvision import transforms
from test_if_cuda_ok import *
import DataPrepare

# INPUT: total_size, train_size, batch_size, img_size
# OUTPUT: train_loader, validate_loader
def BuildTraining(batch_size, img_size,
                  fea_label_file, train_index, validata_index):
    BATCH_SIZE = batch_size
    IMG_SIZE = img_size

    train_set = DataPrepare.DefaultDataset(
        fea_label_file, load_index=train_index)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    validate_set = DataPrepare.DefaultDataset(
        fea_label_file, load_index=validata_index)
    validate_loader = torch.utils.data.DataLoader(validate_set, batch_size=BATCH_SIZE, shuffle=True)
    print('Data load Success')

    return train_loader, validate_loader

# INPUT: total_size, train_size, batch_size, img_size
# OUTPUT: train_loader, validate_loader
def BuildTesting(batch_size, img_size, fea_label_file):
    BATCH_SIZE = batch_size
    test_set = DataPrepare.DefaultDataset(
        fea_label_file)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    print('Data load Success')

    return test_loader