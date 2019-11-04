import numpy as np
from torch.utils.data import Dataset
import torch
from skimage import io, transform
import os
import json
from torch.utils.data import DataLoader


class Data(Dataset):

    def __init__(self, root_dir, mask_type, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.image_dirs = os.listdir(root_dir + '/image')
        self.mask_type = mask_type
        self.transform = transform

    def __len__(self):
        return len(self.image_dirs)

    def __getitem__(self, idx):

        img_path = os.path.join(self.root_dir + '/image', self.image_dirs[idx])
        image = self.normalize_img(io.imread(img_path))
        image = image.transpose((2, 0, 1))  # C*H*W
        image = torch.from_numpy(image)
        with open(self.root_dir + '/desease.json') as f:
            desease = json.load(f)

        if self.mask_type:
            mask_path = os.path.join(self.root_dir + '/' + self.mask_type, self.image_dirs[idx][0:-(len('jpg'))] + 'png')
            mask = self.normalize_mask(io.imread(mask_path, as_gray=True))
            mask = np.expand_dims(mask, axis=0)
            mask = torch.from_numpy(mask)

            sample = {'image': image, 'mask': mask, 'image_name': self.image_dirs[idx],
                      'desease': desease[self.image_dirs[idx]]}
        else:
            sample = {'image': image, 'image_name': self.image_dirs[idx],
                      'desease': desease[self.image_dirs[idx]]}
            # 0 - melanocytic benign; 1 - melanoma; -1 - other

        if self.transform:
            sample = self.transform(sample)

        return sample

    def normalize_img(self, img):
        img = img - np.mean(img)
        img = img / np.std(img)
        # Transformação linear que coloca os valores entre 0 e 1
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        return img

    def normalize_mask(self, mask):

        if np.max(mask) != 0:
            mask = mask/np.max(mask)

        return mask


def get_data_loaders(save_path, batch_size, mask_type):
    data_train = Data(root_dir=save_path + 'data/train', mask_type=mask_type)
    data_validation = Data(root_dir=save_path + 'data/validation', mask_type=mask_type)
    data_test = Data(root_dir=save_path + 'data/test', mask_type=mask_type)

    dataloader_train = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    dataloader_validation = DataLoader(data_validation, batch_size=batch_size)
    dataloader_test = DataLoader(data_test, batch_size=batch_size)

    return dataloader_train, dataloader_validation, dataloader_test


# Create desease Dict
def create_desease_dict(heavy_data_path, dataloader_path):

    data = Data(dataloader_path, mask_type='mask')
    desease = [[], []]
    melanocytic_benign_dir = os.listdir(heavy_data_path + '/Benign')
    melanocytic_melanoma_dir = os.listdir(heavy_data_path + '/Melanoma')

    for i, sample in enumerate(data):
        print(i)
        desease[0].append(sample['image_name'])
        if sample['image_name'] in melanocytic_benign_dir:
            desease[1].append(0)
        elif sample['image_name'] in melanocytic_melanoma_dir:
            desease[1].append(1)
        else:
            desease[1].append(-1)

    desease = dict(zip(desease[0], desease[1]))

    json_file = json.dumps(desease)
    f = open("desease.json", "w")
    f.write(json_file)
    f.close()


#create_desease_dict(heavy_data_path='D:\ISIC_Skin_Cancer\ISIC-2017_Validation_Data',
                   # dataloader_path='C:/Users/pedro/PycharmProjects/ISIC_skin_cancer/data/validation')
