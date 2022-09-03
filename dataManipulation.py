import os
import rarfile
import torch.utils.data
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
import random
import shutil

class Dataset(torch.utils.data.Dataset):
    def __init__(self, path):
        imgs = []
        gf_l = os.listdir(path + 'grasp/')
        ngf_l = os.listdir(path + 'no_grasp/')
        for gf in gf_l:
            imgs.append((path + 'grasp/' + gf, 1))
        for ngf in ngf_l:
            imgs.append((path + 'no_grasp/' + ngf, 0))
        self.imgs = imgs
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, item):
        img, label = self.imgs[item]
        img = Image.open(img)
        img = transforms.ToTensor()(img)
        return img, label

def file_manipulation():
    for i in range(1, 11):
        if os.path.exists(f'./data/{i}'):
            pass
        else :
            rfile = rarfile.RarFile(f'./data/{i}.rar')
            rfile.extractall("./data/")
            rfile.close()
            if i == 2:
                os.rename('./data/1/grasp/6/New folder', './data/1/grasp/6/images')
                os.rename('./data/1/grasp/6/images/no grasp', './data/1/grasp/6/images/no_grasp')
    print("unrar completion")

    if os.path.exists('./data/grasp/') and os.path.exists('./data/no_grasp/'):
        pass
    else :
        os.makedirs("./data/grasp/")
        os.makedirs("./data/no_grasp/")
        for i in range(1, 11):
            dname_list = os.listdir(f'./data/{i}/grasp/')
            for dn in dname_list:
                # ./1/grasp/6下有错误的文件名
                g_fn_l = os.listdir(f'./data/{i}/grasp/' + dn + '/images/grasp/g/')
                ng_fn_l = os.listdir(f'./data/{i}/grasp/' + dn + '/images/no_grasp/g/')
                for g_fn in g_fn_l:
                    os.rename(f'./data/{i}/grasp/' + dn + '/images/grasp/g/' + g_fn,
                              f'./data/{i}/grasp/' + dn + f'/images/grasp/g/{i}_{dn}' + g_fn)
                    shutil.copy(f'./data/{i}/grasp/' + dn + f'/images/grasp/g/{i}_{dn}' + g_fn,
                                './data/grasp/')
                for ng_fn in ng_fn_l:
                    os.rename(f'./data/{i}/grasp/' + dn + '/images/no_grasp/g/' + ng_fn,
                              f'./data/{i}/grasp/' + dn + f'/images/no_grasp/g/n{i}_{dn}' + ng_fn)
                    shutil.copy(f'./data/{i}/grasp/' + dn + f'/images/no_grasp/g/n{i}_{dn}' + ng_fn,
                                './data/no_grasp/')
    print("rename completion")

def split_dataset(percentage):
    if os.path.exists('./data/train/'):
        trfg = os.listdir('./data/train/grasp')
        tefg = os.listdir('./data/test/grasp')
        trfng = os.listdir('./data/train/no_grasp')
        tefng = os.listdir('./data/test/no_grasp')
        for t in trfg:
            os.remove('./data/train/grasp/' + t)
        for t in tefg:
            os.remove('./data/test/grasp/' + t)
        for t in trfng:
            os.remove('./data/train/no_grasp/' + t)
        for t in tefng:
            os.remove('./data/test/no_grasp/' + t)
    else :
        os.makedirs('./data/train/grasp/')
        os.makedirs('./data/train/no_grasp/')
        os.makedirs('./data/test/grasp/')
        os.makedirs('./data/test/no_grasp/')
    g_fn_l = os.listdir('./data/grasp/')
    ng_fn_l = os.listdir('./data/no_grasp/')
    print(f'共有正样本{len(g_fn_l) * 2}', f'共有负样本{len(ng_fn_l)}')
    random.shuffle(g_fn_l)
    random.shuffle(ng_fn_l)
    end_idx = int(len(g_fn_l) * (1 - percentage))
    for i in range(end_idx):
        shutil.copy('./data/grasp/' + g_fn_l[i], './data/train/grasp/')
        os.rename('./data/train/grasp/' + g_fn_l[i], './data/train/grasp/c' + g_fn_l[i])
        shutil.copy('./data/grasp/' + g_fn_l[i], './data/train/grasp/')
    for i in range(end_idx, len(g_fn_l)):
        shutil.copy('./data/grasp/' + g_fn_l[i], './data/test/grasp/')
    end_idx = int(len(ng_fn_l) * (1 - percentage))
    for i in range(end_idx):
        shutil.copy('./data/no_grasp/' + ng_fn_l[i], './data/train/no_grasp/')
    for i in range(end_idx, len(ng_fn_l)):
        shutil.copy('./data/no_grasp/' + ng_fn_l[i], './data/test/no_grasp/')
    print("split dataset has completion")

def load_data(percentage, has_unrar = True, need_split = False):
    if not has_unrar: file_manipulation()
    if need_split: split_dataset(percentage)
    train = Dataset('./data/train/')
    test = Dataset('./data/test/')
    print(f'训练集数量{train.__len__()}测试集数量{test.__len__()}')
    return (DataLoader(train, batch_size = 2, shuffle = True, num_workers = 0),\
           DataLoader(test, batch_size = 2, shuffle = True, num_workers = 0))


