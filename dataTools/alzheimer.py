import numpy as np
import os
from PIL import Image
from torchvision import transforms

def get_ad(base_dir):
    x_tr, y_tr = get_x_y(base_dir, 'train')
    x_te, y_te = get_x_y(base_dir, 'test')
    return (x_tr, y_tr), (x_te, y_te)

def get_x_y(base_dir, txt='train'):
    x = []
    y = []
    txt_path = os.path.join(base_dir, txt+'.txt')
    with open(txt_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            group, file_name, label = line.split(',')
            file_path = os.path.join(base_dir, txt, group, file_name)
            x.append(file_path)
            y.append(int(label))
    return np.array(x), np.array(y)

transform = transforms.Compose(
    [transforms.Resize((224, 224)), transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])]
)

def ad_transform(file_path):
    img = Image.open(file_path)
    img = img.convert('RGB')
    return transform(img)