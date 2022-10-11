from PIL import Image
import os
import torch
import torch.utils.data as data

def read_masks(seg):
    masks = torch.zeros(seg.size()[1:]).long()
    mask = 4 * seg[0, :, :] + 2 * seg[1, :, :] + seg[2, :, :]
    masks[mask == 3] = 0  # (Cyan: 011) Urban land 
    masks[mask == 6] = 1  # (Yellow: 110) Agriculture land 
    masks[mask == 5] = 2  # (Purple: 101) Rangeland 
    masks[mask == 2] = 3  # (Green: 010) Forest land 
    masks[mask == 1] = 4  # (Blue: 001) Water 
    masks[mask == 7] = 5  # (White: 111) Barren land 
    masks[mask == 0] = 6  # (Black: 000) Unknown
    return masks

class Dataset(data.Dataset):
    
    def __init__(self, img_dir, transform, target_transform):

        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.data = []
        self.label = []  
        self.fname = []
       
        # read sat images
        for sat in [x for x in os.listdir(img_dir) if 'sat' in x]:
            img_path = os.path.join(img_dir, sat)
            img = self.transform(Image.open(img_path).convert('RGB'))
            self.data.append(img)
            self.fname.append(sat)
        
        # read mask images
        for i in range(len(self.fname)):
            mask = self.fname[i].replace('sat.jpg', 'mask.png')
            img_path = os.path.join(img_dir, mask)
            if os.path.isfile(img_path):
                img = self.target_transform(Image.open(img_path).convert('RGB')).long()
                label = read_masks(img)
                self.label.append(label)
        
        self.data = torch.stack(self.data)
        if len(self.label) > 0:
            self.label = torch.stack(self.label)
        
    def __getitem__(self, index):
        img = self.data[index]
        fname = self.fname[index]
        if len(self.label) > 0:
            label = self.label[index]
            return img, fname, label
        return img, fname

    def __len__(self):
        return len(self.data)

"""
import pdb
from torchvision import transforms
img_size = 64
transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
target_transform = transforms.Compose([
    transforms.Resize(img_size, interpolation=0),
    transforms.ToTensor(),
])
dataset = Dataset("data/p2_data/train", transform, target_transform)
#dataset = Dataset("viz_data", transform, target_transform)
data_loader = data.DataLoader(dataset, shuffle=True, drop_last=False, pin_memory=True, batch_size=128)

for (step, value) in enumerate(data_loader):

    image = value[0].cuda()
    if len(value) > 2:
        target = value[2].cuda(async=True)
    pdb.set_trace()
"""