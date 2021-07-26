import argparse
import os
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

class GPSNDataset(Dataset):
    def __init__(self, metadata, root_dir,transform1=None):
        self.metadata = pd.read_csv(metadata).values
        self.root_dir = root_dir
        self.transform = transform1

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.metadata[idx][0])
        image =  Image.open(img_name)
        
        if self.transform:
            try:
                image = self.transform(image)
            except OSError:
                print(img_name)
                
        return image, idx , self.metadata[idx][0]

def main():
    path = "./checkpoint/lao_resnet18_200.ckpt"
    model = models.resnet18(pretrained=False)
    #model = nn.DataParallel(model)
    model.fc =  nn.Sequential(nn.Linear(512, 3), nn.Softmax())
    model.load_state_dict(torch.load(path)['state_dict'], strict = True)
    model.cuda()
    cudnn.benchmark = True
    
    test_transform =transforms.Compose([
                      transforms.Resize(256),
                      transforms.CenterCrop(224),
                      transforms.RandomGrayscale(p=1.0),
                      transforms.ToTensor(),
                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    testset = GPSNDataset('./meta_data/lao_z14_metadata.csv', './unlabelled_images', test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=4)
    
    
        
    model.eval()
    label = []
    for batch_idx, (inputs, _, name) in enumerate(testloader):
        #print(name)
        #print(inputs)
        inputs = inputs.cuda()
        logits = torch.argmax(model(inputs), dim = 1)
        label.extend(logits.tolist())
        
        
    print("Eval Finish")
    
    f = open('./meta_data/lao_z14_metadata.csv', 'r', encoding='utf-8')
    images = []
    import csv
    rdr = csv.reader(f)
    for line in rdr:
        images.append(line[0])
    f.close()
    print(images)
    images.pop(0)
    f1 = open('./meta_data/meta_city_lao_z14.csv', 'w', encoding='utf-8')
    f2 = open('./meta_data/meta_rural_lao_z14.csv', 'w', encoding='utf-8')
    f3 = open('./meta_data/meta_nature_lao_z14.csv', 'w', encoding='utf-8')
    wr1 = csv.writer(f1)
    wr1.writerow(['img_name','label'])
    wr2 = csv.writer(f2)
    wr2.writerow(['img_name','label'])
    wr3 = csv.writer(f3)
    wr3.writerow(['img_name','label'])
    
    for i in range(0, len(images)):
        if label[i] == 0:
            wr1.writerow([images[i], label[i]])
        elif label[i] == 1:
            wr2.writerow([images[i], label[i]])
        elif label[i] == 2:
            wr3.writerow([images[i], label[i]])
        if i % 10000 == 0:
            print(i)
            
    f1.close()
    f2.close()
    f3.close()
    
if __name__ == "__main__":
    main()   
