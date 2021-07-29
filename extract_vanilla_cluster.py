import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from utils.siCluster_utils import *
from utils.parameters import *
import glob
import shutil
import copy
import csv
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

urban_num = 2432
urban_cluster_num = 2
rural_cluster_num = 4
def extract_inhabited_cluster(args):
    convnet = models.resnet18(pretrained=True)
    convnet = torch.nn.DataParallel(convnet)    
    print("laoded")
    ckpt = torch.load('./checkpoint/ckpt_vanilla_cluster_lao_z14_50_pretrained.t7')
    convnet.load_state_dict(ckpt, strict = False)
    convnet.module.fc = nn.Sequential()
    convnet.cuda()
    cluster_transform =transforms.Compose([
                      transforms.Resize(256),
                      transforms.CenterCrop(224),
                      transforms.RandomGrayscale(p=1.0),
                      transforms.ToTensor(),
                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])    
    
    clusterset = GPSDataset('./meta_data/inhabited_metadata.csv', './unlabelled_images', cluster_transform)
    clusterloader = torch.utils.data.DataLoader(clusterset, batch_size=128, shuffle=False, num_workers=2)
    features = compute_features(clusterloader, convnet, len(clusterset), 128) 
    
    '''
    for i in range(2,31):
     #   features = compute_features(clusterloader, convnet, len(clusterset), 128) 
        print(len(features))
        kmeans = KMeans(n_clusters=i).fit(features[urban_num:])
        p_label = kmeans.labels_
        score = silhouette_score(features[urban_num:], p_label, metric="euclidean")
        #score = kmeans.inertia_
        print("score of cluster {} in rural is {}".format(i, score))
      #  features = compute_features(clusterloader, convnet, len(clusterset), 128) 
        kmeans = KMeans(n_clusters=i).fit(features[:urban_num])
        p_label = kmeans.labels_
        score = silhouette_score(features[:urban_num], p_label, metric="euclidean")
        #score = kmeans.inertia_
        print("score of cluster {} in city is {}".format(i, score))
    '''
    kmeans = KMeans(n_clusters=urban_cluster_num).fit(features[:urban_num])
    p_label1 = kmeans.labels_
    kmeans = KMeans(n_clusters=rural_cluster_num).fit(features[urban_num:])
    p_label2= kmeans.labels_

    labels1 = p_label1.tolist()
    labels2 = p_label2.tolist()
    f = open('./meta_data/inhabited_metadata.csv', 'r', encoding='utf-8')
    images = []
    rdr = csv.reader(f)
    for line in rdr:
        images.append(line[0])
    f.close()
    images.pop(0)    
    rural_cluster = []
    print(len(images))
    for i in range(0, urban_num):
        rural_cluster.append([images[i], labels1[i]])
    for j in range(urban_num, len(images)):
        rural_cluster.append([images[j], labels2[j-urban_num]+urban_cluster_num])
    return rural_cluster



def extract_nature_cluster(args):
    f = open('./meta_data/inhabited_metadata.csv', 'r', encoding='utf-8')
    images = []
    rdr = csv.reader(f)
    for line in rdr:
        images.append(line[0])
    f.close()
    images.pop(0)    
    nature_cluster = []
    cnum = urban_cluster_num + rural_cluster_num
    for i in range(0, len(images)):
        nature_cluster.append([images[i], cnum])
        
    return nature_cluster



def main(args):
    # make cluster directory
    print("main")
    inhabited_cluster = extract_inhabited_cluster(args)
    nature_cluster = extract_nature_cluster(args)
    total_cluster = inhabited_cluster + nature_cluster
    
    cluster_dir = './cluster_LAO/'
    if not os.path.exists(cluster_dir):
        os.makedirs(cluster_dir)
        for i in range(0, urban_cluster_num + rural_cluster_num + 1):
            os.makedirs(cluster_dir + str(i))

    
    for img_info in total_cluster:
        cur_dir = './unlabelled_images/' + img_info[0]
        new_dir = cluster_dir + str(img_info[1]+1)+'/'+img_info[0].split('/')[1]    ## JAEYEON 0726: str(img_info[1]) --> str(img_info[1]+1) ; img_info[1]으로 하면 0부터 시작인데 0으로 하면 폴더명으로 인식 안됨 따라서 하나 올려서 1부터 시작 

        # print(cur_dir, new_dir)
        # ./unlabelled_images/1/7296_12916.png ./cluster_LAO/0/7296_12916.png
        # NotADirectoryError: [Errno 20] Not a directory: './cluster_LAO/0/7296_12916.png'
        
        #new_file = cluster_dir + str(img_info[1])+'/'+img_info[0].split('/')[1]

        shutil.copy(cur_dir, new_dir)
        #os.rename(cluster_dir + str(img_info[1])+'/'+img_info[0][-14:], new_file)
'''
    file_list = glob.glob("./{}/*/*.png".format(args.cluster_dir))
    grid_dir = cluster_dir + args.grid
    f = open(grid_dir, 'w', encoding='utf-8')
    wr = csv.writer(f)
    wr.writerow(['y_x', 'cluster_id'])
    
    for file in file_list:
        file_split = file.split("/")
        folder_name = file_split[2]
        file_name = file_split[-1].split(".")[0]
        wr.writerow([file_name, folder_name])
    f.close()
    
'''
if __name__ == "__main__":
    args = extract_cluster_parser()
    main(args)    
    