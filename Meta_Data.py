import os
import csv
origin = './unlabelled_images'


f = open('./meta_data/lao_z14_metadata.csv', 'w', encoding='utf-8', newline='')
wr = csv.writer(f)
wr.writerow(['img_name'])
dirs = os.listdir(origin)
for d in dirs:
    imgs = os.listdir(origin+"/"+d)
    for img in imgs:
        wr.writerow([d+'/'+img])

    ''' 
    years = os.listdir(origin)    
    for year in years:
        root=origin+year
        dir_list = os.listdir(root)
        for i in range(0, len(dir_list)):
            if len(dir_list[i])<4:
                dir_list[i] = int(dir_list[i])
        for dir_num in dir_list:
            cur_dir = root + '/' + str(dir_num)
            cur_list = os.listdir(cur_dir)
            for name in cur_list:
                img_name =  year+'/'+str(dir_num) + '/' + name
                wr.writerow([img_name])
    f.close()
    

    cur_dir = root +str(i)
    cur_list = os.listdir(cur_dir)
    for name in cur_list:
        img_name =   name
        wr.writerow([img_name])


f = open('./meta_data/sk_z14_metadata_119.csv', 'w', encoding='utf-8', newline='')
wr = csv.writer(f)
wr.writerow(['img_name'])

years = os.listdir(origin)    
for year in years:
    root=origin+year
    dir_list = os.listdir(root)
    for i in range(0, len(dir_list)):
        if len(dir_list[i])<4:
            dir_list[i] = int(dir_list[i])
    for dir_num in dir_list:
        cur_dir = root + '/' + str(dir_num)
        cur_list = os.listdir(cur_dir)
        for name in cur_list:
            img_name =  year+'/'+str(dir_num) + '/' + name
            wr.writerow([img_name])
f.close()

'''