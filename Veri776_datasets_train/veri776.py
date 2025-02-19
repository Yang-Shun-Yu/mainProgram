from torch.utils.data import Dataset, DataLoader
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
from collections import defaultdict
import torch

import os

# Class----------------------------------------------------------------------------|
# veri776 train 
class Veri776Train(Dataset):
    def __init__(self,img_paths,vehicle_ids,class_map,transform, veri776_root):
        self.img_paths = np.array(img_paths)
        self.vehicle_ids = np.array(vehicle_ids)
        self.class_map = class_map
        self.transform = transform
        self.class_tree = self.build_class_tree(img_paths,vehicle_ids,class_map)
        self.veri776_root = veri776_root

    def build_class_tree(self,img_paths,vehicle_ids,class_map):
        class_tree = defaultdict(list)
        for id,path in zip(vehicle_ids,img_paths):
            class_tree[class_map[id]].append(path)
        return class_tree

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self,index):
        anchor_img = Image.open(os.path.join(self.veri776_root,'image_train',self.img_paths[index]))
        
        positive_img_class = self.class_map[self.vehicle_ids[index]]

        positive_img_candidates = [p for p in self.class_tree[positive_img_class] if p != self.img_paths[index]]

        if not positive_img_candidates:
            positive_img_path = self.img_paths[index]  # Use the anchor image itself as a fallback
        else:
            positive_img_path = np.random.choice(positive_img_candidates)

        # positive_img_path = np.random.choice(self.class_tree[positive_img_class])
        
        
        positive_img = Image.open(os.path.join(self.veri776_root,'image_train',positive_img_path))

        negative_img_class = self.random_number_except(0,len(self.class_map),positive_img_class)
        negative_img_path = np.random.choice(self.class_tree[negative_img_class])

        
        negative_img = Image.open(os.path.join(self.veri776_root,'image_train',negative_img_path))

        if self.transform is not None:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return torch.stack((anchor_img,positive_img,negative_img),dim=0),torch.tensor([positive_img_class,positive_img_class,negative_img_class])

    def random_number_except(self,start,end,exclude):
        numbers = list(range(start,end))
        numbers.remove(exclude)
        return np.random.choice(numbers)

# veri776 test
class Veri776Test:
    def __init__(self, img_file_names, vehicle_ids, transform, veri776_root):
        self.img_file_names = np.array(img_file_names) # for indexing in __getitem__
        self.vehicle_ids = vehicle_ids
        self.transform = transform
        self.veri776_root = veri776_root


    def __len__(self):
        return len(self.img_file_names)


    # def __iter__(self):
    #     for i in range(len(self)):
    #         img = Image.open(os.path.join(self.veri776_root, 'image_test', self.img_file_names[i]))

    #         if self.transform is not None:
    #             img = self.transform(img)


    #         yield self.img_file_names[i], img

    def __getitem__(self,index):
        image_name = self.img_file_names[index]
        img = Image.open(os.path.join(self.veri776_root,'image_test',image_name))
        if self.transform is not None:
            img = self.transform(img)
        return image_name,img



# class Veri776Test(Dataset):
#     def __init__(self, img_file_names, vehicle_ids, transform, veri776_root):
#         self.img_file_names = np.array(img_file_names)  # For indexing in __getitem__
#         self.vehicle_ids = vehicle_ids
#         self.transform = transform
#         self.veri776_root = veri776_root

#     def __len__(self):
#         return len(self.img_file_names)

#     def __getitem__(self, index):
#         img = Image.open(os.path.join(self.veri776_root, 'image_test', self.img_file_names[index]))
#         vehicle_id = self.vehicle_ids[index]

#         if self.transform is not None:
#             img = self.transform(img)

#         return img, vehicle_id  # Returning both the image and its corresponding vehicle ID
# Function-------------------------------------------------------------------------|
def parse_xml(xml_path):
    with open(xml_path) as f:
        et = ET.fromstring(f.read())
        image_paths = []
        vehicle_ids = []
        class_map = dict()
        cur_class = 0

        for item in et.iter('Item'):
            image_paths.append(item.attrib['imageName'])
            vehicle_id = int(item.attrib['vehicleID'])
            vehicle_ids.append(vehicle_id)

            if vehicle_id not in class_map:
                class_map[vehicle_id] = cur_class
                cur_class+=1
        return image_paths,vehicle_ids,class_map

def get_veri776_test(veri_776_path, num_workers, batch_size, transform):
    img_file_names, vehicle_ids, _ = parse_xml(os.path.join(veri_776_path, 'test_label.xml'))
    test_set = Veri776Test(img_file_names, vehicle_ids, transform, veri_776_path)
    # return Veri776Test(img_file_names, vehicle_ids, transform, veri_776_path)
    return DataLoader(test_set, num_workers=num_workers, batch_size=batch_size, shuffle=False)

def get_veri776_train(veri776_path, num_workers, batch_size, transform, drop_last=False, shuffle=False):
    
    img_paths, vehicle_ids, class_map = parse_xml(os.path.join(veri776_path, 'train_label.xml'))
    # img_paths = [os.path.join(veri776_path, 'image_train', path) for path in img_paths]
    train_set = Veri776Train(img_paths, vehicle_ids, class_map, transform,veri776_path)


    return DataLoader(train_set, num_workers=num_workers, batch_size=batch_size, drop_last=drop_last, shuffle=shuffle)