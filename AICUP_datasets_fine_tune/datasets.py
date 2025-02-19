import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import defaultdict
from PIL import Image
import torch
from torchvision import transforms
def parse_image(input_path):
    image_paths = []
    vehicle_ids = []
    class_map = dict()
    cur_class = 0
    for image_file in os.listdir(input_path):
        image_paths.append(image_file)
        vehicle_id = int(image_file.split('_')[-1].split('.')[0])
        vehicle_ids.append(vehicle_id)

        if vehicle_id not in class_map:
            class_map[vehicle_id] = cur_class
            cur_class +=1
    return image_paths,vehicle_ids,class_map

def build_class_tree(img_paths,vehicle_ids,class_map):
    class_tree = defaultdict(list)

    for id,path in zip(vehicle_ids,img_paths):
        class_tree[class_map[id]].append(path)
    return class_tree


class AICUPTrain(Dataset):
    def __init__(self,img_paths,vehicle_ids,class_map,transform,root):
        self.img_paths = np.array(img_paths)
        self.vehicle_ids = np.array(vehicle_ids)
        self.class_map = class_map
        self.transform = transform
        self.root = root
        self.class_tree = build_class_tree(img_paths,vehicle_ids,class_map)


    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, index):
        anchor_img = Image.open(os.path.join(self.root,'train',self.img_paths[index]))
        positive_img_class = self.class_map[self.vehicle_ids[index]]
        # positive_img_path = np.random.choice(self.class_tree[positive_img_class])
        
        # Filter out the anchor image from the candidates when sampling the positive image

        positive_img_candidates = [p for p in self.class_tree[positive_img_class] if p != self.img_paths[index]]
        
        if not positive_img_candidates:
            positive_img_path = self.img_paths[index]  # Use the anchor image itself as a fallback
        else:
            positive_img_path = np.random.choice(positive_img_candidates)

        positive_img = Image.open(os.path.join(self.root,'train',positive_img_path))

        negative_img_class = self.random_number_except(0,len(self.class_map),positive_img_class)
        negative_img_path = np.random.choice(self.class_tree[negative_img_class])
        negative_img = Image.open(os.path.join(self.root,'train',negative_img_path))
        
        if self.transform is not None:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)
        
        return torch.stack((anchor_img,positive_img,negative_img),dim=0),torch.tensor([positive_img_class,positive_img_class,negative_img_class])

    
    def random_number_except(self,start,end,exclude):
        numbers = list(range(start,end))
        numbers.remove(exclude)
        return np.random.choice(numbers)

class AICUPValid(Dataset):
    def __init__(self,img_paths,vehicle_ids,transform,root):
        self.img_paths = np.array(img_paths)
        self.vehicle_ids = np.array(vehicle_ids)
        self.transform = transform
        self.root = root

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = Image.open(os.path.join(self.root,'valid',img_path))
        vehicle_id = self.vehicle_ids[index]
        if self.transform is not None:
            img = self.transform(img)
        return img_path,img,vehicle_id
    

    # def __iter__(self):
    #     for i in range(len(self)):
    #         img = Image.open(os.path.join(self.root, 'valid', self.img_paths[i]))
    #         if self.transform is not None:
    #             img = self.transform(img)
    #         yield self.img_paths[i], img

# class AICUPSimilarity(Dataset):
#     def __init__(self,img_paths,vehicle_ids,class_map,transform,root):
#         self.img_paths = np.array(img_paths)
#         self.vehicle_ids = np.array(vehicle_ids)
#         self.class_map = class_map
#         self.transform = transform
#         self.root = root
#         self.class_tree = build_class_tree(img_paths,vehicle_ids,class_map)

#     def __len__(self):
#         return len(self.img_paths)
    
        
#     def __getitem__(self, index):
#         anchor_img = Image.open(os.path.join(self.root,'train',self.img_paths[index]))
#         positive_img_class = self.class_map[self.vehicle_ids[index]]
#         # positive_img_path = np.random.choice(self.class_tree[positive_img_class])
        
#         # Filter out the anchor image from the candidates when sampling the positive image

#         positive_img_candidates = [p for p in self.class_tree[positive_img_class] if p != self.img_paths[index]]
        
#         if not positive_img_candidates:
#             positive_img_path = self.img_paths[index]  # Use the anchor image itself as a fallback
#         else:
#             positive_img_path = np.random.choice(positive_img_candidates)

#         positive_img = Image.open(os.path.join(self.root,'train',positive_img_path))

#         negative_img_class = self.random_number_except(0,len(self.class_map),positive_img_class)
#         negative_img_path = np.random.choice(self.class_tree[negative_img_class])
#         negative_img = Image.open(os.path.join(self.root,'train',negative_img_path))
        
#         if self.transform is not None:
#             anchor_img = self.transform(anchor_img)
#             positive_img = self.transform(positive_img)
#             negative_img = self.transform(negative_img)
        
#         return torch.stack((anchor_img,positive_img,negative_img),dim=0),torch.tensor([positive_img_class,positive_img_class,negative_img_class])

        
def get_AICUP_train(AICUP_path,num_workers,batch_size, transform, drop_last=False, shuffle=False):
    img_paths, vehicle_ids, class_map = parse_image(os.path.join(AICUP_path, 'train'))
    dataset = AICUPTrain(img_paths, vehicle_ids, class_map, transform, AICUP_path)
    # print(class_map)
    # print(len(class_map))
    total_class = len(class_map)
    return DataLoader(dataset, num_workers=num_workers,batch_size=batch_size, drop_last=drop_last, shuffle=shuffle),total_class

def get_AICUP_valid(AICUP_path,num_workers,batch_size,transform):
    img_paths, vehicle_ids, class_map = parse_image(os.path.join(AICUP_path, 'valid'))
    gallery_dataset = AICUPValid(img_paths,vehicle_ids,transform,AICUP_path)

    class_tree = build_class_tree(img_paths,vehicle_ids,class_map)

    query_paths = []

    for _,paths in class_tree.items():
        np.random.shuffle(paths)
        num_query = max(1,len(paths)//2)
        query_paths.extend(paths[:num_query])
    query_vehicle_ids = []
    for path in query_paths:
        query_vehicle_id = int(path.split('_')[-1].split('.')[0])
        query_vehicle_ids.append(query_vehicle_id)

    query_dataset = AICUPValid(query_paths, vehicle_ids=query_vehicle_ids, transform=transform, root=AICUP_path)

    return DataLoader(gallery_dataset,num_workers=num_workers,batch_size=batch_size),DataLoader(query_dataset, num_workers=num_workers, batch_size=batch_size)




if __name__ == '__main__':
    path = '/home/eddy/Desktop/cropped_image'
    img_paths, vehicle_ids, class_map = parse_image(os.path.join(path, 'train'))
    class_tree = build_class_tree(img_paths,vehicle_ids,class_map)

    # Define a simple transform
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    # dataset = AICUPTrain(img_paths, vehicle_ids, class_map, transform, root)
    # print("Dataset Length:", len(dataset))
    # dataloader = DataLoader(dataset, batch_size=2, shuffle=True)



    # dataloader,_ = get_AICUP_train(path,4,2,transform)
    # for i, (images, labels) in enumerate(dataloader):
    #     print(f"Batch {i + 1}")
    #     print("Images Shape:", images.shape)  # Should be [batch_size, 3, C, H, W]
    #     print("Labels:", labels)             # Should be [batch_size, 3]

    #     # Display one of the images (optional, for debugging purposes)
    #     if i == 0:
    #         transforms.ToPILImage()(images[0][0]).show()

    #     if i > 1:  # Test only a couple of batches
    #         break

    # gallery_dataloader, query_dataloade = get_AICUP_valid(path, num_workers=4, batch_size=2, transform=transform)
    # print(f"query_paths: {query_paths}")
    # print(f'query_paths_len: {len(query_paths)}')
    # print(f"Loaded Validation Dataset with {len(dataloader.dataset)} images.")

    # for i, (img_paths, imgs,id) in enumerate(query_dataloade):
    #     print(f"Batch {i + 1}")
    #     print(f"Image Paths: {img_paths}")  # List of file paths in this batch
    #     print(f"Images Shape: {imgs.shape}")  # [batch_size, C, H, W]
    #     print(f"image ID: {id}")
    #     # Display one of the images (optional, for debugging purposes)
    #     if i == 0:
    #         print("Displaying the first image for debugging...")
    #         transforms.ToPILImage()(imgs[0]).show()

    #     if i > 1:  # Test only a couple of batches
    #         break