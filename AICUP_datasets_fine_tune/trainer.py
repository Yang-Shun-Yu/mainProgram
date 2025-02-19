from tqdm import tqdm
import torch
import os
from einops import rearrange
import pandas as pd
from model import make_model
import Transforms
from collections import defaultdict
class AICUPTrainer:
    def __init__(self,net,backbone,ce_loss_fn=None,triplet_loss_fn=None,center_loss_fn=None,optimizer=None, optimizer_center=None,lr_scheduler=None,device='cpu'):

        self.device = device
        self.net = net.to(device)

        print(f'Training on {self.device}')

        self.ce_loss_fn = ce_loss_fn
        self.triplet_loss_fn = triplet_loss_fn
        self.center_loss_fn = center_loss_fn

        self.optimizer = optimizer
        self.optimizer_center = optimizer_center
        self.lr_scheduler = lr_scheduler
        self.backbone = backbone

        self.metrics = []

    def fit(self,train_dataloader,gallery_dataloader,query_dataloader,epochs,save_dir,early_stopping,check_init=False,similarity='cosine'):

        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if check_init:
            print('check init')
            self.valid(
                gallery_dataloader=gallery_dataloader,
                query_dataloader=query_dataloader,
                save_dir=save_dir,
                similarity=similarity

            )
        best_hit = 0
        patience = 0

        for epoch in range(epochs):
            triplet_loss,ce_loss,center_loss = self.train(train_dataloader=train_dataloader,current_epoch=epoch,total_epochs=epochs)
            hits,total_queries = self.valid(
                gallery_dataloader=gallery_dataloader,
                query_dataloader=query_dataloader,
                save_dir=save_dir,
                similarity=similarity
            )
            if hits > best_hit:
                print(f'New best model found!')
                print(f'hits: {hits}')
                print('saving the model------')
                best_hit = hits
                patience=0

                if self.center_loss_fn is not None:
                    torch.save(self.net.state_dict(),os.path.join(save_dir,f'{self.backbone}_center_loss_best.pth'))
                else:
                    torch.save(self.net.state_dict(),os.path.join(save_dir,f'{self.backbone}_best.pth'))
            
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.metrics.append({
                'Epoch':epoch+1,
                'Triplet Loss': triplet_loss,
                'CE Loss': ce_loss,
                'Center Loss': center_loss,
                'Hits': hits / total_queries
            })
            patience += 1
            self.save_metrics_to_csv(save_dir)
            if patience >= early_stopping:
                return


    def save_metrics_to_csv(self, save_dir):


        # Create the file name using the backbone
        if self.center_loss_fn is not None:
            file_name = f"{self.backbone}_centerloss_metrics.csv"
        else:
            file_name = f"{self.backbone}_metrics.csv"
        
        file_path = os.path.join(save_dir, file_name)

        # Convert the latest metrics to a DataFrame
        df = pd.DataFrame(self.metrics)

        # If file doesn't exist, write with headers; else, append without headers
        if not os.path.exists(file_path):
            df.to_csv(file_path, index=False)
        else:
            # Append without headers
            df.tail(1).to_csv(file_path, index=False, mode='a', header=False)

        print(f"Metrics saved to {file_path}")


    def train(self,train_dataloader,current_epoch,total_epochs):

        self.net.train()
        total_ce_loss = 0
        total_triplet_loss = 0
        total_center_loss = 0

        for images,labels in tqdm(train_dataloader,dynamic_ncols=True,desc=f'Training on Epoch {current_epoch + 1 }/{total_epochs}'):

            images, labels = images.to(self.device), labels.to(self.device)

            anchors = images[:, 0, :].squeeze()
            positives = images[:,1,:].squeeze()
            negatives = images[:,2,:].squeeze()

            anchor_embeddings,_,anchor_out = self.net(anchors)
            positive_embeddings,_,positive_out = self.net(positives)
            negative_embeddings,_,negative_out = self.net(negatives)

            triplet_loss = self.triplet_loss_fn(anchor_embeddings,positive_embeddings,negative_embeddings)

            preds = rearrange([anchor_out,positive_out,negative_out],'t b e -> (b t) e')
            labels_batch = torch.flatten(labels)

            ce_loss = self.ce_loss_fn(preds,labels_batch)
            # cent_loss = self.center_loss_fn(cent_preds, labels_batch)
            if self.center_loss_fn:
                cent_preds = rearrange([anchor_embeddings, positive_embeddings, negative_embeddings], 't b e -> (b t) e')
                cent_loss = self.center_loss_fn(cent_preds, labels_batch)
                self.optimizer_center.zero_grad()
            else:
                cent_loss = torch.tensor(0.0, device=anchor_embeddings.device)  # Ensure cent_loss is a tensor


            self.optimizer.zero_grad()
            loss = triplet_loss + ce_loss + 3.5e-4*cent_loss

            loss.backward()
            self.optimizer.step()

            if self.center_loss_fn:
                self.optimizer_center.step()
                
            total_ce_loss +=ce_loss.item()
            total_triplet_loss +=triplet_loss.item()
            total_center_loss +=cent_loss.item()


        print(f'Epoch {current_epoch + 1}/{total_epochs} | Triplet Loss: {total_triplet_loss:.4f}, CE Loss: {total_ce_loss:.4f}, Center Loss: {total_center_loss:.4f}')
        return total_triplet_loss, total_ce_loss, total_center_loss
    @torch.no_grad()
    def valid(self,gallery_dataloader,query_dataloader,save_dir,similarity):

        self.net.eval()
        gallery_dict = self.build_feature_mapping(gallery_dataloader)
        gallery_features = torch.stack(list(gallery_dict.values()))
        gallery_keys = list(gallery_dict.keys())  # List of gallery image paths
        # print(len(gallery_features))
        # print(len(gallery_keys))

        total_queries = 0
        hits = 0

        for query_paths,_,_ in tqdm(query_dataloader,dynamic_ncols=True, desc='processing query'):
            
            for query_path in query_paths:
                total_queries+=1
                query_feature = gallery_dict[query_path]
                sorted_arg = self.calculate_similarity(query_feature, gallery_features, similarity)
                # Map sorted indices to gallery image paths
                sorted_arg_paths = [gallery_keys[i] for i in sorted_arg.tolist()]
                hits += self.calculate_query_hit(query_path,sorted_arg_paths)

        print(
            f'R@1 hits: {hits / total_queries}'
        )

        return hits,total_queries
    
    # @torch.no_grad()
    # def calculate_threshold(self,gallery_dataloader,query_dataloader,similarity):
    #     self.net.eval()
    #     gallery_dict = self.build_feature_mapping(gallery_dataloader)
    #     gallery_features = torch.stack(list(gallery_dict.values()))
    #     gallery_keys = list(gallery_dict.keys())  # List of gallery image paths
    #     thresholds = []
    #     for query_paths,_,_ in tqdm(query_dataloader,dynamic_ncols=True, desc='calculate similarity'):
    #         for query_path in query_paths:
    #             query_feature = gallery_dict[query_path]
    #             dist = self.get_cosine_similarity(query_feature,gallery_features)

    #             sorted_dist, _ = torch.sort(dist, descending=True)

    #             sorted_arg = self.calculate_similarity(query_feature, gallery_features, similarity)
    #             sorted_arg_paths = [gallery_keys[i] for i in sorted_arg.tolist()]
    #             index = self.calculate_query_similarity(query_path,sorted_arg_paths)
    #             print(sorted_dist[index-1])
    #             print(sorted_arg[index-1])  

    #             print(sorted_dist[index])
    #             print(sorted_arg[index])
    #             thresholds.append(sorted_dist[index])
    #     return thresholds

    @torch.no_grad()
    def calculate_threshold_all_camera(self, gallery_dataloader, query_dataloader):
        self.net.eval()
        gallery_dict = self.build_feature_mapping(gallery_dataloader)
        classified_gallery,time_windows = classify_gallery_by_time(gallery_dict)


        thresholds = {}
        for time_window in time_windows:
            gallery_features = torch.stack([tensor for _, tensor in classified_gallery[time_window]])
            gallery_keys = [key for key, _ in classified_gallery[time_window]]
            similarities = []
            labels = []
            for query_paths, _, _ in tqdm(query_dataloader, dynamic_ncols=True, desc='Calculating similarities'):
                for query_path in query_paths:
                    parts = query_path.split('_')
                    query_path_time_window = '_'.join(parts[:3])
                    query_camera_id = parts[3]
                    if query_path_time_window != time_window:
                        continue
                    query_feature = gallery_dict[query_path]
                    dist = self.get_cosine_similarity(query_feature, gallery_features)
                    sorted_dist, sorted_indices = torch.sort(dist, descending=True)
                    sorted_arg_paths = [gallery_keys[i] for i in sorted_indices.tolist()]

                    query_id = int(query_path.split('_')[-1].split('.')[0])
                    for path, similarity in zip(sorted_arg_paths, sorted_dist):
                        gallery_id = int(path.split('_')[-1].split('.')[0])
                        if path == query_path:
                            continue
                        label = 1 if gallery_id == query_id else 0
                        similarities.append(similarity.item())
                        labels.append(label)
                            # print(len(labels))
            # Convert lists to numpy arrays for threshold optimization
            import numpy as np
            similarities = np.array(similarities)
            labels = np.array(labels)

            # Determine the optimal threshold
            from sklearn.metrics import precision_recall_curve
            precision, recall, threshold_values = precision_recall_curve(labels, similarities)
            f1_scores = 2 * (precision * recall) / (precision + recall)
            optimal_threshold = threshold_values[np.argmax(f1_scores)]

            time_range = "_".join(time_window.split('_')[1:])

            thresholds[time_range] = optimal_threshold
        return thresholds
    

    @torch.no_grad()
    def calculate_threshold_single_camera(self, gallery_dataloader, query_dataloader):
        self.net.eval()
        gallery_dict = self.build_feature_mapping(gallery_dataloader)
        classified_gallery,time_windows,camera_ids = classify_gallery_by_time_and_camera(gallery_dict)

        # for query_paths,_,_ in tqdm(query_dataloader,dynamic_ncols=True,desc='classify the query paths'):
        #     for query_path in query_paths:
        #         parts = query_path.split('_')
        #         time_window = '_'.join(parts[:3])  # '1015_150000_151900'
        #         camera_id = parts[3]  # '0'

        thresholds = defaultdict(dict)
        for time_window in time_windows:
            for camera_id in camera_ids:
                gallery_features = torch.stack([tensor for _, tensor in classified_gallery[time_window][camera_id]])
                gallery_keys = [key for key, _ in classified_gallery[time_window][camera_id]]

                similarities = []
                labels = []

                for query_paths, _, _ in tqdm(query_dataloader, dynamic_ncols=True, desc='Calculating similarities'):
                    for query_path in query_paths:
                        parts = query_path.split('_')
                        query_path_time_window = '_'.join(parts[:3])
                        query_camera_id = parts[3]
                        if query_path_time_window != time_window or query_camera_id !=camera_id:
                            continue
                        query_feature = gallery_dict[query_path]
                        dist = self.get_cosine_similarity(query_feature, gallery_features)
                        sorted_dist, sorted_indices = torch.sort(dist, descending=True)
                        sorted_arg_paths = [gallery_keys[i] for i in sorted_indices.tolist()]

                        query_id = int(query_path.split('_')[-1].split('.')[0])
                        for path, similarity in zip(sorted_arg_paths, sorted_dist):
                            gallery_id = int(path.split('_')[-1].split('.')[0])
                            if path == query_path:
                                continue
                            label = 1 if gallery_id == query_id else 0
                            similarities.append(similarity.item())
                            labels.append(label)
                # print(len(labels))
                # Convert lists to numpy arrays for threshold optimization
                import numpy as np
                similarities = np.array(similarities)
                labels = np.array(labels)

                # Determine the optimal threshold
                from sklearn.metrics import precision_recall_curve
                precision, recall, threshold_values = precision_recall_curve(labels, similarities)
                f1_scores = 2 * (precision * recall) / (precision + recall)
                optimal_threshold = threshold_values[np.argmax(f1_scores)]

                time_range = "_".join(time_window.split('_')[1:])

                thresholds[time_range][camera_id] = optimal_threshold

        return thresholds
    













    def build_feature_mapping(self,gallery_dataloader):
        gallery_dict = dict()
        for img_paths,imgs,vehicle_ids in tqdm(gallery_dataloader, dynamic_ncols=True, desc='Trainer.build_feature_mapping'):
            imgs = imgs.to(self.device)

            _,features,_ = self.net(imgs)
            for img_path,feature in zip(img_paths,features):
                gallery_dict[img_path] = feature.squeeze()

        return gallery_dict
    
    def calculate_similarity(self,query_feature,gallery_features,similarity):
        if similarity == 'euclidean':
            # Calculate Euclidean distance
            dist = self.get_euclidean_dist(query_feature, gallery_features)
            sorted_arg = torch.argsort(dist) 

        elif similarity == 'cosine':
            # Calculate Cosine similarity
            dist = self.get_cosine_similarity(query_feature, gallery_features)
            sorted_arg = torch.argsort(dist, descending=True)
        return sorted_arg

    def get_euclidean_dist(self,query_feature,gallery_features):
        query_feature = rearrange(query_feature,'(b n f) -> b n f',b=1,n=1)
        gallery_features = rearrange(gallery_features,'(b n) f -> b n f',b=1)

        dist_mat = torch.cdist(query_feature,gallery_features).squeeze()

        return dist_mat
    
    def get_cosine_similarity(self,query_feature,gallery_features):
        query_feature = query_feature.unsqueeze(0) 
        query_feature = torch.nn.functional.normalize(query_feature, p=2, dim=1)
        gallery_features = torch.nn.functional.normalize(gallery_features, p=2, dim=1)
        similarity = torch.mm(query_feature, gallery_features.t())
        return similarity.squeeze()
    
    def calculate_query_hit(self,query_path,sorted_arg_paths):
        query_id = int(query_path.split('_')[-1].split('.')[0])
        for path in sorted_arg_paths:
            id = int(path.split('_')[-1].split('.')[0])
            if path == query_path:
                continue
            return 1 if id == query_id else 0
        
    def calculate_query_similarity(self,query_path,sorted_arg_paths):
        index = -1
        query_id = int(query_path.split('_')[-1].split('.')[0])
        for path in sorted_arg_paths:
            id = int(path.split('_')[-1].split('.')[0])
            index+=1
            if path == query_path or id == query_id:
                continue
            return index

def classify_gallery_by_time_and_camera(gallery_dict):
    
    classified_dict = defaultdict(lambda: defaultdict(list))
    time_windows = []
    camera_ids = []

    for img_name, tensor_data in gallery_dict.items():
        parts = img_name.split('_')
        time_window = '_'.join(parts[:3])  # '1015_150000_151900'
        camera_id = parts[3]  # '0'
        if time_window not in time_windows:
            time_windows.append(time_window)
        if camera_id not in camera_ids:
            camera_ids.append(camera_id)

        classified_dict[time_window][camera_id].append((img_name, tensor_data))

    return classified_dict,time_windows,camera_ids

def classify_gallery_by_time(gallery_dict):
    
    classified_dict = defaultdict(list)
    time_windows = []


    for img_name, tensor_data in gallery_dict.items():
        parts = img_name.split('_')
        time_window = '_'.join(parts[:3])  # '1015_150000_151900'
        camera_id = parts[3]  # '0'
        if time_window not in time_windows:
            time_windows.append(time_window)

        classified_dict[time_window].append((img_name, tensor_data))

    return classified_dict,time_windows

def prepare_trainer_and_calculate_threshold(path, backbone, custom_weights_path, device='cpu', num_workers=16, batch_size_train=16, batch_size_valid=4):
    """
    Prepare dataloaders, model, and trainer, then calculate the optimal threshold.

    Parameters:
        path (str): Path to the dataset.
        backbone (str): Backbone name (e.g., 'swin').
        custom_weights_path (str): Path to the custom weights file.
        device (str): Device to use ('cpu' or 'cuda').
        num_workers (int): Number of workers for dataloaders.
        batch_size_train (int): Batch size for the training dataloader.
        batch_size_valid (int): Batch size for the validation dataloader.

    Returns:
        float: Optimal threshold for the model.
    """

    # Define transformations
    train_transform = Transforms.get_training_transform()
    valid_transform = Transforms.get_valid_transform()

    # Prepare dataloaders
    gallery_dataloader, query_dataloader = datasets.get_AICUP_valid(
        path, num_workers=num_workers, batch_size=batch_size_valid, transform=valid_transform
    )
    _, total_class = datasets.get_AICUP_train(
        path, num_workers=num_workers, batch_size=batch_size_train, transform=train_transform, shuffle=True
    )

    # Initialize the model
    net = make_model(backbone=backbone, num_classes=total_class)

    # Load custom weights
    state_dict = torch.load(custom_weights_path, map_location='cpu')
    net.load_state_dict(state_dict)
    print("Custom weights loaded successfully, excluding the classification layer.")

    # Initialize the trainer
    trainer = AICUPTrainer(
        backbone=backbone,
        net=net,
        ce_loss_fn=None,
        triplet_loss_fn=None,
        center_loss_fn=None,
        optimizer=None,
        lr_scheduler=None,
        device=device
    )

    # Calculate and return the optimal threshold
    single_camera_thresholds = trainer.calculate_threshold_single_camera(gallery_dataloader,query_dataloader)
    all_camera_thresholds = trainer.calculate_threshold_all_camera(gallery_dataloader,query_dataloader)
    return single_camera_thresholds,all_camera_thresholds
import datasets
from torchvision import transforms

if __name__ == '__main__':
    dataset_path = '/home/eddy/Desktop/cropped_image'
    backbone_model = 'swin'
    weights_path = '/home/eddy/Desktop/MasterThesis/mainProgram/AICUP_datasets_fine_tune/swin_center_lr_0.5_loss_3e-4_smmothing_0.1/swin_center_loss_best.pth'
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

    single_camera_thresholds,all_camera_thresholds = prepare_trainer_and_calculate_threshold(
        path=dataset_path,
        backbone=backbone_model,
        custom_weights_path=weights_path,
        device=device_type
    )

    print(f"single_camera_thresholds: {single_camera_thresholds}")
    print(f"all_camera_thresholds: {all_camera_thresholds}")
