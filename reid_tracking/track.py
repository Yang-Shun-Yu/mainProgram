from data_preprocess import create_label_feature_map
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from model import make_model
import torch
import torch.nn.functional as F
import Transforms
import math
from collections import OrderedDict
from collections import defaultdict

import copy

class OrderedDefaultdict(OrderedDict):
    def __init__(self, default_factory=None, *args, **kwargs):
        if default_factory is not None and not callable(default_factory):
            raise TypeError('first argument must be callable or None')
        self.default_factory = default_factory
        super().__init__(*args, **kwargs)

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        self[key] = value = self.default_factory()
        return value
# Function to calculate cosine similarity
def cosine_similarity(tensor1, tensor2):


    return F.cosine_similarity(tensor1, tensor2)
# Function to write buffer to disk

def write_buffer_to_disk(buffer):
    buffer_path,buffer_feature_id_list = buffer.popitem(last=False)
    # label_path,id_list = buffer.popitem(last=False)
    
    folder = buffer_path.split('/')[-2]
    file = buffer_path.split('/')[-1]

    folder_path = os.path.join('/home/eddy/Desktop/MasterThesis/mainProgram/reid_tracking/labels',folder)
    file_path = os.path.join(folder_path,file)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    
    # with open(file_path, 'w') as file:
    #     file.write("feature          id\n")
    #     for feature,id in buffer_feature_id_list:
    #         file.write(f"{feature}          {id}\n")

    with open(file_path, 'w') as file:
        for _,id,_,_,_,_ in buffer_feature_id_list:

            file.write(f"{id}\n")

import os

def write_storage(merge_storage, storage_forward, storage_reverse):
    for storage, label_folder in zip(
        [merge_storage, storage_forward, storage_reverse], 
        ['merge_labels', 'forward_labels', 'reverse_labels']
    ):
        for file_path, entries in storage.items():
            parts = file_path.split(os.sep)
            parts[-3] = label_folder  # Change the folder name to match the label type
            
            folder_path = os.sep.join(parts[:-1])  # Construct the folder path (excluding the file name)
            os.makedirs(folder_path, exist_ok=True)  # Ensure the folder exists
            
            file_path = os.sep.join(parts)  # Construct the full file path
            
            with open(file_path, 'w') as file:
                for id, _, _ in entries:
                    file.write(f"{id}\n")

        




def save_buffer_to_storage(buffer,storage):

    # buffer_path,buffer_feature_id_list = next(iter(buffer.items()))
    buffer_path,buffer_feature_id_list = buffer.popitem(last=False)
    folder = buffer_path.split('/')[-2]
    file = buffer_path.split('/')[-1]

    folder_path = os.path.join('/home/eddy/Desktop/MasterThesis/mainProgram/reid_tracking/labels',folder)
    file_path = os.path.join(folder_path,file)

    for _,buffer_id,buffer_center_x_ratio,buffer_center_y_ratio,_,_ in buffer_feature_id_list:
        storage[file_path].append((buffer_id,buffer_center_x_ratio,buffer_center_y_ratio))


# def merge_storages(storage_forward, storage_reverse):
#     # Make a deep copy so that merge_storage can be modified independently
#     merge_storage = copy.deepcopy(storage_forward)
    
#     # Build clusters for forward and reverse storage
#     forward_cluster = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
#     reverse_cluster = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
#     # Process storage_forward with progress visualization
#     for file_path, entries in tqdm(storage_forward.items(), desc="Processing forward storage"):
#         parts = file_path.split(os.sep)
#         time = parts[-2]
#         cam = parts[-1].split('_')[0]
#         for buffer_id, center_x, center_y in entries:
#             forward_cluster[time][cam][buffer_id].append((center_x, center_y))
    
#     # Process storage_reverse with progress visualization
#     for file_path, entries in tqdm(storage_reverse.items(), desc="Processing reverse storage"):
#         parts = file_path.split(os.sep)
#         if len(parts) < 2:
#             continue
#         time = parts[-2]
#         cam = parts[-1].split('_')[0]
#         for buffer_id, center_x, center_y in entries:
#             reverse_cluster[time][cam][buffer_id].append((center_x, center_y))
    
#     # Create an ID mapping by comparing clusters.
#     id_mapping = {}
#     # Loop through times; here we use tqdm to track outer loop progress.
#     for time in tqdm(forward_cluster, desc="Mapping IDs over times"):
#         for cam in tqdm(forward_cluster[time], desc=f"Time {time} cameras", leave=False):
#             for f_id, f_coords in forward_cluster[time][cam].items():
#                 match_number = 0
#                 match_found = False
#                 # Check reverse_cluster for matching coordinates
#                 for r_id, r_coords in reverse_cluster[time][cam].items():
#                     for (fx, fy) in f_coords:
#                         for (rx, ry) in r_coords:
#                             if fx == rx and fy == ry:
#                                 match_number += 1
#                                 if match_number == 2:
#                                     match_found = True
#                                     # Store the mapping with a tuple key: (time, cam, r_id)
#                                     id_mapping[(time, cam, r_id)] = f_id
#                                     break
#                         if match_found:
#                             break
#                     if match_found:
#                         break
                        
#     # Update merge_storage using the id_mapping
#     for file_path, entries in tqdm(storage_reverse.items(), desc="Updating merge storage"):
#         # Start with the current merge_storage list for this file_path.
#         current_entries = merge_storage[file_path]

#         parts = file_path.split(os.sep)
#         if len(parts) < 2:
#             continue
#         time = parts[-2]
#         cam = parts[-1].split('_')[0]

#         # For each entry in the reverse storage, update the current_entries.
#         for (buffer_id, center_x, center_y) in entries:
#             key = (time, cam, buffer_id)
#             # Use a tuple key (time, cam, buffer_id) to check in id_mapping.
#             if key in id_mapping:
#                 new_id = id_mapping[key]
#                 if any(f_buffer_id == new_id and (f_center_x != center_x or f_center_y != center_y)  for f_buffer_id, f_center_x, f_center_y in current_entries):
#                     # If it does, remove the mapping to avoid duplicates.
#                     continue
#                 else:

#                     updated_entries = []
#                     for (f_buffer_id, f_center_x, f_center_y) in current_entries:
#                         # If the coordinates match, replace the forward ID with the new_id.
#                         if center_x == f_center_x and center_y == f_center_y:
#                             updated_entries.append((new_id, f_center_x, f_center_y))
#                         else:
#                             updated_entries.append((f_buffer_id, f_center_x, f_center_y))
#                     current_entries = updated_entries
#         # Finally, assign the fully updated entries back to merge_storage.
#         merge_storage[file_path] = current_entries
#     return merge_storage

def write_id_mapping_to_txt(id_mapping, output_file):
    """
    Write the id_mapping dictionary to a text file.
    Each line in the file will be in the format:
        (time, cam, reverse_buffer_id): forward_buffer_id
    """
    with open(output_file, "w") as f:
        for key, value in id_mapping.items():
            f.write(f"{key}: {value}\n")

def merge_storages(storage_forward, storage_reverse):
    # Make a deep copy so that merge_storage can be modified independently
    merge_storage = copy.deepcopy(storage_forward)
    
    # Build clusters for forward and reverse storage
    forward_cluster = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    reverse_cluster = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    # Process storage_forward with progress visualization
    for file_path, entries in tqdm(storage_forward.items(), desc="Processing forward storage"):
        parts = file_path.split(os.sep)
        time = parts[-2]
        cam = parts[-1].split('_')[0]
        for buffer_id, center_x, center_y in entries:
            forward_cluster[time][cam][buffer_id].append((center_x, center_y))
    
    # Process storage_reverse with progress visualization
    for file_path, entries in tqdm(storage_reverse.items(), desc="Processing reverse storage"):
        parts = file_path.split(os.sep)
        if len(parts) < 2:
            continue
        time = parts[-2]
        cam = parts[-1].split('_')[0]
        for buffer_id, center_x, center_y in entries:
            reverse_cluster[time][cam][buffer_id].append((center_x, center_y))
    
    # Create an ID mapping by comparing clusters.
    # id_mapping = {}
    # # Loop through times; here we use tqdm to track outer loop progress.
    # for time in tqdm(forward_cluster, desc="Mapping IDs over times"):
    #     for cam in tqdm(forward_cluster[time], desc=f"Time {time} cameras", leave=False):
    #         for f_id, f_coords in forward_cluster[time][cam].items():

    #             # Check reverse_cluster for matching coordinates
    #             for r_id, r_coords in reverse_cluster[time][cam].items():

    #                 find the most have same coordinate and let  id_mapping[(time, cam, r_id)] = the most have same coordinate f_id and the most have same coordinate must have at least 2 same  
                    # match_number = 0
                    # match_found = False
                    # for (fx, fy) in f_coords:
                    #     for (rx, ry) in r_coords:
                    #         if fx == rx and fy == ry:
                    #             match_number += 1
                    #             if match_number == 2:
                    #                 match_found = True
                    #                 # Store the mapping with a tuple key: (time, cam, r_id)
                    #                 id_mapping[(time, cam, r_id)] = f_id
                    #                 break
                    #     if match_found:
                    #         break
                    # if match_found:
                    #     break
    id_mapping = {}
    # For each time and camera, iterate over each reverse id candidate...
    for time in tqdm(forward_cluster, desc="Mapping IDs over times"):
        for cam in tqdm(forward_cluster[time], desc=f"Time {time} cameras", leave=False):
            for r_id, r_coords in reverse_cluster[time][cam].items():
                best_count = 0
                best_f_id = None
                # ...and compare to each forward id candidate.
                for f_id, f_coords in forward_cluster[time][cam].items():
                    count = 0
                    for (rx, ry) in r_coords:
                        for (fx, fy) in f_coords:
                            if rx == fx and ry == fy:
                                count += 1
                    # If this forward id produces more matches, record it.
                    if count > best_count:
                        best_count = count
                        best_f_id = f_id
                # Only map if at least 2 coordinates match.
                if best_count >= 2:
                    id_mapping[(time, cam, r_id)] = best_f_id

    write_id_mapping_to_txt(id_mapping, "id_mapping.txt")
    # Update merge_storage using the id_mapping
    # for file_path, entries in tqdm(storage_reverse.items(), desc="Updating merge storage"):
    #     # Start with the current merge_storage list for this file_path.
    #     current_entries = merge_storage[file_path]

    #     parts = file_path.split(os.sep)
    #     if len(parts) < 2:
    #         continue
    #     time = parts[-2]
    #     cam = parts[-1].split('_')[0]
    #     updated_entries = []
    #     # For each entry in the reverse storage, update the current_entries.
    #     for index,(buffer_id, center_x, center_y) in enumerate(entries):
    #         # Use a tuple key (time, cam, buffer_id) to check in id_mapping.
            
    #         if (time, cam, buffer_id) in id_mapping:
    #             # updated_entries.append((id_mapping[(time, cam, buffer_id)], center_x, center_y))  # Only update if no match found

    #             match_found = False 
    #             # Process all entries for the current file_path.
    #             f_buffer_id, f_center_x, f_center_y = current_entries[index]
    #             for (f_buffer_id, f_center_x, f_center_y) in current_entries:
    #                 if (f_buffer_id == id_mapping[(time, cam, buffer_id)]):
    #                     updated_entries.append(current_entries[index])
    #                     match_found = True
    #                     del id_mapping[(time, cam, buffer_id)]
    #                     break
                
    #             if not match_found:
    #                 updated_entries.append((id_mapping[(time, cam, buffer_id)], center_x, center_y))  # Only update if no match found
    #             # Update the current_entries for the next iteration.
                
    #         else:
    #             updated_entries.append(current_entries[index])
    #     current_entries = updated_entries

    #     # Finally, assign the fully updated entries back to merge_storage.
    #     merge_storage[file_path] = current_entries
    # return merge_storage

    for file_path, entries in tqdm(storage_reverse.items(), desc="Updating merge storage"):
        # Start with the current merge_storage list for this file_path.
        current_entries = merge_storage[file_path]

        parts = file_path.split(os.sep)
        if len(parts) < 2:
            continue
        time = parts[-2]
        cam = parts[-1].split('_')[0]

        # Process each reverse entry for this file.
        for (buffer_id, center_x, center_y) in entries:
            key = (time, cam, buffer_id)
            # Only update if the mapping exists.
            if key in id_mapping:
                new_id = id_mapping[key]
                # Check if current_entries already contains an entry with new_id,
                # but with coordinates that do not match (to avoid duplicate mappings).
                if any(f_buffer_id == new_id and (f_center_x != center_x or f_center_y != center_y)
                    for f_buffer_id, f_center_x, f_center_y in current_entries):
                    # If such an entry exists, remove the mapping to avoid duplicates.
                    id_mapping.pop(key, None)
                    # Optionally, skip processing this reverse entry.
                    continue

                updated_entries = []
                # Process all entries for the current file_path.
                for (f_buffer_id, f_center_x, f_center_y) in current_entries:
                    # If coordinates match, substitute the id with the mapped new_id.
                    if center_x == f_center_x and center_y == f_center_y:
                        updated_entries.append((new_id, f_center_x, f_center_y))
                    else:
                        updated_entries.append((f_buffer_id, f_center_x, f_center_y))
                # Update current_entries for further processing.
                current_entries = updated_entries
        # Finally, assign the fully updated entries back to merge_storage.
        merge_storage[file_path] = current_entries

    return merge_storage
# def check_same_id(storage):

#     max_id = 0
#     for file_path, entries in storage.items():
#         for (buffer_id, _, _) in entries:
#             if buffer_id > max_id:
#                 max_id = buffer_id

#     for file_path, entries in storage.items():
#         parts = file_path.split(os.sep)
#         time = parts[-2]
#         cam = parts[-1].split('_')[0]
#         frame = int(parts[-1].split('_')[1].split('.')[0])
#         if frame == 1:
#             continue
#         else:
     
#                 # Loop until there are no duplicate indices left.
#             cont = 0
#             while True:
#                 cont += 1
#                 # Build a mapping from assigned_id to the indices in temp_feature_assignments
#                 assigned_id_dict = defaultdict(list)
#                 for index, (assigned_id, _, _,) in enumerate(entries):
#                     assigned_id_dict[assigned_id].append(index)
                            
#                 # Identify any IDs that appear more than once.
#                 duplicate_indices = {key: value for key, value in assigned_id_dict.items() if len(value) > 1}

#                 # If there are no duplicates, exit the loop.
#                 if not duplicate_indices:
#                     break

#                 # Resolve duplicates by keeping one "best" index per duplicate_id.
#                 for duplicate_id, indices in list(duplicate_indices.items()):
#                     # Get the latest buffer entry (you may choose a different strategy here)
#                     buffer_path, buffer_feature_id_list = next(reversed(buffer.items()))
#                     keep_index = -1

#                     # Find the best candidate among the duplicate indices.
#                     for buffer_feature, buffer_id, buffer_center_x_ratio, buffer_center_y_ratio, buffer_displacement_x, buffer_displacement_y in buffer_feature_id_list:
#                         if buffer_id != duplicate_id:
#                             continue

#                         # Use displacement if available.
#                         if buffer_displacement_x is not None and buffer_displacement_y is not None:
#                             min_distance = float('inf')
#                             for idx in indices:
#                                 predict_center_x = buffer_center_x_ratio + buffer_displacement_x
#                                 predict_center_y = buffer_center_y_ratio + buffer_displacement_y

#                                 delta_x = temp_feature_assignments[idx][2] - predict_center_x
#                                 delta_y = temp_feature_assignments[idx][3] - predict_center_y
#                                 displacement_magnitude = math.sqrt(delta_x**2 + delta_y**2)

#                                 if displacement_magnitude < min_distance:
#                                     keep_index = idx
#                                     min_distance = displacement_magnitude
#                         else:
#                             max_similarity = -1
#                             for idx in indices:
#                                 similarity = cosine_similarity(temp_feature_assignments[idx][0], buffer_feature)
#                                 similarity = similarity.squeeze().item()  
#                                 if similarity > max_similarity:
#                                     keep_index = idx
#                                     max_similarity = similarity

#                     # Remove the chosen index (keep_index) from the list so that only the others remain for reassignment.
#                     if keep_index in duplicate_indices[duplicate_id]:
#                         duplicate_indices[duplicate_id].remove(keep_index)

#                 # For all remaining duplicate indices, reassign new IDs.

#                 for duplicate_id, indices in list(duplicate_indices.items()):
#                     buffer_path, buffer_feature_id_list = next(reversed(buffer.items()))
#                     for idx in indices:
#                         feature, _, center_x_ratio, center_y_ratio, displacement_x, displacement_y = temp_feature_assignments[idx]

#                         new_id = None
#                         similarity_matrix = []
#                         # Check against the buffer to see if there's a close candidate.
#                         for buffer_feature, buffer_id, buffer_center_x_ratio, buffer_center_y_ratio, buffer_displacement_x, buffer_displacement_y in buffer_feature_id_list:
#                             if buffer_id == duplicate_id:
#                                 continue
#                             similarity = cosine_similarity(feature, buffer_feature)
#                             similarity = similarity.squeeze().item()
#                             similarity_matrix.append((similarity, buffer_id,buffer_center_x_ratio,buffer_center_y_ratio))
                        
#                         # Sort the similarity matrix from high to low based on similarity value.
#                         sorted_similarity = sorted(similarity_matrix, key=lambda x: x[0], reverse=True)
                        
#                         # 取得第 cont 高的相似度 (the cont-th highest value)
#                         if len(sorted_similarity) >= cont:
#                             candidate_similarity, candidate_buffer_id,candidate_buffer_center_x_ratio, candidate_buffer_center_y_ratio= sorted_similarity[cont - 1]
#                             # 比較該相似度與 single_camera_thresholds[time_range][cam_id]
#                             if candidate_similarity > single_camera_thresholds[time_range][cam_id]:
#                                 new_id = candidate_buffer_id
#                                 displacement_x = center_x_ratio-candidate_buffer_center_x_ratio
#                                 displacement_y = center_y_ratio-candidate_buffer_center_y_ratio
                        
#                         # If no candidate was close enough, assign a new id.
#                         if new_id is None:
#                             new_id = id_counter
#                             id_counter += 1
                        
#                         # Update the assignment in temp_feature_assignments.
                        

#                         temp_feature_assignments[idx] = (feature, new_id, center_x_ratio, center_y_ratio, displacement_x, displacement_y)




        








import sys
import os
import torch
# Add the folder to sys.path

sys.path.append(os.path.abspath("/home/eddy/Desktop/MasterThesis/mainProgram/AICUP_datasets_fine_tune"))

# Now import the function
from trainer import prepare_trainer_and_calculate_threshold



if __name__=='__main__':
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

    
    buffer_size  = 5 
    image_root_directory = '/home/eddy/Desktop/MasterThesis/mainProgram/AICUP_datasets/test/images'
    label_root_directory = '/home/eddy/Desktop/MasterThesis/mainProgram/AICUP_datasets/test/labels'
    
    save_path = os.path.join('/home/eddy/Desktop/MasterThesis/mainProgram/reid_tracking','labels')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 定義圖像預處理
    image_transform = Transforms.get_valid_transform()
    net = make_model(backbone='swin', num_classes=3441)
    custom_weights = '/home/eddy/Desktop/MasterThesis/mainProgram/AICUP_datasets_fine_tune/swin_center_lr_0.5_loss_3e-4_smmothing_0.1/swin_center_loss_best.pth'
    state_dict = torch.load(custom_weights, map_location='cpu')
 
    # Load the updated state_dict into the model
    net.load_state_dict(state_dict)
    net.eval()

    label_to_feature_map = create_label_feature_map(net,image_root_directory,label_root_directory,image_transform)

    # Initialize variables
    buffer = OrderedDefaultdict(list)
    storage_forward = defaultdict(list)

    id_counter = 1

    old_cam = None
    first_frame = True


    for label_path, objects in label_to_feature_map.items():
        # Skip if feature_list is empty
        time_range = label_path.split('/')[-2]
        time_range = "_".join(time_range.split('_')[1:])
        cam_id = label_path.split('/')[-1]
        cam_id = cam_id.split('_')[0]
        if old_cam ==None:
            old_cam = cam_id
        elif old_cam != cam_id:
            while buffer:
                save_buffer_to_storage(buffer,storage_forward)
                # write_buffer_to_disk(buffer)

            id_counter = 1
            old_cam = cam_id
            first_frame = True


        if len(buffer) > buffer_size:
            save_buffer_to_storage(buffer,storage_forward)
            # write_buffer_to_disk(buffer)

        # Check if feature_list is empty or contains only empty lists
        if objects == [[]]:
            # print(f"Empty or all sublists empty in feature_list for label_path: {label_path}")
            buffer[label_path].append((None, None, None, None, None,None))
            continue
        # print("feature_list: ",feature_list)
        temp_feature_assignments = []
        for obj in objects:
            feature = obj["feature"]
            center_x_ratio = obj['center_x_ratio']
            center_y_ratio = obj['center_y_ratio']
            assigned_id = None
            displacement_x = None
            displacement_y = None
            max_similarity = -1
            candidate = []
            found = False
            if first_frame:
                assigned_id = id_counter
                id_counter += 1
            else:
                for buffer_path, buffer_feature_id_list in reversed(buffer.items()):
                
                    for buffer_feature, buffer_id,buffer_center_x_ratio,buffer_center_y_ratio,buffer_displacement_x,buffer_displacement_y in buffer_feature_id_list:

                        if buffer_feature is not None:
                            similarity = cosine_similarity(feature,buffer_feature)
                            similarity = similarity.squeeze()
                            if similarity > single_camera_thresholds[time_range][cam_id] :
                                found = True
                                # candidate.append(buffer_id,buffer_center_x_ratio,buffer_center_y_ratio,buffer_displacement_x,buffer_displacement_y)
                                candidate.append((buffer_id, buffer_center_x_ratio, buffer_center_y_ratio, buffer_displacement_x, buffer_displacement_y))

                    if len(candidate)>1:
                        min_distance = float('inf')
                        for buffer_id,buffer_center_x_ratio,buffer_center_y_ratio,buffer_displacement_x,buffer_displacement_y in candidate:
                            # 先判斷位移方向是否一致
                            if buffer_displacement_x is None and buffer_displacement_y is None:
                                continue
                            temp_displacement_x = center_x_ratio-buffer_center_x_ratio
                            temp_displacement_y = center_y_ratio-buffer_center_y_ratio
                            # 將位移向量轉換為 torch 張量
                            temp_displacement = torch.tensor([center_x_ratio - buffer_center_x_ratio, 
                                  center_y_ratio - buffer_center_y_ratio], dtype=torch.float32)

                            buffer_displacement = torch.tensor([buffer_displacement_x, buffer_displacement_y], dtype=torch.float32)

                            # 计算内积
                            dot_product = torch.dot(temp_displacement, buffer_displacement)
                            if dot_product < 0:
                                continue


                            
                            predict_center_x = buffer_center_x_ratio+buffer_displacement_x
                            predict_center_y = buffer_center_y_ratio+buffer_displacement_y

                            delta_x = center_x_ratio-predict_center_x
                            delta_y = center_y_ratiocreate_label_feature_map= math.sqrt(delta_x**2 + delta_y**2)

                            if displacement_magnitude < min_distance:
                                assigned_id = buffer_id
                                min_distance = displacement_magnitude
                                displacement_x = center_x_ratio-buffer_center_x_ratio
                                displacement_y = center_y_ratio-buffer_center_y_ratio
                    
                        

                    elif len(candidate)==1:
                        assigned_id = candidate[0][0]
                        displacement_x = center_x_ratio-candidate[0][1]
                        displacement_y = center_y_ratio-candidate[0][2]
                    # 沒有 buffer_displacement
                    if len(candidate)>1 and assigned_id is None:
                        min_distance = float('inf')
                        for buffer_id,buffer_center_x_ratio,buffer_center_y_ratio,buffer_displacement_x,buffer_displacement_y in candidate:
                            delta_x = center_x_ratio-buffer_center_x_ratio
                            delta_y = center_y_ratio-buffer_center_y_ratio
                            displacement_magnitude = math.sqrt(delta_x**2 + delta_y**2)
                            if displacement_magnitude < min_distance:
                                assigned_id = buffer_id
                                min_distance = displacement_magnitude
                                displacement_x = center_x_ratio-buffer_center_x_ratio
                                displacement_y = center_y_ratio-buffer_center_y_ratio

                    if found:
                        break
            # 沒配對到
            if assigned_id==None:
                buffer_path, buffer_feature_id_list = next(reversed(buffer.items()))
                min_distance = float('inf')
                close_feature = None
                for buffer_feature, buffer_id,buffer_center_x_ratio,buffer_center_y_ratio,buffer_displacement_x,buffer_displacement_y in buffer_feature_id_list:
                    if buffer_displacement_x is not None and buffer_displacement_y is not None:
                        # 先判斷位移方向是否一致
                        temp_displacement_x = center_x_ratio-buffer_center_x_ratio
                        temp_displacement_y = center_y_ratio-buffer_center_y_ratio
                        temp_displacement = torch.tensor([center_x_ratio - buffer_center_x_ratio, 
                                  center_y_ratio - buffer_center_y_ratio], dtype=torch.float32)

                        buffer_displacement = torch.tensor([buffer_displacement_x, buffer_displacement_y], dtype=torch.float32)

                        # 计算内积
                        dot_product = torch.dot(temp_displacement, buffer_displacement)
                        if dot_product < 0:
                            continue

                        predict_center_x = buffer_center_x_ratio+buffer_displacement_x
                        predict_center_y = buffer_center_y_ratio+buffer_displacement_y

                        delta_x = center_x_ratio-predict_center_x
                        delta_y = center_y_ratio-predict_center_y
                        displacement_magnitude = math.sqrt(delta_x**2 + delta_y**2)

                        if displacement_magnitude < min_distance:
                            assigned_id = buffer_id
                            min_distance = displacement_magnitude
                            displacement_x = center_x_ratio-buffer_center_x_ratio
                            displacement_y = center_y_ratio-buffer_center_y_ratio
                            close_feature = buffer_feature
                            
                if assigned_id ==None:
                    assigned_id = id_counter
                    id_counter += 1
                else:
                    similarity = cosine_similarity(feature,close_feature)
                    similarity = similarity.squeeze()
                    if similarity < single_camera_thresholds[time_range][cam_id]/2:
                        assigned_id = id_counter
                        id_counter += 1
                 

                # if max_similarity is None or max_similarity < single_camera_thresholds[time_range][cam_id]:
                #     assigned_id = id_counter
                #     id_counter += 1
            temp_feature_assignments.append((feature, assigned_id,center_x_ratio,center_y_ratio,displacement_x,displacement_y))

        # 處理同一個frame出現相同id情況


        # Loop until there are no duplicate indices left.
        cont = 0
        while True:
            cont += 1
            # Build a mapping from assigned_id to the indices in temp_feature_assignments
            assigned_id_dict = defaultdict(list)
            for index, (_, assigned_id, _, _, _, _) in enumerate(temp_feature_assignments):
                assigned_id_dict[assigned_id].append(index)
                        
            # Identify any IDs that appear more than once.
            duplicate_indices = {key: value for key, value in assigned_id_dict.items() if len(value) > 1}

            # If there are no duplicates, exit the loop.
            if not duplicate_indices:
                break

            # Resolve duplicates by keeping one "best" index per duplicate_id.
            for duplicate_id, indices in list(duplicate_indices.items()):
                # Get the latest buffer entry (you may choose a different strategy here)
                buffer_path, buffer_feature_id_list = next(reversed(buffer.items()))
                keep_index = -1

                # Find the best candidate among the duplicate indices.
                for buffer_feature, buffer_id, buffer_center_x_ratio, buffer_center_y_ratio, buffer_displacement_x, buffer_displacement_y in buffer_feature_id_list:
                    if buffer_id != duplicate_id:
                        continue

                    # Use displacement if available.
                    if buffer_displacement_x is not None and buffer_displacement_y is not None:
                        min_distance = float('inf')
                        for idx in indices:
                            predict_center_x = buffer_center_x_ratio + buffer_displacement_x
                            predict_center_y = buffer_center_y_ratio + buffer_displacement_y

                            delta_x = temp_feature_assignments[idx][2] - predict_center_x
                            delta_y = temp_feature_assignments[idx][3] - predict_center_y
                            displacement_magnitude = math.sqrt(delta_x**2 + delta_y**2)

                            if displacement_magnitude < min_distance:
                                keep_index = idx
                                min_distance = displacement_magnitude
                    else:
                        max_similarity = -1
                        for idx in indices:
                            similarity = cosine_similarity(temp_feature_assignments[idx][0], buffer_feature)
                            similarity = similarity.squeeze().item()  
                            if similarity > max_similarity:
                                keep_index = idx
                                max_similarity = similarity

                # Remove the chosen index (keep_index) from the list so that only the others remain for reassignment.
                if keep_index in duplicate_indices[duplicate_id]:
                    duplicate_indices[duplicate_id].remove(keep_index)

            # For all remaining duplicate indices, reassign new IDs.

            for duplicate_id, indices in list(duplicate_indices.items()):
                buffer_path, buffer_feature_id_list = next(reversed(buffer.items()))
                for idx in indices:
                    feature, _, center_x_ratio, center_y_ratio, displacement_x, displacement_y = temp_feature_assignments[idx]

                    new_id = None
                    similarity_matrix = []
                    # Check against the buffer to see if there's a close candidate.
                    for buffer_feature, buffer_id, buffer_center_x_ratio, buffer_center_y_ratio, buffer_displacement_x, buffer_displacement_y in buffer_feature_id_list:
                        if buffer_id == duplicate_id:
                            continue
                        similarity = cosine_similarity(feature, buffer_feature)
                        similarity = similarity.squeeze().item()
                        similarity_matrix.append((similarity, buffer_id,buffer_center_x_ratio,buffer_center_y_ratio))
                    
                    # Sort the similarity matrix from high to low based on similarity value.
                    sorted_similarity = sorted(similarity_matrix, key=lambda x: x[0], reverse=True)
                    
                    # 取得第 cont 高的相似度 (the cont-th highest value)
                    if len(sorted_similarity) >= cont:
                        candidate_similarity, candidate_buffer_id,candidate_buffer_center_x_ratio, candidate_buffer_center_y_ratio= sorted_similarity[cont - 1]
                        # 比較該相似度與 single_camera_thresholds[time_range][cam_id]
                        if candidate_similarity > single_camera_thresholds[time_range][cam_id]:
                            new_id = candidate_buffer_id
                            displacement_x = center_x_ratio-candidate_buffer_center_x_ratio
                            displacement_y = center_y_ratio-candidate_buffer_center_y_ratio
                    
                    # If no candidate was close enough, assign a new id.
                    if new_id is None:
                        new_id = id_counter
                        id_counter += 1
                    
                    # Update the assignment in temp_feature_assignments.
                    

                    temp_feature_assignments[idx] = (feature, new_id, center_x_ratio, center_y_ratio, displacement_x, displacement_y)
            
            # The loop will recompute duplicate_indices in the next iteration.
        first_frame = False
        buffer[label_path].extend(temp_feature_assignments)

    while buffer:
        save_buffer_to_storage(buffer,storage_forward)
        # write_buffer_to_disk(buffer)

    #reverse tracking the camera
    reversed_label_to_feature_map = dict(
        sorted(
            label_to_feature_map.items(),
            key=lambda item: (
                item[0].split('/')[-2],  # Sort by subdirectory name (ascending)
                -int(item[0].split('/')[-1].split('.')[0])  # Sort by filename (descending)
            )
        )
    )    


    # Initialize variables
    buffer = OrderedDefaultdict(list)
    storage_reverse = defaultdict(list)
    id_counter = 1

    old_cam = None
    first_frame = True


    for label_path, objects in reversed_label_to_feature_map.items():
        # Skip if feature_list is empty
        time_range = label_path.split('/')[-2]
        time_range = "_".join(time_range.split('_')[1:])
        cam_id = label_path.split('/')[-1]
        cam_id = cam_id.split('_')[0]
        if old_cam ==None:
            old_cam = cam_id
        elif old_cam != cam_id:
            while buffer:
                save_buffer_to_storage(buffer,storage_reverse)
                # write_buffer_to_disk(buffer)

            id_counter = 1
            old_cam = cam_id
            first_frame = True


        if len(buffer) > buffer_size:
            save_buffer_to_storage(buffer,storage_reverse)
            # write_buffer_to_disk(buffer)

        # Check if feature_list is empty or contains only empty lists
        if objects == [[]]:
            # print(f"Empty or all sublists empty in feature_list for label_path: {label_path}")
            buffer[label_path].append((None, None, None, None, None,None))
            continue
        # print("feature_list: ",feature_list)
        temp_feature_assignments = []
        for obj in objects:
            feature = obj["feature"]
            center_x_ratio = obj['center_x_ratio']
            center_y_ratio = obj['center_y_ratio']
            assigned_id = None
            displacement_x = None
            displacement_y = None
            max_similarity = -1
            candidate = []
            found = False
            if first_frame:
                assigned_id = id_counter
                id_counter += 1
            else:
                for buffer_path, buffer_feature_id_list in reversed(buffer.items()):
                
                    for buffer_feature, buffer_id,buffer_center_x_ratio,buffer_center_y_ratio,buffer_displacement_x,buffer_displacement_y in buffer_feature_id_list:

                        if buffer_feature is not None:
                            similarity = cosine_similarity(feature,buffer_feature)
                            similarity = similarity.squeeze()
                            if similarity > single_camera_thresholds[time_range][cam_id] :
                                found = True
                                # candidate.append(buffer_id,buffer_center_x_ratio,buffer_center_y_ratio,buffer_displacement_x,buffer_displacement_y)
                                candidate.append((buffer_id, buffer_center_x_ratio, buffer_center_y_ratio, buffer_displacement_x, buffer_displacement_y))

                    if len(candidate)>1:
                        min_distance = float('inf')
                        for buffer_id,buffer_center_x_ratio,buffer_center_y_ratio,buffer_displacement_x,buffer_displacement_y in candidate:
                            # 先判斷位移方向是否一致
                            if buffer_displacement_x is None and buffer_displacement_y is None:
                                continue
                            temp_displacement_x = center_x_ratio-buffer_center_x_ratio
                            temp_displacement_y = center_y_ratio-buffer_center_y_ratio
                            # 將位移向量轉換為 torch 張量
                            temp_displacement = torch.tensor([center_x_ratio - buffer_center_x_ratio, 
                                  center_y_ratio - buffer_center_y_ratio], dtype=torch.float32)

                            buffer_displacement = torch.tensor([buffer_displacement_x, buffer_displacement_y], dtype=torch.float32)

                            # 计算内积
                            dot_product = torch.dot(temp_displacement, buffer_displacement)
                            if dot_product < 0:
                                continue


                            
                            predict_center_x = buffer_center_x_ratio+buffer_displacement_x
                            predict_center_y = buffer_center_y_ratio+buffer_displacement_y

                            delta_x = center_x_ratio-predict_center_x
                            delta_y = center_y_ratio-predict_center_y
                            displacement_magnitude = math.sqrt(delta_x**2 + delta_y**2)

                            if displacement_magnitude < min_distance:
                                assigned_id = buffer_id
                                min_distance = displacement_magnitude
                                displacement_x = center_x_ratio-buffer_center_x_ratio
                                displacement_y = center_y_ratio-buffer_center_y_ratio
                    
                        

                    elif len(candidate)==1:
                        assigned_id = candidate[0][0]
                        displacement_x = center_x_ratio-candidate[0][1]
                        displacement_y = center_y_ratio-candidate[0][2]
                    # 沒有 buffer_displacement
                    if len(candidate)>1 and assigned_id is None:
                        min_distance = float('inf')
                        for buffer_id,buffer_center_x_ratio,buffer_center_y_ratio,buffer_displacement_x,buffer_displacement_y in candidate:
                            delta_x = center_x_ratio-buffer_center_x_ratio
                            delta_y = center_y_ratio-buffer_center_y_ratio
                            displacement_magnitude = math.sqrt(delta_x**2 + delta_y**2)
                            if displacement_magnitude < min_distance:
                                assigned_id = buffer_id
                                min_distance = displacement_magnitude
                                displacement_x = center_x_ratio-buffer_center_x_ratio
                                displacement_y = center_y_ratio-buffer_center_y_ratio

                    if found:
                        break
            # 沒配對到
            if assigned_id==None:
                buffer_path, buffer_feature_id_list = next(reversed(buffer.items()))
                min_distance = float('inf')
                close_feature = None
                for buffer_feature, buffer_id,buffer_center_x_ratio,buffer_center_y_ratio,buffer_displacement_x,buffer_displacement_y in buffer_feature_id_list:
                    if buffer_displacement_x is not None and buffer_displacement_y is not None:
                        # 先判斷位移方向是否一致
                        temp_displacement_x = center_x_ratio-buffer_center_x_ratio
                        temp_displacement_y = center_y_ratio-buffer_center_y_ratio
                        temp_displacement = torch.tensor([center_x_ratio - buffer_center_x_ratio, 
                                  center_y_ratio - buffer_center_y_ratio], dtype=torch.float32)

                        buffer_displacement = torch.tensor([buffer_displacement_x, buffer_displacement_y], dtype=torch.float32)

                        # 计算内积
                        dot_product = torch.dot(temp_displacement, buffer_displacement)
                        if dot_product < 0:
                            continue

                        predict_center_x = buffer_center_x_ratio+buffer_displacement_x
                        predict_center_y = buffer_center_y_ratio+buffer_displacement_y

                        delta_x = center_x_ratio-predict_center_x
                        delta_y = center_y_ratio-predict_center_y
                        displacement_magnitude = math.sqrt(delta_x**2 + delta_y**2)

                        if displacement_magnitude < min_distance:
                            assigned_id = buffer_id
                            min_distance = displacement_magnitude
                            displacement_x = center_x_ratio-buffer_center_x_ratio
                            displacement_y = center_y_ratio-buffer_center_y_ratio
                            close_feature = buffer_feature
                            
                if assigned_id ==None:
                    assigned_id = id_counter
                    id_counter += 1
                else:
                    similarity = cosine_similarity(feature,close_feature)
                    similarity = similarity.squeeze()
                    if similarity < single_camera_thresholds[time_range][cam_id]/2:
                        assigned_id = id_counter
                        id_counter += 1
                 

                # if max_similarity is None or max_similarity < single_camera_thresholds[time_range][cam_id]:
                #     assigned_id = id_counter
                #     id_counter += 1
            temp_feature_assignments.append((feature, assigned_id,center_x_ratio,center_y_ratio,displacement_x,displacement_y))

        # 處理同一個frame出現相同id情況


        # Loop until there are no duplicate indices left.
        cont = 0
        while True:
            cont += 1
            # Build a mapping from assigned_id to the indices in temp_feature_assignments
            assigned_id_dict = defaultdict(list)
            for index, (_, assigned_id, _, _, _, _) in enumerate(temp_feature_assignments):
                assigned_id_dict[assigned_id].append(index)
                        
            # Identify any IDs that appear more than once.
            duplicate_indices = {key: value for key, value in assigned_id_dict.items() if len(value) > 1}

            # If there are no duplicates, exit the loop.
            if not duplicate_indices:
                break

            # Resolve duplicates by keeping one "best" index per duplicate_id.
            for duplicate_id, indices in list(duplicate_indices.items()):
                # Get the latest buffer entry (you may choose a different strategy here)
                buffer_path, buffer_feature_id_list = next(reversed(buffer.items()))
                keep_index = -1

                # Find the best candidate among the duplicate indices.
                for buffer_feature, buffer_id, buffer_center_x_ratio, buffer_center_y_ratio, buffer_displacement_x, buffer_displacement_y in buffer_feature_id_list:
                    if buffer_id != duplicate_id:
                        continue

                    # Use displacement if available.
                    if buffer_displacement_x is not None and buffer_displacement_y is not None:
                        min_distance = float('inf')
                        for idx in indices:
                            predict_center_x = buffer_center_x_ratio + buffer_displacement_x
                            predict_center_y = buffer_center_y_ratio + buffer_displacement_y

                            delta_x = temp_feature_assignments[idx][2] - predict_center_x
                            delta_y = temp_feature_assignments[idx][3] - predict_center_y
                            displacement_magnitude = math.sqrt(delta_x**2 + delta_y**2)

                            if displacement_magnitude < min_distance:
                                keep_index = idx
                                min_distance = displacement_magnitude
                    else:
                        max_similarity = -1
                        for idx in indices:
                            similarity = cosine_similarity(temp_feature_assignments[idx][0], buffer_feature)
                            similarity = similarity.squeeze().item()  
                            if similarity > max_similarity:
                                keep_index = idx
                                max_similarity = similarity

                # Remove the chosen index (keep_index) from the list so that only the others remain for reassignment.
                if keep_index in duplicate_indices[duplicate_id]:
                    duplicate_indices[duplicate_id].remove(keep_index)

            # For all remaining duplicate indices, reassign new IDs.

            for duplicate_id, indices in list(duplicate_indices.items()):
                buffer_path, buffer_feature_id_list = next(reversed(buffer.items()))
                for idx in indices:
                    feature, _, center_x_ratio, center_y_ratio, displacement_x, displacement_y = temp_feature_assignments[idx]

                    new_id = None
                    similarity_matrix = []
                    # Check against the buffer to see if there's a close candidate.
                    for buffer_feature, buffer_id, buffer_center_x_ratio, buffer_center_y_ratio, buffer_displacement_x, buffer_displacement_y in buffer_feature_id_list:
                        if buffer_id == duplicate_id:
                            continue
                        similarity = cosine_similarity(feature, buffer_feature)
                        similarity = similarity.squeeze().item()
                        similarity_matrix.append((similarity, buffer_id,buffer_center_x_ratio,buffer_center_y_ratio))
                    
                    # Sort the similarity matrix from high to low based on similarity value.
                    sorted_similarity = sorted(similarity_matrix, key=lambda x: x[0], reverse=True)
                    
                    # 取得第 cont 高的相似度 (the cont-th highest value)
                    if len(sorted_similarity) >= cont:
                        candidate_similarity, candidate_buffer_id,candidate_buffer_center_x_ratio, candidate_buffer_center_y_ratio= sorted_similarity[cont - 1]
                        # 比較該相似度與 single_camera_thresholds[time_range][cam_id]
                        if candidate_similarity > single_camera_thresholds[time_range][cam_id]:
                            new_id = candidate_buffer_id
                            displacement_x = center_x_ratio-candidate_buffer_center_x_ratio
                            displacement_y = center_y_ratio-candidate_buffer_center_y_ratio
                    
                    # If no candidate was close enough, assign a new id.
                    if new_id is None:
                        new_id = id_counter
                        id_counter += 1
                    
                    # Update the assignment in temp_feature_assignments.
                    

                    temp_feature_assignments[idx] = (feature, new_id, center_x_ratio, center_y_ratio, displacement_x, displacement_y)
            
            # The loop will recompute duplicate_indices in the next iteration.
        first_frame = False
        buffer[label_path].extend(temp_feature_assignments)

    while buffer:
        save_buffer_to_storage(buffer,storage_reverse)
        # write_buffer_to_disk(buffer)

    output = merge_storages(storage_forward,storage_reverse)

    write_storage(output,storage_forward,storage_reverse)

    



    #     for feature in feature_list:
    #         assigned_id = None
    #         max_similarity = -1
    #         if first_frame:
    #             assigned_id = id_counter
    #             id_counter += 1

    #         else:
                       
    #             for buffer_feature_id_list in buffer.values():
    #                 for buffer_feature, buffer_id in buffer_feature_id_list:
    #                     if buffer_feature is not None:
    #                         # print('feature : ',feature)
    #                         # print(buffer_feature)

    #                         similarity = cosine_similarity(feature,buffer_feature)
    #                         # print(similarity)
    #                         similarity = similarity.squeeze()
    #                         # print(similarity)

    #                         if similarity > max_similarity :
    #                             max_similarity = similarity
    #                             assigned_id = buffer_id
                
    #             if max_similarity is None or max_similarity < single_camera_thresholds[time_range][cam_id]:
    #                 assigned_id = id_counter
    #                 id_counter += 1
    #         temp_feature_assignments.append((feature, assigned_id))
    #         # buffer[label_path].append((feature, assigned_id))

    #     first_frame = False
    #     buffer[label_path].extend(temp_feature_assignments)

    # while buffer:
    #     write_buffer_to_disk(buffer)


# 1.與buffer配對後 當沒有偵測到匹配車輛的時候 理論上要給新的id 但這時 先判斷未匹配的車輛的位置是否跟之前 車兩位移的位置相近 降低threshold値在配對一次
# buffer 還要除存車輛前一刻的位置+位移資訊
# 正向tracking + 逆向tracking

            





    


