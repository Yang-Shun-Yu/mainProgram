# # from collections import OrderedDict, defaultdict

# # class OrderedDefaultdict(OrderedDict):
# #     def __init__(self, default_factory=None, *args, **kwargs):
# #         if default_factory is not None and not callable(default_factory):
# #             raise TypeError('first argument must be callable or None')
# #         self.default_factory = default_factory
# #         super().__init__(*args, **kwargs)

# #     def __missing__(self, key):
# #         if self.default_factory is None:
# #             raise KeyError(key)
# #         self[key] = value = self.default_factory()
# #         return value
    
# # # Initialize an OrderedDefaultdict with list as the default factory
# # buffer = OrderedDefaultdict(list)
# # # Add items to the buffer
# # buffer['key1'].append('value1')
# # buffer['key1'].append('value2')
# # buffer['key1'].append('value3')
# # buffer['key2'].append('value2')
# # buffer['key3'].append('value3')

# # # Access and remove the oldest item
# # # oldest_key, oldest_value = buffer.popitem(last=False)
# # for v in buffer.values():
# #     print(v)

# # # print(f'Removed: {oldest_key} -> {oldest_value}')

# # import torch
# # data1 = [[0.5906, -0.4366, 0.2052, 0.4422, 0.3192, -0.2488]]
# # tensor1 = torch.tensor(data1)

# # data2 = [[0.5906, 0.4366, 0.2052, 0.4422, 0.3192, -0.2488]]

# # tensor2 = torch.tensor(data2)
# # import torch.nn.functional as F

# # similiarty = F.cosine_similarity(tensor1.unsqueeze(0), tensor2.unsqueeze(0))

# # print(similiarty)

# # a = 0

# # if similiarty>a:
# #     a = similiarty
# #     print(a)

# # feature_list =  [[]]

# # if feature_list == [[]]:
# #     print("feature_list contains a single empty list.")
# # else:
# #     print("feature_list does not contain a single empty list.")
# filename = '1016_150000_151900/0_00001.txt'
# filename = '/home/eddy/Desktop/MasterThesis/mainProgram/AICUP_datasets/test/labels/1016_150000_151900/0_00001.txt'

# result1 = filename.split('/')[-2]
# result1 = "_".join(result1.split('_')[1:])
# result2 = filename.split('/')[-1]
# result2 = result2.split('_')[0]
# print(result1)  # Output: 0
# print(result2)  # Output: 0

# import sys
# import os
# import torch
# # Add the folder to sys.path
# sys.path.append(os.path.abspath("AICUP_datasets_fine_tune"))

# # Now import the function
# from trainer import prepare_trainer_and_calculate_threshold


# dataset_path = '/home/eddy/Desktop/cropped_image'
# backbone_model = 'swin'
# weights_path = '/home/eddy/Desktop/MasterThesis/mainProgram/AICUP_datasets_fine_tune/swin_center_lr_0.5_loss_3e-4_smmothing_0.1/swin_center_loss_best.pth'
# device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

# single_camera_thresholds,all_camera_thresholds = prepare_trainer_and_calculate_threshold(
#     path=dataset_path,
#     backbone=backbone_model,
#     custom_weights_path=weights_path,
#     device=device_type
# )

# print(f"single_camera_thresholds: {single_camera_thresholds}")
# print(f"all_camera_thresholds: {all_camera_thresholds}")


# import numpy as np

# # 定义两个向量
# vector1 = np.array([1, 2])
# vector2 = np.array([-4, 5])

# # 计算内积
# dot_product = np.dot(vector1, vector2)

# print(f"向量内积: {dot_product}")


# from collections import defaultdict

# # 假設 temp_feature_assignments 是包含 (feature, assigned_id, center_x_ratio, center_y_ratio, displacement_x, displacement_y) 的列表
# temp_feature_assignments = [
#     ("feat1", 1, 0.2, 0.3, 0.1, 0.1),
#     ("feat2", 2, 0.4, 0.5, -0.2, 0.0),
#     ("feat3", 3, 0.6, 0.7, 0.3, -0.1),
#     ("feat4", 4, 0.8, 0.9, -0.4, 0.2),
#     ("feat5", 5, 0.1, 0.2, 0.0, -0.3)
# ]

# # 建立字典來追蹤 assigned_id 的索引
# assigned_id_dict = defaultdict(list)

# # 遍歷 temp_feature_assignments，將 assigned_id 存入字典
# for index, (_, assigned_id, _, _, _, _) in enumerate(temp_feature_assignments):
#     assigned_id_dict[assigned_id].append(index)

# # 找出所有重複的 assigned_id 及其對應索引
# duplicate_indices = {key: value for key, value in assigned_id_dict.items() if len(value) > 1}

# # 修正迴圈，正確遍歷 duplicate_indices 字典
# for duplicate_id, indices in duplicate_indices.items():  # 這裡應該用 .items()
#     for index in indices:
#         print(f"Duplicate ID: {duplicate_id}, Index: {index}")
# # print("重複的 assigned_id 及其索引:")
# # print(duplicate_indices)


import os
from collections import defaultdict

def merge_storages(storage_forward, storage_reverse):
    """
    Merge forward and reverse tracking storages by comparing the clusters.
    
    Both storages are expected to be dictionaries where:
        key: file_path (e.g., '/.../labels/<time>/<cam>_something.txt')
        value: list of tuples (buffer_id, center_x_ratio, center_y_ratio)
    
    The function will:
      1. Cluster entries by time and camera.
      2. For each (time, cam) cluster, compare the forward ids and reverse ids.
         If any coordinate in a reverse id cluster matches one in a forward id cluster 
         (within a tolerance epsilon), then all occurrences of the reverse id will be updated 
         to the forward id.
      3. Merge the updated reverse storage with the forward storage.
    
    Returns:
        merged_storage: a dictionary with the merged results.
    """
    # Create nested dictionaries for clustering:
    # Structure: cluster[time][cam][object_id] = list of (center_x_ratio, center_y_ratio)
    forward_cluster = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    reverse_cluster = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    # Populate forward_cluster
    for file_path, entries in storage_forward.items():
        # Extract time and camera from file_path
        # e.g., if file_path is "/.../labels/1016_190000_191900/2_someFile.txt",
        # then time = "1016_190000_191900" and cam = "2" (first token of the file name)
        parts = file_path.split(os.sep)
        if len(parts) < 2:
            continue  # skip invalid path
        time = parts[-2]
        cam = parts[-1].split('_')[0]
        for buffer_id, center_x, center_y in entries:
            forward_cluster[time][cam][buffer_id].append((center_x, center_y))
    
    # Populate reverse_cluster similarly
    for file_path, entries in storage_reverse.items():
        parts = file_path.split(os.sep)
        if len(parts) < 2:
            continue
        time = parts[-2]
        cam = parts[-1].split('_')[0]
        for buffer_id, center_x, center_y in entries:
            reverse_cluster[time][cam][buffer_id].append((center_x, center_y))
    
    # Use a small tolerance when comparing coordinates
    epsilon = 1e-3
    # Build a mapping: for each (time, cam), map reverse_id to forward_id if a match is found.
    # We'll use a tuple key (time, cam, reverse_id) for clarity.
    id_mapping = {}
    
    # For each time and camera that appear in the forward cluster...
    for time in forward_cluster:
        for cam in forward_cluster[time]:
            # Only compare if the reverse cluster has data for the same time and cam.
            if cam not in reverse_cluster[time]:
                continue
            # Compare each forward id with each reverse id.
            for f_id, f_coords in forward_cluster[time][cam].items():
                for r_id, r_coords in reverse_cluster[time][cam].items():
                    # Check if any coordinate in r_coords matches any in f_coords.
                    match_found = False
                    for (fx, fy) in f_coords:
                        for (rx, ry) in r_coords:
                            if abs(fx - rx) < epsilon and abs(fy - ry) < epsilon:
                                # We consider the object detected in reverse with r_id to be the same as forward f_id.
                                id_mapping[(time, cam, r_id)] = f_id
                                match_found = True
                                break  # break out of innermost loop
                        if match_found:
                            break  # found a match, no need to check more for this pair
    
    # Now update the reverse storage: for each entry, if its id is mapped, update it to the forward id.
    updated_storage_reverse = {}
    for file_path, entries in storage_reverse.items():
        parts = file_path.split(os.sep)
        if len(parts) < 2:
            continue
        time = parts[-2]
        cam = parts[-1].split('_')[0]
        new_entries = []
        for (buffer_id, center_x, center_y) in entries:
            # Replace the reverse id with the forward id if a mapping exists.
            key = (time, cam, buffer_id)
            new_id = id_mapping.get(key, buffer_id)
            new_entries.append((new_id, center_x, center_y))
        updated_storage_reverse[file_path] = new_entries
    
    # Finally, merge the updated reverse storage with the forward storage.
    # Here we simply combine the lists from each storage; if both contain the same file_path,
    # we extend the forward list with the reverse (now updated) list.
    merged_storage = {}
    # Start with storage_forward
    for key, entries in storage_forward.items():
        merged_storage[key] = entries.copy()
    
    # Merge in updated reverse storage
    for key, entries in updated_storage_reverse.items():
        if key in merged_storage:
            merged_storage[key].extend(entries)
        else:
            merged_storage[key] = entries.copy()
    
    return merged_storage

# ---------------------------
# Example usage:
# if __name__ == "__main__":
    # Imagine these storages came from your forward and reverse tracking routines.
    # The keys are file paths and the values are lists of tuples: (id, center_x, center_y).
    # storage_forward = {
    #     "/path/to/labels/1016_190000_191900/2_cam.txt": [
    #         (1, 0.45, 0.55),
    #         (2, 0.70, 0.80)
    #     ]
    # }
    
    # storage_reverse = {
    #     "/path/to/labels/1016_190000_191900/2_cam.txt": [
    #         (20, 0.45, 0.55),  # This reverse id should match forward id 1
    #         (20, 0.70, 0.80)
    #     ]
    # }
    
    # merged = merge_storages(storage_forward, storage_reverse)
    # for file, entries in merged.items():
    #     print(f"{file}:")
    #     for entry in entries:
    #         print("  ", entry)
    # a = [1,2,3,4,5]

    # print(a[:-1])

    # test = '/home/eddy/Desktop/train/test/images/1016_190000_191900/6_00087.txt'
    # parts = test.split(os.sep)
    # cam = parts[-1].split('_')[0]
    # frame = int(parts[-1].split('_')[1].split('.')[0])

    # print(frame)

    # a = [1,2,3,4,5,6]

    # for index,v in enumerate(a):
    #     print(index)


    # label_path = '/home/eddy/Desktop/MasterThesis/mainProgram/AICUP_datasets/test/labels/1016_150000_151900/6_00350.txt'
    # time_range = os.path.basename(os.path.dirname(label_path))
    # print(time_range)
    # import math

    # cand_disp_y = 1
    # cand_disp_x = 1
    # a = -1
    # b = -0.5

    # # Calculate angles (in radians) for the predicted displacement (buffer) and the actual displacement.
    # angle_pred = math.atan2(cand_disp_y, cand_disp_x)
    # angle_actual = math.atan2(b, a)

    # # Compute the absolute difference between the angles.
    # angle_diff = abs(angle_pred - angle_actual)
    # print(angle_diff)
    # # Normalize the angle difference to be within [0, pi]
    # if angle_diff > math.pi:
    #     angle_diff = 2 * math.pi - angle_diff

    # print(angle_diff)

    # # Set an acceptable threshold (e.g., 30 degrees in radians).
    # angle_threshold = math.radians(30)
import os



def update_merge_labels(merge_labels_folder, labels_folder):
    for subfolder in os.listdir(merge_labels_folder):
        merge_subfolder_path = os.path.join(merge_labels_folder, subfolder)
        
        # Ensure the subfolder is a directory
        if not os.path.isdir(merge_subfolder_path):
            continue
        
        for txt_file in os.listdir(merge_subfolder_path):
            merge_file_path = os.path.join(merge_subfolder_path, txt_file)

            # Read the contents of merge_labels file
            with open(merge_file_path, "r") as merge_file:
                merge_lines = merge_file.readlines()

            # Remove lines that contain only 'None\n'
            cleaned_lines = [line for line in merge_lines if line.strip() != "None"]

            # If all lines were 'None\n', the file should be empty
            if not cleaned_lines:
                # print(f"File contains only 'None': {merge_file_path}. Clearing it.")
                with open(merge_file_path, "w") as merge_file:
                    merge_file.write("")  # Write nothing, making it empty
            elif cleaned_lines != merge_lines:  # Only write back if changes were made
                print(f"Removing 'None' lines from: {merge_file_path}")
                with open(merge_file_path, "w") as merge_file:
                    merge_file.write("\n".join(cleaned_lines) + "\n")


    # Iterate through each subfolder in merge_labels
    for subfolder in os.listdir(merge_labels_folder):
        merge_subfolder_path = os.path.join(merge_labels_folder, subfolder)
        labels_subfolder_path = os.path.join(labels_folder, subfolder)

        # Check if corresponding subfolder exists in labels
        if not os.path.exists(labels_subfolder_path):
            print(f"Skipping {subfolder}, corresponding folder not found in labels.")
            continue

        # Iterate through all txt files in merge_labels subfolder
        for txt_file in os.listdir(merge_subfolder_path):
            merge_file_path = os.path.join(merge_subfolder_path, txt_file)
            labels_file_path = os.path.join(labels_subfolder_path, txt_file)

            # Check if the corresponding labels file exists
            if not os.path.exists(labels_file_path):
                print(f"Skipping {txt_file} in {subfolder}, corresponding file not found in labels.")
                continue
            
            # Read the contents of labels file
            with open(labels_file_path, "r") as labels_file:
                labels_lines = labels_file.readlines()

            # Read the contents of merge_labels file
            with open(merge_file_path, "r") as merge_file:
                merge_lines = merge_file.readlines()

            # Ensure merge_labels has the same number of lines as labels

            if len(labels_lines) != len(merge_lines):
                print(f"Skipping {txt_file} in {subfolder}, mismatch in line count between labels and merge_labels.")
                continue

            # Process each line
            updated_lines = []
            for i, label_line in enumerate(labels_lines):
                label_parts = label_line.strip().split()
                merge_value = merge_lines[i].strip()  # The value from merge_labels

                # Replace the last value in label_parts with the merge_value
                # label_parts[-1] = merge_value
                label_parts[-1] = str(int(label_parts[-1]))

                # Reconstruct the line and add to the list
                # updated_lines.append(" ".join(label_parts))
                updated_lines.append(label_parts[-1])

            # Write the modified content back to merge_labels file
            with open(merge_file_path, "w") as merge_file:
                merge_file.write("\n".join(updated_lines) + "\n")

            # print(f"Updated {txt_file} in {subfolder}")

# Example usage
merge_labels_folder = "/home/eddy/Desktop/MasterThesis/mainProgram/reid_tracking/merge_labels"
labels_folder = "/home/eddy/Desktop/train/test/labels"
update_merge_labels(merge_labels_folder, labels_folder)


# def update_merge_labels(labels_folder):
#     for subfolder in os.listdir(labels_folder):
#         labels_subfolder_path = os.path.join(labels_folder, subfolder)
        
#         # Ensure the subfolder is a directory

        
#         for txt_file in os.listdir(labels_subfolder_path):
#             labels_file_path = os.path.join(labels_subfolder_path, txt_file)

#             # Read the contents of merge_labels file

#             with open(labels_file_path, "r") as labels_file:
#                 labels_lines = labels_file.readlines()
#             # Process each line
#             updated_lines = []
#             for i, label_line in enumerate(labels_lines):
#                 label_parts = label_line.strip().split()


#                 # Replace the last value in label_parts with the merge_value
#                 label_parts[-1] = str(int(label_parts[-1]) - 4287)


#                 # Reconstruct the line and add to the list
#                 updated_lines.append(" ".join(label_parts))

#             # Write the modified content back to merge_labels file
#             with open(labels_file_path, "w") as labels_file:
#                 labels_file.write("\n".join(updated_lines) + "\n")


# merge_labels_folder = "/home/eddy/Desktop/MasterThesis/mainProgram/reid_tracking/merge_labels"
# labels_folder = "/home/eddy/Desktop/train/test/labels"
# update_merge_labels(labels_folder)

