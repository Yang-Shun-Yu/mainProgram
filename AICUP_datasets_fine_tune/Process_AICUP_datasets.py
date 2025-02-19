# import os
# from PIL import Image

# def crop_and_save_images(input_dir,output_dir):
#     """
#     Crop images based on labels and save the cropped images.

#     Args:
#         images_dir (str): Path to the directory containing images.
#         labels_dir (str): Path to the directory containing label files.
#         output_dir (str): Path to the directory where cropped images will be saved.
#     """
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
    
#     images_dir = os.path.join(input_dir,'images')
#     labels_dir = os.path.join(input_dir,'labels')

#     for sub_labels_dir in os.listdir(labels_dir):

#         sub_labels_dir_path = os.path.join(labels_dir,sub_labels_dir)
#         sub_images_dir_path = os.path.join(images_dir,sub_labels_dir)

#         for label_file in os.listdir(sub_labels_dir_path):


#             # Ensure we're only processing .txt files

#             if not label_file.endswith('.txt'):
#                 continue
#             label_path = os.path.join(sub_labels_dir_path,label_file)
#             image_path = os.path.join(sub_images_dir_path,label_file.replace('.txt','.jpg'))
#             if not os.path.exists(image_path):
#                 print(f"Image not found: {image_path}")
#                 continue


#             with open(label_path,'r') as f:
#                 lines = f.readlines()
#             for line in lines:
#                 parts = line.strip().split()

#                 class_id = int(parts[0])
#                 center_x = float(parts[1])
#                 center_y = float(parts[2])
#                 width = float(parts[3])
#                 height = float(parts[4])
#                 track_id = int(parts[5])

#                 with Image.open(image_path) as img:
#                     img_width, img_height = img.size
#                     left = int((center_x-width/2)*img_width)
#                     right = int((center_x+width/2)*img_width)
#                     top = int((center_y-height/2)*img_height)
#                     bottom = int((center_y+height/2)*img_height)

#                     cropped_img = img.crop((left,top,right,bottom))
#                     cropped_image_name = f"{sub_labels_dir}_{label_file.replace('.txt','')}_{track_id}.jpg"
#                     cropped_image_path = os.path.join(output_dir, cropped_image_name)

#                     cropped_img.save(cropped_image_path)
#                     print(f"Saved cropped image: {cropped_image_path}")





# # Example usage
# input_dir = "/home/eddy/Desktop/train/train"
# output_dir = "/home/eddy/Desktop/train/train/cropped_images"

# crop_and_save_images(input_dir, output_dir)


import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_label_file(label_file_path, image_file_path, sub_labels_dir, output_dir):
    """
    Process a single label file: open its corresponding image once,
    crop all bounding boxes, then save them.
    """
    # If the image doesn’t exist, skip
    if not os.path.exists(image_file_path):
        return [f"Image not found: {image_file_path}"]

    # Read label lines
    with open(label_file_path, 'r') as f:
        lines = f.readlines()
    if not lines:
        return [f"No bounding boxes in: {label_file_path}"]

    # Open the image once
    with Image.open(image_file_path) as img:
        img_width, img_height = img.size

        saved_images_info = []
        base_name = os.path.basename(label_file_path).replace('.txt', '')

        # Crop each bounding box
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 6:
                # skip lines that don't have enough info
                continue

            class_id = int(parts[0])
            center_x = float(parts[1])
            center_y = float(parts[2])
            width    = float(parts[3])
            height   = float(parts[4])
            track_id = int(parts[5])

            left   = int((center_x - width / 2)  * img_width)
            right  = int((center_x + width / 2)  * img_width)
            top    = int((center_y - height / 2) * img_height)
            bottom = int((center_y + height / 2) * img_height)

            cropped_img = img.crop((left, top, right, bottom))

            cropped_image_name = f"{sub_labels_dir}_{base_name}_{track_id}.jpg"
            cropped_image_path = os.path.join(output_dir, cropped_image_name)

            cropped_img.save(cropped_image_path)
            saved_images_info.append(f"Saved: {cropped_image_path}")

    return saved_images_info


def crop_and_save_images(input_dir, output_dir, max_workers=4):
    """
    Crop images based on labels and save the cropped images, in parallel.

    Args:
        input_dir (str): Path to the directory containing 'images' and 'labels' subdirs.
        output_dir (str): Path where cropped images will be saved.
        max_workers (int): Number of threads/processes to use for parallelization.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images_dir = os.path.join(input_dir, 'images')
    labels_dir = os.path.join(input_dir, 'labels')

    # A container for Future tasks
    futures = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for sub_labels_dir in os.listdir(labels_dir):
            sub_labels_dir_path = os.path.join(labels_dir, sub_labels_dir)
            sub_images_dir_path = os.path.join(images_dir, sub_labels_dir)

            # Skip if not a directory
            if not os.path.isdir(sub_labels_dir_path):
                continue

            for label_file in os.listdir(sub_labels_dir_path):
                if not label_file.endswith('.txt'):
                    continue

                label_path = os.path.join(sub_labels_dir_path, label_file)
                image_path = os.path.join(sub_images_dir_path, label_file.replace('.txt', '.jpg'))

                # Submit a parallel job to process each label file
                future = executor.submit(
                    process_label_file,
                    label_path,
                    image_path,
                    sub_labels_dir,
                    output_dir
                )
                futures.append(future)

        # Collect results (and handle them if you like)
        for future in as_completed(futures):
            results = future.result()
            # You could log these or save them to a file instead
            # to minimize console I/O overhead.
            for r in results:
                print(r)

# Example usage
if __name__ == "__main__":
    input_dir = "/home/eddy/Desktop/train/test"
    output_dir = "/home/eddy/Desktop/train/test/test"
    
    # Increase or decrease max_workers depending on your system
    crop_and_save_images(input_dir, output_dir, max_workers=16)
