import os
import shutil
import random
import numpy  as     np
from   PIL    import Image

# =========================================================================
# (i)   Select a specified number of files of each class from a dataset,
# (ii)  make a NPY file with all the selected files,
# (iii) copy the selected files to a new folder.
# =========================================================================

def create_image_dataset(dataset_path, nclasses, files_per_class, npy_file, output_path):
    all_data       = []
    selected_files = {}

    # Get list of class subfolders
    class_folders = sorted(
        [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
    )[:nclasses]

    for id, class_folder in enumerate(class_folders):
        class_path     = os.path.join(dataset_path, class_folder)
        image_files    = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
        selected_files[id] = random.sample(image_files, min(files_per_class, len(image_files)))

        # Calculate the minimum number of files per class
        num_files = len(selected_files[id])
        if id == 0:
            min_files_class = num_files
        else:
            if (num_files < min_files_class):
                min_files_class = num_files

    for it in range(min_files_class):
        for cls, class_folder in enumerate(class_folders):
            class_path = os.path.join(dataset_path, class_folder)
            file_name  = selected_files[cls][it]
            file_path  = os.path.join(class_path, file_name)

            #print(f'iter: {it} class: {cls} image file: {file_path}')

            with Image.open(file_path) as img:
                img_array = np.array(img)
                all_data.append(img_array)

            shutil.copy(file_path, output_path)

    # Convert list to numpy array
    data_array = np.array(all_data)

    print(f'Array of images shape: {data_array.shape}')

    # Save to npy file
    np.save(npy_file, data_array)

    print(f"Saved {len(all_data)} images to {npy_file}")


if __name__ == "__main__":
    results_path    = 'OUR_WORK_DIR_HERE/results'
    dataset_path    = 'OUR_DATASETS_ROOT/cifar10_64x64/test'  # The actual dataset path
    nclasses        = 10          # Number of class subfolders to process
    files_per_class = 560         # Number of images to select per class
    npy_file        = f'{results_path}/cifar10test_5600_real_images.npy'  # Output file name
    output_path     = f'{results_path}/cifar10test_5600_real_images'      # Folder to copy the selected real images

    # Create folder to copy the selected real images
    os.makedirs(f'{results_path}/cifar10test_5600_real_images', exist_ok=True)

    create_image_dataset(dataset_path, nclasses, files_per_class, npy_file, output_path)
