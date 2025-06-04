import os
import math
import numpy as np

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch
from .utils import load_data, get_transform


numpy_to_torch_dtype_map = {
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
}


def max_per_memory(image_shape:tuple, dtype:np.dtype, max_memory:float):
    bytes_per_pixel = np.dtype(dtype).itemsize
    size_of_one_image_bytes = np.prod(image_shape) * bytes_per_pixel
    byte_in_gigabytes = 9.3132257461548E-10
    return int(max_memory // (size_of_one_image_bytes * byte_in_gigabytes))


def create_chunks(root_folder_path:str, partition_path:str, output_folder_path:str, image_size:tuple, image_dtype:np.dtype, max_chunk_memory:float):
    """
    Splits a large image dataset into smaller chunks and saves them as .pt files.

    Args:
        root_folder_path (str): Path to the folder containing the images.
        partition_path (str): Path to the file that defines dataset partitions (e.g., train/test split).
        output_folder_path (str): Folder where the chunk files will be saved.
        image_size (tuple): Size of each image as (C, D, W, H).
        image_dtype (np.dtype): Data type of the images (e.g., np.float32).
        max_chunk_memory (float): Maximum memory (in gigabytes) allowed per chunk.

    Returns:
        None
    """
    os.makedirs(output_folder_path, exist_ok=True)
    image_paths, labels = load_data(root_folder_path, partition_path)
    
    if len(image_paths) != len(labels):
        raise ValueError(f"Number of images ({len(image_paths)}) does not match number of labels ({len(labels)}).")
    
    image_count = len(image_paths)
    image_indices = np.arange(image_count)

    C, D, W, H = image_size
    max_images_per_chunk = max_per_memory(image_size, image_dtype, max_chunk_memory)
    number_of_chunks = math.ceil(image_count / max_images_per_chunk)
    image_dtype_torch = numpy_to_torch_dtype_map[image_dtype]
    max_zero_pad = len(str(number_of_chunks))
    for chunk_id in range(number_of_chunks):
        if chunk_id == (number_of_chunks - 1):
            max_images_per_chunk -= (number_of_chunks * max_images_per_chunk) - image_count

        image_buffer = torch.empty([max_images_per_chunk, C, D, W, H], dtype=image_dtype_torch) 

        label_buffer = torch.empty([max_images_per_chunk], dtype=torch.int64) 
        transform = get_transform()
        start_img_idx = max_images_per_chunk * chunk_id
        end_img_idx = start_img_idx + max_images_per_chunk
        for idx, img_idx in enumerate(image_indices[start_img_idx:end_img_idx]):
            img_tensor =  transform(image_paths[img_idx])
            image_buffer[idx] = img_tensor.to(dtype=image_dtype_torch)
            label_buffer[idx] = labels[img_idx]

        padded_chunk_id = str(chunk_id).zfill(max_zero_pad)
        save_path = os.sep.join([output_folder_path, f'chunk_{padded_chunk_id}.pt'])
        torch.save({"images" : image_buffer, "labels" : label_buffer}, save_path)

if __name__ == "__main__":
    create_chunks(
        root_folder_path='F:/ADNI/1.2mm_data',
        partition_path='C:/Users/ropja/Desktop/eval_logs/presentation/cn_mci/val_2.json',
        output_folder_path='F:/test_chunk',
        image_size=(1, 128, 128, 128),
        image_dtype=np.float16,
        max_chunk_memory= 0.00838861
    )


    
