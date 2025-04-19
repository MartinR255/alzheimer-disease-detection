import os
import argparse
from pathlib import Path
from datetime import datetime



from utils import (
    load_yaml_config, 
    find_dicom_directories,
    del_file, 
    load_model, 
    get_transform
)
from process import MRIPreprocessor

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch
from monai.networks.nets import resnet18, resnet34

from unlabeled_dataset import UnlabeledDataset
from monai.data import DataLoader



def main(config_path:str):





    today_date = datetime.today().strftime('%Y-%m-%d')

    config = load_yaml_config(config_path)

    # Setup model
    spatial_dims = 3
    n_input_channels = 1
    num_classes = 5
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model =  resnet34(
        spatial_dims=spatial_dims, 
        n_input_channels=n_input_channels, 
        num_classes=num_classes
    ).to(device)
    model = load_model(model, config['model_path'])


    # Setup preprocessing/data cleaning
    mni_template = config['mni_template_path']
    save_nifti_format = config['save_nifti_format']
    save_brain_mask = config['save_brain_mask']

    root_output_folder = os.sep.join([config['output_root_folder_path'], today_date])
    os.makedirs(root_output_folder, exist_ok=True)

    preprocessor = MRIPreprocessor(verbose=True) 
    data_paths = find_dicom_directories(config['raw_mri_root_folder_path'])
    for folder_path in data_paths:
        scan_id = Path(folder_path).stem
    
        output_folder_path = os.sep.join([root_output_folder, scan_id])
        os.makedirs(output_folder_path, exist_ok=True)

        nifti_file_path = os.sep.join([output_folder_path, f'{scan_id}.nii.gz'])
        processed_file_path = os.sep.join([output_folder_path, f'{scan_id}_processed.nii.gz']) 
       
        processed_file_mask_path = None
        if save_brain_mask:
            processed_file_mask_path = os.sep.join([output_folder_path, f'{scan_id}_mask.nii.gz'])

        # Data cleaning
        try:
            preprocessor.dicom_to_nifti(folder_path, nifti_file_path) \
                        .reorient_image() \
                        .skull_strip() \
                        .register_to_mni(mni_template, 'Affine') \
                        .save_mri(processed_file_path, processed_file_mask_path)
        except Exception as e:
            print(f'Error processing {folder_path}: {e}')


        # Data preprocessing
        processed_image = None

        # Classification 
        model.eval()
        predicted_label = None
        with torch.no_grad():
            inputs = processed_image.to(device)
            model_out = model(inputs)
            model_out_argmax = model_out.argmax(dim=1)
            predicted_label = model_out_argmax.cpu().numpy()




   

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)
    args = parser.parse_args()

    main(config_path=args.config_path)