import os
import argparse
from pathlib import Path
from datetime import datetime



from utils import (
    load_yaml_config, 
    find_dicom_directories,
    del_file
)
from process import MRIPreprocessor



def main(config_path:str):
    today_date = datetime.today().strftime('%Y-%m-%d')

    config = load_yaml_config(config_path)
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


        # Classification 


   

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)
    args = parser.parse_args()

    main(config_path=args.config_path)