import os
import json
import argparse
from pathlib import Path
from datetime import datetime

import nibabel as nib

from utils import (
    load_yaml_config, 
    find_dicom_directories,
    del_file, 
    load_model, 
    get_transform_clean_tensor,
    get_transform_resample_tensor
)
from process import MRIPreprocessor

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch
from monai.networks.nets import resnet18
import ants
import numpy as np


import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

class P:

    def __init__(self, run_id:str, config_path:str):
        self._run_id = run_id
        self._config = load_yaml_config(config_path)
        self._setup_paths()
        self._setup_model()
        self._predicted_labels = {}



    def _setup_paths(self):
        self._mni_template = self._config['mni_template_path']
        self._save_nifti_format = self._config['save_nifti_format']
        self._save_brain_mask = self._config['save_brain_mask']

        self._root_output_folder = os.sep.join([self._config['output_root_folder_path'], self._run_id])
        os.makedirs(self._root_output_folder, exist_ok=True)


    def _setup_model(self):
        spatial_dims = 3
        n_input_channels = 1
        num_classes = 5
        
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model =  resnet18(
            spatial_dims=spatial_dims, 
            n_input_channels=n_input_channels, 
            num_classes=num_classes
        ).to(self._device)
        self._model = load_model(self._model, self._config['model_path'])


    def _clean_data(self,
                preprocessor:MRIPreprocessor, 
                folder_path:str, 
                nifti_file_path:str, 
                processed_file_path:str, 
                processed_file_mask_path:str
            ):
        try:
            preprocessor.dicom_to_nifti(folder_path, nifti_file_path) \
            .load_mri(nifti_file_path) \
            .reorient_image('LAS') \
            .skull_strip() \
            .register_to_mni(self._mni_template, 'Affine') \
            .save_mri(processed_file_path, processed_file_mask_path)
        except Exception as e:
            print(f'Error processing {folder_path}: {e}')
        

    def _save_predicted(self):
        json_file_path = os.sep.join([self._root_output_folder, 'predicted_labels.json'])
        with open(json_file_path, 'w', encoding ='utf8') as json_file:
            json.dump(self._predicted_labels, json_file, indent=6)


    def _classification(self, scan_id, image_data):
        with torch.no_grad():
            input_image = image_data.to(self._device)
            model_out = self._model(input_image)
            model_out_argmax = model_out.argmax(dim=1)
            predicted_label = model_out_argmax

        self._predicted_labels[scan_id] = predicted_label.item()

    
    def process(self):
        data_paths = find_dicom_directories(self._config['raw_mri_root_folder_path'])
        preprocessor = MRIPreprocessor(verbose=True) 
        for folder_path in data_paths:
            try:
                scan_id = Path(folder_path).stem
                output_folder_path = os.sep.join([self._root_output_folder, scan_id])
                os.makedirs(output_folder_path, exist_ok=True)

                nifti_file_path = os.sep.join([output_folder_path, f'{scan_id}.nii.gz'])
                processed_file_path = os.sep.join([output_folder_path, f'{scan_id}_processed.nii.gz'])  
                processed_file_mask_path = None
                if self._save_brain_mask:
                    processed_file_mask_path = os.sep.join([output_folder_path, f'{scan_id}_mask.nii.gz'])


                self._clean_data(preprocessor, 
                   folder_path, 
                   nifti_file_path, 
                   processed_file_path, 
                   processed_file_mask_path
                )

                # Data ...
                transform = get_transform_clean_tensor()
                mri_tensor_cleaned = transform(processed_file_path)

                resample_tensor_transform = get_transform_resample_tensor()
                image_data = resample_tensor_transform(mri_tensor_cleaned).unsqueeze(0)

                self._classification(scan_id, image_data)

                # GradCam
                # ...

            except Exception as e:
                print(e)
            finally:
                self._save_predicted()



def main(run_id:str, config_path:str):
    q = P(run_id, config_path)
    q.process()
            
        


        # Remove channel dimension if it's 1
        # if mri_tensor.shape[0] == 1:
        #     mri_tensor = mri_tensor.squeeze(0)  # Now shape is [D, H, W]

        # # Convert to NumPy array
        # mri_np = mri_tensor.numpy()

        # # Get dimensions
        # D, H, W = mri_np.shape

        # # Initial slice indices
        # init_axial = D // 2
        # init_coronal = H // 2
        # init_sagittal = W // 2

        # Create figure and axes
        # fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        # plt.subplots_adjust(bottom=0.25)

        # # Display initial slices
        # axial_im = axes[0].imshow(mri_np[init_axial, :, :], cmap='gray')
        # axes[0].set_title(f'Axial Slice {init_axial}')
        # axes[0].axis('off')

        # coronal_im = axes[1].imshow(mri_np[:, init_coronal, :], cmap='gray')
        # axes[1].set_title(f'Coronal Slice {init_coronal}')
        # axes[1].axis('off')

        # sagittal_im = axes[2].imshow(mri_np[:, :, init_sagittal], cmap='gray')
        # axes[2].set_title(f'Sagittal Slice {init_sagittal}')
        # axes[2].axis('off')

        # # Define slider axes
        # axcolor = 'lightgoldenrodyellow'
        # axial_ax = plt.axes([0.15, 0.1, 0.65, 0.03], facecolor=axcolor)
        # coronal_ax = plt.axes([0.15, 0.06, 0.65, 0.03], facecolor=axcolor)
        # sagittal_ax = plt.axes([0.15, 0.02, 0.65, 0.03], facecolor=axcolor)

        # # Create sliders
        # axial_slider = Slider(axial_ax, 'Axial Slice', 0, D - 1, valinit=init_axial, valstep=1)
        # coronal_slider = Slider(coronal_ax, 'Coronal Slice', 0, H - 1, valinit=init_coronal, valstep=1)
        # sagittal_slider = Slider(sagittal_ax, 'Sagittal Slice', 0, W - 1, valinit=init_sagittal, valstep=1)

        # # Update functions
        # def update_axial(val):
        #     idx = int(axial_slider.val)
        #     axial_im.set_data(mri_np[idx, :, :])
        #     axes[0].set_title(f'Axial Slice {idx}')
        #     fig.canvas.draw_idle()

        # def update_coronal(val):
        #     idx = int(coronal_slider.val)
        #     coronal_im.set_data(mri_np[:, idx, :])
        #     axes[1].set_title(f'Coronal Slice {idx}')
        #     fig.canvas.draw_idle()

        # def update_sagittal(val):
        #     idx = int(sagittal_slider.val)
        #     sagittal_im.set_data(mri_np[:, :, idx])
        #     axes[2].set_title(f'Sagittal Slice {idx}')
        #     fig.canvas.draw_idle()

        # # Connect sliders to update functions
        # axial_slider.on_changed(update_axial)
        # coronal_slider.on_changed(update_coronal)
        # sagittal_slider.on_changed(update_sagittal)

        # plt.show()

        
if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=str, required=True)
    parser.add_argument('--config_path', type=str, required=True)
    args = parser.parse_args()

    main(run_id=args.run_id, config_path=args.config_path)