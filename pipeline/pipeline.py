import os
import json
import argparse
from pathlib import Path

from utils import (
    load_yaml_config, 
    find_dicom_directories,
    get_transform_clean_tensor,
    get_transform_resample_tensor,
    get_network
)
from process import MRIPreprocessor

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch
from monai.networks.nets import resnet18

from captum.attr import GuidedGradCam, Occlusion

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from medcam import medcam

from monai.transforms import Resize
import ants

class Pipeline:

    def __init__(self, run_id:str, config_path:str):
        self._run_id = run_id
        self._config = load_yaml_config(config_path)
        self._setup_paths()
        self._setup_model()
        self._setup_interpretability_methods()
        self._predicted_labels = {}



    def _setup_paths(self):
        self._mni_template = self._config['mni_template_path']
        self._save_nifti_format = self._config['save_nifti_format']
        self._save_brain_mask = self._config['save_brain_mask']
        self._root_output_folder = os.sep.join([self._config['output_root_folder_path'], self._run_id])
        os.makedirs(self._root_output_folder, exist_ok=True)


    def _setup_model(self):
        model_config = load_yaml_config(self._config['model_config_path'])
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = get_network(model_config['model']).to(self._device)


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
            model_out, gradcam_plus_plus_heatmap = self._model(input_image)
            model_out_argmax = model_out.argmax(dim=1)
            predicted_label = model_out_argmax.item()
      
        self._predicted_labels[scan_id] = predicted_label
        return predicted_label, gradcam_plus_plus_heatmap
    

    def _setup_interpretability_methods(self):
        layers = [self._model.layer4[-1]]
        self._gradcam = GradCAM(model=self._model, target_layers=layers)
        self._guided_gradcam = GuidedGradCam(model=self._model, layer=self._model.layer4[-1])
        self._occlusion = Occlusion(self._model)

        # Inject model with M3d-CAM for GradCAM++
        self._model = medcam.inject(self._model, backend='gcampp', layer='layer4', return_attention=True)

    
    def _get_guided_gradcam_heatmap(self, image_data, predicted_label, resize_target=None):
        image_data = image_data.to(self._device)
        image_data = image_data.requires_grad_(True)
        heatmap = self._guided_gradcam.attribute(image_data, predicted_label)

        heatmap = heatmap.detach().squeeze(0).cpu()
        if resize_target:
            resize_transform = Resize(spatial_size=resize_target, mode='trilinear')
            heatmap = resize_transform(heatmap)

        return heatmap.numpy()


    def _get_gradcam_heatmap(self, image_data, predicted_label, resize_target=None):
        """
        Generates a GradCAM heatmap.

        Args:
            image_data (Tensor): Input image tensor.
            predicted_label (int): The predicted class label.
            resize_target (tuple, optional): If provided, resizes the resulting heatmap to the specified spatial dimensions (Depth, Height, Width).

        Returns:
            np.ndarray: The GradCAM heatmap as a NumPy array.
        """
        classes = [ClassifierOutputTarget(predicted_label)]
        heatmap = self._gradcam(input_tensor=image_data, targets=classes)
        
        if resize_target:
            resize_transform = Resize(spatial_size=resize_target, mode='trilinear')
            heatmap = resize_transform(heatmap)

        return heatmap.numpy()
    
    def _get_gradcam_pp_heatmap(self, image_data, resize_target=None):
        """
        Returns a GradCAM++ heatmap computed during classification.

        Args:
            image_data (Tensor): Input image tensor.
            predicted_label (int): The predicted class label.
            resize_target (tuple, optional): If provided, resizes the resulting heatmap to the specified spatial dimensions (Depth, Height, Width).

        Returns:
            np.ndarray: The Grad-CAM heatmap as a NumPy array.
        """
        heatmap = image_data.squeeze(0)

        if resize_target:
            resize_transform = Resize(spatial_size=resize_target, mode='trilinear')
            heatmap = resize_transform(heatmap)

        return heatmap.numpy()
        

    def _get_occlusion_attributions(self, image_data, predicted_label, resize_target=None):
        """
        Computes occlusion-based attribution maps to visualize which regions of the input image contributed most to the model's prediction.

        Args:
            image_data (Tensor): Input image tensor.
            predicted_label (int): The predicted class label.
            resize_target (tuple, optional): If provided, resizes the resulting attribution map to the specified spatial dimensions (Depth, Height, Width).

        Returns:
            np.ndarray: The occlusion attribution map as a NumPy array.
        """
        image_data = image_data.to(self._device)
        sliding_window_shapes = (1, 16, 16, 16)  # (Channels, Depth, Height, Width)
        strides = (1, 4, 4, 4)  # (Channels, Depth, Height, Width)
        
        attributions_occ = self._occlusion.attribute(
            image_data,
            target=predicted_label,
            strides=strides,
            sliding_window_shapes=sliding_window_shapes,
            baselines=0
        )
        attributions_occ = attributions_occ.squeeze(0).cpu()

        if resize_target:
            resize_transform = Resize(spatial_size=resize_target, mode='trilinear')
            attributions_occ = resize_transform(attributions_occ)

        return attributions_occ.numpy() # remove batch dimension
        

    def _save_image(self, image, spacing, origin, direction, path):
        """
        Creates and saves new ants image with provided spacing, origin, direction to specified path.
        """
        if len(image.shape) > 3:
            image = image.squeeze(0)

        ants_img = ants.from_numpy(
            image,
            origin=origin,
            spacing=spacing,
            direction=direction
        )
        ants.image_write(ants_img, path)


    def process(self):
        data_paths = find_dicom_directories(self._config['raw_mri_root_folder_path'])
        preprocessor = MRIPreprocessor(verbose=True) 
        for folder_path in data_paths:
            try:
                scan_id = Path(folder_path).stem
                output_folder_path = os.sep.join([self._root_output_folder, scan_id])
                os.makedirs(output_folder_path, exist_ok=True)

                nifti_file_path = os.sep.join([output_folder_path, f'{scan_id}.nii.gz'])
                processed_file_path = os.sep.join([output_folder_path, f'{scan_id}_clean.nii.gz'])  
                processed_file_mask_path = None
                if self._save_brain_mask:
                    processed_file_mask_path = os.sep.join([output_folder_path, f'{scan_id}_mask.nii.gz'])

                
                self._clean_data(preprocessor, 
                   folder_path, 
                   nifti_file_path, 
                   processed_file_path, 
                   processed_file_mask_path
                )

                # Data preprocessing
                clean_transform = get_transform_clean_tensor()
                mri_tensor_cleaned = clean_transform(processed_file_path)
                cleaned_mri_shape = tuple(mri_tensor_cleaned.shape)[1:] # cropped image dimensions without empty slices

                resample_tensor_transform = get_transform_resample_tensor()
                image_data = resample_tensor_transform(mri_tensor_cleaned)

                # Classification
                image_data = image_data.unsqueeze(0) # add batch dimension
                predicted_label, gradcam_plus_plus_heatmap = self._classification(scan_id, image_data)


                # Get orientation data from preprocessed image
                spacing = preprocessor._mri_image.spacing
                origin = preprocessor._mri_image.origin
                direction = preprocessor._mri_image.direction

                # Resample and save original image from net
                processed_file_path = os.sep.join([output_folder_path, f'processed.nii.gz'])  
                mri_tensor_cleaned_np = mri_tensor_cleaned.numpy()
                self._save_image(mri_tensor_cleaned_np, spacing, origin, direction, processed_file_path) 

                # GradCam++
                heatmap = self._get_gradcam_pp_heatmap(
                    image_data=gradcam_plus_plus_heatmap.cpu(), 
                    resize_target=cleaned_mri_shape
                )
                heatmap_file_path = os.sep.join([output_folder_path, f'gradcam_plus_plus_heatmap.nii.gz']) 
                self._save_image(heatmap, spacing, origin, direction, heatmap_file_path)

               
                # GradCam
                heatmap = self._get_gradcam_heatmap(
                    image_data=image_data, 
                    predicted_label=predicted_label, 
                    resize_target=cleaned_mri_shape
                )
                heatmap_file_path = os.sep.join([output_folder_path, f'gradcam_heatmap.nii.gz']) 
                self._save_image(heatmap, spacing, origin, direction, heatmap_file_path) 


                # Guided GradCam
                heatmap = self._get_guided_gradcam_heatmap(
                    image_data=image_data, 
                    predicted_label=predicted_label, 
                    resize_target=cleaned_mri_shape
                )
                heatmap_file_path = os.sep.join([output_folder_path, f'guided_gradcam_heatmap.nii.gz']) 
                self._save_image(heatmap, spacing, origin, direction, heatmap_file_path) 


                # Occlusion
                occlusion_att = self._get_occlusion_attributions(
                    image_data=image_data, 
                    predicted_label=predicted_label, 
                    resize_target=cleaned_mri_shape
                )
                occlusion_att_file_path = os.sep.join([output_folder_path, f'occlusion_att.nii.gz'])  
                self._save_image(occlusion_att, spacing, origin, direction, occlusion_att_file_path)
            except Exception as e:
                print(e)
            finally:
                self._save_predicted()

            
def main(run_id:str, config_path:str):
    p = Pipeline(run_id, config_path)
    p.process()
            
    
if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=str, required=True)
    parser.add_argument('--config_path', type=str, required=True)
    args = parser.parse_args()

    main(run_id=args.run_id, config_path=args.config_path)
