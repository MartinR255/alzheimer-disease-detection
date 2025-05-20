import os
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
from torchvision.transforms import Normalize, ToTensor
from monai.transforms import ScaleIntensity

from pytorch_grad_cam.utils.image import show_cam_on_image

import ants

import cv2


class Dashboard:

    def __init__(self):
        self._axial_slices = []
        self._coronal_slices = []
        self._saggital_slices = []
        self._sliders = []
        self._update_slicers_functions = []


    def _load_image(self, file_path):
        mri_image = ants.image_read(file_path).numpy()
        mri_image = np.expand_dims(mri_image, axis=0)
        return mri_image
        

    def _create_window(self):
        self._fig, self._axes = plt.subplots(4, 3, figsize=(9, 9))
        plt.subplots_adjust(bottom=0.25)


    def _load_images(self, mri_file_path, grad_cam_file_path, guided_gradcam_file_path, occlusion_attr_file_path):
        mri_image = self._normalize(self._load_image(mri_file_path))
        gradcam_image = self._normalize(self._load_image(grad_cam_file_path))
        guided_gradcam_image = self._normalize(self._load_image(guided_gradcam_file_path))
        
        occlusion_image = self._load_image(occlusion_attr_file_path)
        occlusion_image = np.clip(occlusion_image, a_min=0, a_max=None)
        occlusion_image = self._normalize(occlusion_image)

        # rotate for correct visualization
        mri_image = np.rot90(mri_image, k=1, axes=(3, 2))
        gradcam_image = np.rot90(gradcam_image, k=1, axes=(3, 2))
        guided_gradcam_image = np.rot90(guided_gradcam_image, k=1, axes=(3, 2))
        occlusion_image = np.rot90(occlusion_image, k=1, axes=(3, 2))

        gradcam_heat = self._gradcam_overlay(mri_image, gradcam_image)
        occlusion_heat = self._red_channel_overlay(mri_image, occlusion_image)

        return mri_image, gradcam_heat, guided_gradcam_image, occlusion_heat


    def _red_channel_overlay(self, image, overlay_image, alpha=0.5):
        """
        Applies a red-channel overlay onto a grayscale MRI volume slice-by-slice.

        Parameters
        ----------
        base_volume : np.ndarray
            3D grayscale MRI volume of shape (1, D, H, W). Represents the background image.
        overlay_volume : np.ndarray
            3D volume of same shape as base_volume. Represents the intensity of the red overlay.
        alpha : float, optional
            Blend factor between base MRI and red overlay. Defaults to 0.5.

        Returns
        -------
        overlay_rgb_volume : np.ndarray
            4D volume of shape (D, H, W, 3) representing RGB overlays for each slice.
        """
        image = np.squeeze(image) # shape [D, H, W]
        overlay_image = np.squeeze(overlay_image) # shape [D, H, W]

        overlay_slices = []
        for slice_idx in range(image.shape[0]):
            slice_img = image[slice_idx] # [H,W]
            slice_occ = overlay_image[slice_idx] # [H,W]

            # Create a red-only heatmap: stack zeros for G,B, and slice_occ for R
            heatmap_red = np.zeros((slice_occ.shape[0], slice_occ.shape[1], 3), dtype=np.float32)
            heatmap_red[..., 0] = slice_occ           # Red channel

            # Replicate MRI into RGB as grayscale base
            gray_rgb = np.stack([slice_img]*3, axis=-1)  # [H, W, 3]

            overlay = (1 - alpha) * gray_rgb + heatmap_red * alpha
            overlay_slices.append(overlay)

        overlay_image = np.stack(overlay_slices, axis=0)
        return overlay_image
    

    def _gradcam_overlay(self, image, heatmap):
        image = np.squeeze(image)         # shape: (D, H, W)
        heatmap = np.squeeze(heatmap) # shape: (D, H, W)

        overlay_slices = []
        for d in range(heatmap.shape[0]):
            img_slice = image[d]         # shape: (H, W), float in [0, 1]
            heatmap_slice = heatmap[d] # shape: (H, W), float in [0, 1]

            # Convert grayscale image to RGB
            # rgb_img_slice = cv2.merge([slice_img, slice_img, slice_img])
            rgb_img_slice = np.stack([img_slice]*3, axis=-1)

            overlay = show_cam_on_image(rgb_img_slice, heatmap_slice, use_rgb=True)
            overlay_slices.append(overlay)

        overlay_image = np.stack(overlay_slices, axis=0)  # shape: (D, H, W, 3), dtype: uint8
        return overlay_image


    def _normalize(self, image):
        """
        Normalizes data and scales intensity to range [0, 1]
        """
        to_tensor_transform = ToTensor()
        tensor_image = to_tensor_transform(image.squeeze(0))
        
        mean, std = tensor_image.mean(), tensor_image.std()
        normalize_transform = Normalize(mean=mean, std=std)
        normalized_image = normalize_transform(tensor_image)

        scaler = ScaleIntensity(minv=0, maxv=1.0, dtype=torch.float16)
        scaled_image = scaler(normalized_image)

        scaled_image = scaled_image.unsqueeze(0)
        return scaled_image.numpy()


    def build_dashboard(self, mri_file_path, gradcam_file_path, guided_gradcam_file_path, occlusion_attr_file_path):
        self._create_window()

        images = self._load_images(
            mri_file_path, gradcam_file_path, guided_gradcam_file_path, occlusion_attr_file_path
        )
        mri_image, grad_cam_image, guided_gradcam_image, occlusion_image = images
        
        guided_gradcam_image = guided_gradcam_image.squeeze(0)
        mri_image = mri_image.squeeze(0)
        D, H, W = mri_image.shape  
        init_axial = D // 2
        init_coronal = H // 2
        init_sagittal = W // 2  
        init_slices = [init_axial, init_coronal, init_sagittal]

        self._images = [mri_image, grad_cam_image, guided_gradcam_image, occlusion_image]

        self._init_mri_slices_grayscale(mri_image, 0, init_slices)
        self._init_mri_slices_grayscale(grad_cam_image, 1, init_slices)
        self._init_mri_slices_grayscale(guided_gradcam_image, 2, init_slices)
        self._init_mri_slices_grayscale(occlusion_image, 3, init_slices)
        
        self._init_sliders(mri_image.shape, init_slices)
        for slider, update_func in zip(self._sliders, self._update_slicers_functions):
            slider.on_changed(update_func)
        plt.show()


    def _init_mri_slices_grayscale(self, image, row, init_slices):
        init_axial, init_coronal, init_sagittal = init_slices

        axial_im = self._axes[row][0].imshow(image[init_axial, :, :], cmap='gray')     
        self._axes[row][0].set_title(f'Axial Slice {init_axial}')
        self._axes[row][0].axis('off') 
        
        coronal_im = self._axes[row][1].imshow(image[:, init_coronal, :], cmap='gray')
        self._axes[row][1].set_title(f'Coronal Slice {init_coronal}')
        self._axes[row][1].axis('off') 
        
        sagittal_im = self._axes[row][2].imshow(image[:, :, init_sagittal], cmap='gray')
        self._axes[row][2].set_title(f'Sagittal Slice {init_sagittal}')
        self._axes[row][2].axis('off')

        self._axial_slices.append(axial_im)
        self._coronal_slices.append(coronal_im)
        self._saggital_slices.append(sagittal_im)


    def _init_sliders(self, image_shape, init_slices):
        init_axial, init_coronal, init_sagittal = init_slices
        D, H, W = image_shape

        # Define slider axes
        axcolor = 'lightgoldenrodyellow'
        axial_ax = plt.axes([0.15, 0.1, 0.65, 0.03], facecolor=axcolor)
        coronal_ax = plt.axes([0.15, 0.06, 0.65, 0.03], facecolor=axcolor)
        sagittal_ax = plt.axes([0.15, 0.02, 0.65, 0.03], facecolor=axcolor) 

        # Create sliders
        axial_slider = Slider(axial_ax, 'Axial Slice', 0, D - 1, valinit=init_axial, valstep=1)
        coronal_slider = Slider(coronal_ax, 'Coronal Slice', 0, H - 1, valinit=init_coronal, valstep=1)
        sagittal_slider = Slider(sagittal_ax, 'Sagittal Slice', 0, W - 1, valinit=init_sagittal, valstep=1) 

        self._sliders.extend([axial_slider, coronal_slider, sagittal_slider])
        self._update_slicers_functions.extend([self._update_axial, self._update_coronal, self._update_sagittal])


    def _update_axial(self, val):
        axial_slider = self._sliders[0]
        idx = int(axial_slider.val)
    
        for ax_idx, slice_cor in enumerate(self._axial_slices):
            slice_cor.set_data(self._images[ax_idx][idx, :, :])
            self._axes[ax_idx][0].set_title(f'Coronal Slice {idx}')
        self._fig.canvas.draw_idle()  


    def _update_coronal(self, val):
        coronal_slider = self._sliders[1]
        idx = int(coronal_slider.val)

        for ax_idx, slice_cor in enumerate(self._coronal_slices):
            slice_cor.set_data(self._images[ax_idx][:, idx, :])
            self._axes[ax_idx][1].set_title(f'Coronal Slice {idx}')
        self._fig.canvas.draw_idle()  


    def _update_sagittal(self, val):
        sagittal_slider = self._sliders[2]
        idx = int(sagittal_slider.val)

        for ax_idx, slice_sgg in enumerate(self._saggital_slices):
            slice_sgg.set_data(self._images[ax_idx][:, :, idx])
            self._axes[ax_idx][2].set_title(f'Sagittal Slice {idx}')
        self._fig.canvas.draw_idle()  


        

def main():
    mri_file_path = ''
    gradcam_file_path = '' 
    guided_gradcam_file_path = ''
    occlusion_attr_file_path= ''

    d = Dashboard()
    d.build_dashboard(
        mri_file_path=mri_file_path, 
        gradcam_file_path=gradcam_file_path, 
        guided_gradcam_file_path=guided_gradcam_file_path,
        occlusion_attr_file_path=occlusion_attr_file_path
    )


if __name__ == '__main__':
    main()