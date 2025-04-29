import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

class Dashboard:

    def __init__(self):
        self._axial_slices = []
        self._coronal_slices = []
        self._saggital_slices = []
        self._sliders = []
        self._update_slicers_functions = []


    def _load_iamge(self, file_path):
        sitk_image = sitk.ReadImage(file_path)
        mri_image = sitk.GetArrayFromImage(sitk_image) 

        if mri_image.shape[0] == 1:
            mri_image = mri_image.squeeze(0)

        return mri_image
        

    def _create_window(self):
        self._fig, self._axes = plt.subplots(3, 3, figsize=(9, 9))
        plt.subplots_adjust(bottom=0.25)


    def _load_images(self, mri_file_path, grad_cam_file_path, occlusion_attr_file_path):
        mri_image = self._load_iamge(mri_file_path)
        grad_cam_image = self._get_gradcam_image(mri_image, grad_cam_file_path)
        occlusion_image = self._get_occlusion_image(mri_image, occlusion_attr_file_path)
        self._images = [mri_image, grad_cam_image, occlusion_image]


    def _get_gradcam_image(self, mri_image, heatmap_path):
        return mri_image


    def _get_occlusion_image(self, mri_image, occlusion_attr_path):
        return mri_image


    def build_dashboard(self, mri_file_path, grad_cam_file_path, occlusion_attr_file_path):
        self._create_window()
        self._load_images(mri_file_path, grad_cam_file_path, occlusion_attr_file_path)
        
        mri_image = self._images[0]
        D, H, W = mri_image.shape  

        init_axial = D // 2
        init_coronal = H // 2
        init_sagittal = W // 2  
        init_slices = [init_axial, init_coronal, init_sagittal]

        # Init mri image slices
        for row, image in enumerate(self._images):
            self._init_mri_slices(image, row, init_slices)

        # Bind sliders to update function on change
        self._init_sliders(mri_image.shape, init_slices)
        for slider, update_func in zip(self._sliders, self._update_slicers_functions):
            slider.on_changed(update_func)
        plt.show()


    def _init_mri_slices(self, image, row, init_slices):
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
        for ax_idx, slice_ax in enumerate(self._axial_slices):
            slice_ax.set_data(self._images[0][idx, :, :])
            self._axes[ax_idx][0].set_title(f'Axial Slice {idx}')
        self._fig.canvas.draw_idle()  


    def _update_coronal(self, val):
        coronal_slider = self._sliders[1]
        idx = int(coronal_slider.val)
        for ax_idx, slice_cor in enumerate(self._coronal_slices):
            slice_cor.set_data(self._images[1][:, idx, :])
            self._axes[ax_idx][1].set_title(f'Coronal Slice {idx}')
        self._fig.canvas.draw_idle()  


    def _update_sagittal(self, val):
        sagittal_slider = self._sliders[2]
        idx = int(sagittal_slider.val)
        for ax_idx, slice_sgg in enumerate(self._saggital_slices):
            slice_sgg.set_data(self._images[2][:, :, idx])
            self._axes[ax_idx][2].set_title(f'Sagittal Slice {idx}')
        self._fig.canvas.draw_idle()  


        

def main():
    mri_file_path = ''
    grad_cam_file_path = ''
    occlusion_attr_file_path=  ''

    d = Dashboard()
    d.build_dashboard(
        mri_file_path=mri_file_path, 
        grad_cam_file_path=grad_cam_file_path, 
        occlusion_attr_file_path=occlusion_attr_file_path
    )


if __name__ == '__main__':
    main()