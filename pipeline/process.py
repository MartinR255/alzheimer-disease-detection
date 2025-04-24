import os
import logging
import numpy as np

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import ants
import antspynet.utilities as aspyputil
import dicom2nifti


class MRIPreprocessor:
    
    def __init__(self, verbose=False):
        self._initialize_logger(verbose)
        self._mri_image = None
        self._mri_mask = None


    def _initialize_logger(self, verbose=False):
        """Initialize the logger for the preprocessor.

        Args:
            verbose (bool, optional): If True, sets logging level to DEBUG, otherwise INFO. Defaults to False.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG if verbose else logging.INFO)


    def load_mri(self, file_path: str):
        """Load an MRI image from a file.

        Args:
            file_path (str): Path to the MRI image file.

        Returns:
            MRIPreprocessor: The preprocessor instance for method chaining.
        """
        self.logger.debug(f'Loading MRI data from {file_path}')
        self._mri_image = ants.image_read(file_path)

        return self


    def load_mri_mask(self, file_path: str):
        """Load an MRI mask from a file.

        Args:
            file_path (str): Path to the MRI mask file.

        Returns:
            MRIPreprocessor: The preprocessor instance for method chaining.
        """
        self.logger.debug(f'Loading MRI mask data from {file_path}')
        self._mri_mask = ants.image_read(file_path)
       
        return self
    

    def dicom_to_nifti(self, dicom_folder_path:str, output_file_path:str, orientation:str = None):
        """
        Convert the DICOM series to a NIfTI file with the specified name
        """
        self.logger.debug(f'Converting DICOM series to NIfTI: {output_file_path}')

        dicom2nifti.dicom_series_to_nifti(dicom_folder_path, output_file=output_file_path, reorient_nifti=False)
       
        return self
        

    def resample_volume(self, target_shape:tuple = (1, 1, 1)):
        """Resample the MRI volume to a specified shape.

        Args:
            target_shape (tuple): Target shape for resampling.

        Returns:
            MRIPreprocessor: The preprocessor instance for method chaining.
        """
        self.logger.debug(f"Resampling to shape: {target_shape}")   
        self._mri_image = ants.resample_image(self._mri_image, target_shape, use_voxels=False, interp_type=3)
        return self


    def register_to_mni(self, mni_template_path:str, *tranformation_types:str):
        """Register the MRI image to MNI space using ANTs registration.

        Args:
            mni_template_path (str): Path to the MNI template file.
            tranformation_types (str): Types of transformations to apply (e.g., 'Affine', 'SyN').

        Returns:
            MRIPreprocessor: The preprocessor instance for method chaining.
        """
        self.logger.debug(f"Starting MNI registration with template: {mni_template_path}")
        
        mri_image = ants.n4_bias_field_correction(self._mri_image)

        mni_template = ants.image_read(mni_template_path)
        for tranformation_type in tranformation_types:
            registration = ants.registration(
                fixed=mni_template,
                moving=mri_image,
                type_of_transform=tranformation_type
            )
            mri_image = registration['warpedmovout']
       
        self._mri_image = mri_image
        return self


    def reorient_image(self, orientation:str='LAS'):
        self.logger.debug(f'Reorienting MRI image to {orientation} orientation')
       
        self._mri_image = ants.reorient_image2(image=self._mri_image, orientation=orientation)

        if self._mri_mask:
            self._mri_mask = ants.reorient_image2(self._mri_mask, orientation=orientation)
        return self
    

    def skull_strip(self):
        """Remove non-brain tissue from the MRI image using ANTs brain extraction.

        Creates and applies a brain mask to remove skull and other non-brain tissues.
        Updates both the MRI image and the generated mask.

        Returns:
            MRIPreprocessor: The preprocessor instance for method chaining.
        """
        self._mri_mask = aspyputil.brain_extraction(self._mri_image, 't1') 
        self._mri_image = self._mri_image * self._mri_mask

        return self
    

    def add_adaptive_gaussian_noise(self, mean:float = 0.0, standardDeviation:float = 1.0):
        self.logger.debug(f'Adding adaptive gaussian noise to MRI image')
        self._mri_image = ants.add_noise_to_image(
            self._mri_image, 
            'additivegaussian', 
            (mean, standardDeviation)
        )
        return self
    

    def add_salt_and_pepper_noise(self, probability:float = 0.1, saltValue:float = 0.0, pepperValue:float = 100.0):
        self.logger.debug(f'Adding salt and pepper noise to MRI image')
        self._mri_image = ants.add_noise_to_image(
            self._mri_image, 
            'saltandpepper', 
            (probability, saltValue, pepperValue)
        )
        return self
    

    def save_mri(self, mri_output_path: str, mask_output_path:str=None):
        """Save the preprocessed MRI image to a file.

        Args:
            output_path (str): Path where the processed image will be saved.

        Returns:
            MRIPreprocessor: The preprocessor instance for method chaining.
        """
        self.logger.debug(f'Saving preprocessed image to {mri_output_path}')
        ants.image_write(self._mri_image, mri_output_path)

        if mask_output_path:
            self.logger.debug(f'Saving mask of preprocessed image to {mask_output_path}')
            ants.image_write(self._mri_mask, mask_output_path)

        return self
    
    
    def get_image(self):
        """
        Returns the image data from ANTs image(ants_image.numpy()).
        """
        return self._mri_image.numpy()

    
    def image_from(self, data:np.ndarray) -> None:
        """
        Creates new ANTs image with the same header but with different image data

        Args:
            data (np.ndarray): Path where the processed image will be saved.

        Returns:
            MRIPreprocessor: The preprocessor instance for method chaining.
        """
        self._mri = ants.new_image_like(self._mri_image, data)
        return self
    
