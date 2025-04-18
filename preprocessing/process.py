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
    

    def dicom_to_nifti(self, dicom_folder_path:str, output_file_path:str):
        """
        Convert the DICOM series to a NIfTI file with the specified name
        """
        dicom2nifti.dicom_series_to_nifti(dicom_folder_path, output_file_path, reorient_nifti=True)

        return self.load_mri(output_file_path)
        

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


    def zscore_normalization(self):
        """Perform z-score normalization on the brain voxels within the mask.

        The normalization is applied only to voxels within the brain mask,
        transforming them to have zero mean and unit standard deviation.

        Returns:
            MRIPreprocessor: The preprocessor instance for method chaining.
        """
        if self._mri_mask is None:
            self.logger.debug("Mask is None, skipping z-score normalization")
            return self
        
        self.logger.debug("Performing z-score normalization")
        mri_image = self._mri_image.numpy()
        mri_mask = self._mri_mask.numpy()
        brain_voxels = mri_image[mri_mask > 0]
        mean_intensity = np.mean(brain_voxels)
        std_intensity = np.std(brain_voxels)

        normalized_data = np.zeros_like(mri_image, dtype=np.float32)
       
        # Apply normalization within the brain mask
        normalized_data[mri_mask > 0] = (brain_voxels - mean_intensity) / std_intensity
        
        self._mri_image.new_image_like(normalized_data)

        return self
    

    def reorient_image(self):
        self._mri_image = ants.reorient_image2(self._mri_image, orientation='RAS')

        if self._mri_mask:
            self._mri_mask = ants.reorient_image2(self._mri_mask, orientation='RAS')
        return self
    

    def skull_strip(self):
        """Remove non-brain tissue from the MRI image using ANTs brain extraction.

        Creates and applies a brain mask to remove skull and other non-brain tissues.
        Updates both the MRI image and the generated mask.

        Returns:
            MRIPreprocessor: The preprocessor instance for method chaining.
        """
        output_mask = aspyputil.brain_extraction(self._mri_image, 't1') 

        self._mri_image = self._mri_image * output_mask
        self._mri_mask = output_mask

        return self


    def save_mri(self, output_path: str):
        """Save the preprocessed MRI image to a file.

        Args:
            output_path (str): Path where the processed image will be saved.

        Returns:
            MRIPreprocessor: The preprocessor instance for method chaining.
        """
        self.logger.debug(f'Saving preprocessed data to {output_path}')
        ants.image_write(self._mri_image, output_path)

        return self
    
