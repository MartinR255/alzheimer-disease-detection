
import os
import numpy as np
import json

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch
from monai.networks.nets import resnet18, resnet34
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from unlabeled_dataset import UnlabeledDataset
from monai.data import DataLoader


from utils import load_model, get_transform







def main():
    model_path = os.sep.join([''])
    mri_images_path = os.sep.join([''])
    image_paths = [mri_images_path]

    batch_size = 1
    transofrm = get_transform()
    dataset = UnlabeledDataset(image_paths=image_paths, transform=transofrm)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)



    spatial_dims = 3
    n_input_channels = 1
    num_classes = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model =  resnet34(
        spatial_dims=spatial_dims, 
        n_input_channels=n_input_channels, 
        num_classes=num_classes
    ).to(device)
    model = load_model(model, model_path)
    layers = [model.layer4[2].conv2]
    gradcam = GradCAM(model=model, target_layers=layers)

    
    model.eval()
    
    predicted_labels = torch.tensor([], dtype=torch.int64).to(device) 
    with torch.no_grad():
        for batch_data in data_loader:
            inputs, _ = batch_data[0].to(device), batch_data[1]
            model_out = model(inputs)
            model_out_argmax = model_out.argmax(dim=1)
            predicted_labels = torch.cat((predicted_labels, model_out_argmax))
            

    # get gradcam gradients
    image_index = 0
    for batch_data in data_loader:
        inputs, _ = batch_data[0].to(device), batch_data[1]
    
            
        labels = predicted_labels[image_index:image_index+batch_size]
        for image, predicted_label in zip(inputs, labels):
            classes = [ClassifierOutputTarget(predicted_label.item())]
            heatmap = gradcam(input_tensor=image.unsqueeze(0), targets=classes)

        image_index += batch_size



if __name__ == '__main__':
    main()
      
