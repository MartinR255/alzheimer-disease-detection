import os
import argparse

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from utils import get_memory_dataset, load_yaml_config, get_loss
from utils import get_resnet_model
from tester import Tester

import torch

from report import Report

from monai.data import DataLoader
from monai.utils import set_determinism
 


def main(run_id:int, data_config:dict, train_config:dict): 
    """
    Setup paths to data
    """
    mri_images_path = data_config['images_path']
    test_partiton_path = data_config['test_partiton_path'] 
    test_transformed_data_path = data_config['test_preproc_chunk_path']   
    test_results_path = os.sep.join([data_config['save_eval_logs_path'], 'test_results.csv'],)
    report_root_path = data_config['save_eval_logs_path']

    """
    Load configs 
    """
    load_model_path = train_config['model']['load_model_path']
    batch_size = train_config['batch_size']
    num_workers = train_config['num_workers']
    num_classes = train_config['model']['num_classes']
    epoch = train_config['epoch']

    """
    Prepare data
    """
    set_determinism(seed=42)
    test_ds = get_memory_dataset(test_transformed_data_path) 
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers, 
        pin_memory=False,
        shuffle=True
    )
    
    """
    Prepare model and loss function
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_resnet_model(train_config['model']).to(device)
    loss_function = get_loss(train_config['loss'])

    """
    Prepare report
    """
    report  = Report(num_classes=num_classes, root_path=report_root_path)
    test_run_table_columns = [
        'ID', 'Epoch', 'Loss',
        'Accuracy', 'Precision', 'Recall', 'F1', 'AUROC'
    ]
    report.create_table('test_results', test_run_table_columns, test_results_path)


    """
    Test model
    """
    tester = Tester(
        model=model, 
        loss_function=loss_function, 
        test_data=test_loader, 
        device=device,
        report=report
    )
    tester.test(run_id, epoch, load_model_path)  
