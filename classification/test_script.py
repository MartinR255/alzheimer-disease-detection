import os
import argparse
from pathlib import Path

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from utils import get_memory_dataset, get_loss, get_network
from tester import Tester

import torch

from report import Report

from monai.data import DataLoader
from monai.utils import set_determinism
 

def main(run_id:int, data_config:dict, train_config:dict): 
    """
    Setup paths to data
    """
    test_transformed_data_path = data_config['test_preproc_chunk_path']   
    test_results_path = os.sep.join([data_config['save_eval_logs_path'], 'test_results.csv'])
    test_results_separate_path = os.sep.join([data_config['save_eval_logs_path'], 'test_results_separate.csv'])
    report_root_path = data_config['save_eval_logs_path']

    """
    Load configs 
    """
    load_model_path = train_config['model']['load_model_path']
    batch_size = train_config['training']['batch_size']
    num_workers = train_config['training']['num_workers']
    num_classes = train_config['model']['num_classes']
    

    model_filename = Path(train_config['model']['load_model_path']).name
    _, model_epoch = model_filename.split('.')[0].split('_') # runID_epochNum.pth
    epoch = model_epoch
    

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
    model = get_network(train_config['model']).to(device)
    loss_function = get_loss(train_config['loss'], device)

    """
    Prepare report
    """
    report  = Report(num_classes=num_classes, root_path=report_root_path)
    test_run_table_columns = [
        'ID', 'Epoch', 'Loss',
        'Accuracy', 'Precision', 'Recall', 'F1', 'AUROC'
    ]
    report.create_table('test_results', test_run_table_columns, test_results_path)

    test_run_table_columns_separate = [
        'ID', 'Epoch', 'Group',
        'Accuracy', 'Precision', 'Recall', 'F1', 'AUROC'
    ]
    report.create_table('test_results_separate', test_run_table_columns_separate, test_results_separate_path)


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
