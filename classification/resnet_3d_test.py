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
 


def main(run_id: int = -1, data_config_file_path:str = None, train_config_file_path:str = None): 
    data_config = load_yaml_config(data_config_file_path)
    train_config = load_yaml_config(train_config_file_path)

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
    batch_size = train_config['testing']['batch_size']
    num_workers = train_config['testing']['num_workers']
    num_classes = train_config['model']['num_classes']

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

    print(model)
    """
    Prepare report
    """
    report  = Report(num_classes=num_classes, root_path=report_root_path)
    test_run_table_columns = [
        'ID', 'Loss', 
        'Accuracy', 'Precision', 'Recall', 'F1', 'AUROC'
    ]
    report.create_table('test_results', test_run_table_columns, test_results_path)


    """
    Train model
    """
    tester = Tester(
        model=model, 
        loss_function=loss_function, 
        test_data=test_loader, 
        device=device,
        report=report
    )
    tester.test(run_id, load_model_path)  

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=int, default=-1)
    parser.add_argument('--data_config', type=str, default=None)
    parser.add_argument('--test_config', type=str, default=None)
    args = parser.parse_args()

    main(
        run_id=args.run_id, 
        data_config_file_path=args.data_config,
        train_config_file_path=args.test_config
    )
