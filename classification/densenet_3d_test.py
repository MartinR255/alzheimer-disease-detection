import os
import argparse

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from utils import get_memory_dataset
from tester import Tester

import torch

from report import Report

import monai 
from monai.networks.nets import densenet121, densenet169, densenet201
from monai.data import DataLoader
from monai.utils import set_determinism

     
def main(run_id: int = -1, batch_size: int = 4, num_workers: int = 0, model_path: str = None): 
    """
    Setup paths to data
    """
    mri_images_path = os.sep.join(['pre_processed_mri'])
    test_partition_path = os.sep.join(['mri_classification', 'data', 'test_5.json'])
    test_transformed_data_path = os.sep.join(['mri_classification', 'data', 'test_proc_5.pt'])
    test_results_path = os.sep.join(['mri_classification', 'eval_logs', 'test_results.csv'])
    report_root_path = os.sep.join(['mri_classification', 'eval_logs'])

    
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
    num_classes = 5
    pretrained = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = densenet169(
        spatial_dims=3, 
        in_channels=1, 
        out_channels=num_classes
    ).to(device)
    loss_function = torch.nn.CrossEntropyLoss()



    """
    Prepare report
    """
    report  = Report(num_classes=5, root_path=report_root_path)
    test_run_table_columns = [
        'ID', 'Epoch Number', 'Loss', 
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
        model_path=model_path,
        report=report
    )
    tester.test(run_id) 

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--model_path', type=str, default=None)
    args = parser.parse_args()

    main(
        run_id=args.run_id, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        model_path=args.model_path
    )
