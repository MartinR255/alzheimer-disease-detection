import os
import argparse
from datetime import datetime

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from utils import get_memory_dataset
from trainer import Trainer

import torch
from report import Report

import monai 
from monai.networks.nets import densenet121, densenet169, densenet201
from monai.data import DataLoader
from monai.utils import set_determinism



def main(run_id:int = -1, batch_size:int = 4, num_workers:int = 0, epoch_num:int = 5, validation_interval:int = 1, model_path:str = None): 
    """
    Setup paths to data
    """
    mri_images_path = os.sep.join(['pre_processed_mri'])
    train_partiton_path = os.sep.join(['mri_classification', 'data', 'train_5.json'])
    validation_partition_path = os.sep.join(['mri_classification', 'data', 'val_5.json'])
    train_transformed_data_path = os.sep.join(['mri_classification', 'data', 'train_proc_5.pt'])
    validation_transformed_data_path = os.sep.join(['mri_classification', 'data', 'val_proc_5.pt'])
    train_results_path = os.sep.join(['mri_classification', 'eval_logs', 'train_results.csv'])
    val_results_path = os.sep.join(['mri_classification', 'eval_logs', 'val_results.csv'])
    train_params_path = os.sep.join(['mri_classification', 'eval_logs', 'train_params.csv'])
    save_model_path = os.sep.join(['mri_classification', 'eval_logs', 'models']) 
    report_root_path = os.sep.join(['mri_classification', 'eval_logs'])
    
    
    """
    Prepare data
    """
    set_determinism(seed=42)

    train_ds = get_memory_dataset(train_transformed_data_path) 
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers, 
        pin_memory=False,
        shuffle=True
    )
    
    val_ds = get_memory_dataset(validation_transformed_data_path)
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers, 
        pin_memory=False
    )
    

    """
    Prepare model, loss function, optimizer etc.
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

    learning_rate = 1e-3 # initial training range 1e-4 to 1e-3, fine-tuning range: 1e-5 to 1e-6
    betas = (0.9, 0.999) # first parameter: 0.9 to 0.95, second parameter: 0.999 to 0.9999
    weight_decay = 1e-5 # range 1e-5 to 1e-2, default 0
    optimizer = torch.optim.Adam(
        params=model.parameters(), 
        lr=learning_rate, 
        betas=betas, 
        weight_decay=weight_decay
    )


    """
    Prepare report
    """
    report  = Report(num_classes=5, root_path=report_root_path)
    training_run_table_columns = [
        'ID', 'Epoch Number', 'Loss', 
        'Accuracy', 'Precision', 'Recall', 'F1', 'AUROC'
    ]
    report.create_table('Training_Results', training_run_table_columns, train_results_path)
    report.create_table('Validation_Results', training_run_table_columns, val_results_path)


    """
    Train model
    """
    trainer = Trainer(
        model=model, 
        optimizer=optimizer, 
        loss_function=loss_function, 
        train_data=train_loader, 
        validation_data=val_loader, 
        validation_interval=validation_interval, 
        device=device,
        save_model_path=save_model_path,
        report=report
    )
    start_train_time = datetime.now()
    trainer.train(run_id, epoch_num, model_path)
    end_train_time = datetime.now()

    
    """
    Save data about traning
    """
    report.create_table('Training_parameters', [
        'ID', 'Epoch Number', 'Training Data Size', 'Validation Data Size', 
        'Batch Size', 'Num Classes', 'Network Type', 'Pretrained', 'Optimizer', 'Learning Rate', 'Betas', 
        'Weight Decay', 'Loss Function', 'Validation Interval', 'Training Duration (seconds)'
    ], train_params_path)

    training_duration = end_train_time - start_train_time 
    report.add_row('Training_parameters', [
        run_id, epoch_num, len(train_ds), len(val_ds), batch_size, num_classes,
        'densenet169', pretrained, 'Adam', learning_rate, betas, weight_decay, 'CrossEntropyLoss', 
        validation_interval, training_duration.total_seconds()
    ])

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--epoch_num', type=int, default=5)
    parser.add_argument('--validation_interval', type=int, default=1)
    parser.add_argument('--pretrained', type=bool, default=False)
    parser.add_argument('--model_path', type=str, default=None)
    args = parser.parse_args()

    main(
        run_id=args.run_id, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        epoch_num=args.epoch_num, 
        validation_interval=args.validation_interval, 
        pretrained=args.pretrained,
        model_path=args.model_path
    )