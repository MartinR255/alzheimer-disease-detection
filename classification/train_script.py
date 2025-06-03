import os
import argparse
from datetime import datetime

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from utils import (
    get_memory_dataset, 
    load_yaml_config, 
    get_optimizer, 
    get_loss, 
    copy_config,
    get_network,
    get_scheduler            
)


from trainer import Trainer

import torch

from report import Report

from monai.data import DataLoader
from monai.utils import set_determinism


def main(run_id:int = -1, data_config_file_path:str = None, train_config_file_path:str = None): 
    data_config = load_yaml_config(data_config_file_path)
    train_config = load_yaml_config(train_config_file_path)

    copy_config(data_config_file_path, os.sep.join([data_config['save_eval_logs_path'], 'data_configs', f'{run_id}_data_config.yaml']))
    copy_config(train_config_file_path,  os.sep.join([data_config['save_eval_logs_path'], 'train_configs', f'{run_id}_train_config.yaml']))

    """
    Setup paths to data
    """
    train_transformed_data_path = data_config['train_preproc_chunk_path']  
    validation_transformed_data_path = data_config['validation_preproc_chunk_path']  
    
    train_results_path = os.sep.join([data_config['save_eval_logs_path'], 'train_results.csv'],)
    val_results_path = os.sep.join([data_config['save_eval_logs_path'], 'val_results.csv'])
    train_params_path = os.sep.join([data_config['save_eval_logs_path'], 'train_params.csv'])

    save_model_path = os.sep.join([data_config['save_eval_logs_path'], 'models'])
    report_root_path = data_config['save_eval_logs_path']

    """
    Load configs 
    """
    load_model_path = train_config['model']['load_model_path']
    batch_size = train_config['training']['batch_size']
    num_epochs = train_config['training']['num_epochs']
    num_workers = train_config['training']['num_workers']
    validation_interval = train_config['training']['validation_interval']

    num_classes = train_config['model']['num_classes']


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_network(train_config['model']).to(device)
    optimizer = get_optimizer(model.parameters(), train_config['optimizer'])
    loss_function = get_loss(train_config['loss'], device)
    scheduler = get_scheduler(train_config['scheduler'], optimizer) if train_config['scheduler']['name'] is not None else None
    
   
   
    """
    Prepare report
    """
    report  = Report(num_classes=num_classes, root_path=report_root_path)
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
        report=report,
        scheduler=scheduler
    )
    start_train_time = datetime.now()
    trainer.train(run_id, num_epochs, load_model_path)
    end_train_time = datetime.now()

    
    """
    Save data about traning
    """
    report.create_table('Training_parameters', [
        'ID', 'Epoch Number', 'Training Data Size', 'Validation Data Size', 'Training Duration (seconds)'
    ], train_params_path)

    training_duration = end_train_time - start_train_time 
    report.add_row('Training_parameters', [
        run_id, num_epochs, len(train_ds), len(val_ds), training_duration.total_seconds()
    ])

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=int, default=-1)
    parser.add_argument('--data_config', type=str, default=None)
    parser.add_argument('--train_config', type=str, default=None)
    args = parser.parse_args()

    main(
        run_id=args.run_id, 
        data_config_file_path=args.data_config,
        train_config_file_path=args.train_config
    )
