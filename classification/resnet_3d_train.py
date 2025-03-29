import os
import argparse
from datetime import datetime

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from utils import get_memory_dataset

import torch
from torch.amp import GradScaler

from report import Report

import monai 
from monai.networks.nets import resnet18, resnet34, resnet50, resnet101
from monai.data import DataLoader
from monai.utils import set_determinism



class Trainer():

    def __init__(self, 
                model:monai.networks.nets, 
                optimizer:torch.optim.Optimizer, 
                loss_function:torch.nn.Module, 
                train_data:DataLoader, 
                validation_data:DataLoader, 
                validation_interval:int,
                device:torch.device,
                save_model_path:str,
                report: Report     
    ) -> None: 
        self._model = model
        self._optimizer = optimizer
        self._loss_function = loss_function
        self._train_data = train_data
        self._validation_data = validation_data
        self._validation_interval = validation_interval
        self._device = device
        self._save_model_path = save_model_path
        self._report = report

        self._report.init_metrics('train')
        self._report.init_metrics('validation')


    def _run_epoch(self, epoch:int) -> None:
        print('Training')
        self._model.train()

        torch.cuda.empty_cache()
        epoch_loss = 0
        epoch_step = 0
        self._report.reset_metrics('train')
        for batch_data in self._train_data: 
            epoch_step += 1
            
            inputs, labels = batch_data[0].to(self._device), batch_data[1].to(self._device) 
            self._optimizer.zero_grad() 
            with torch.autocast(device_type=self._device.type, dtype=torch.float16):
                model_out = self._model(inputs)
                loss = self._loss_function(model_out, labels)
            
            self._scaler.scale(loss).backward() 
            self._scaler.step(self._optimizer)
            self._scaler.update()

            model_out_argmax = model_out.argmax(dim=1)
            self._report.update_metrics('train', 'accuracy', model_out_argmax, labels)
            self._report.update_metrics('train', 'precision', model_out_argmax, labels)
            self._report.update_metrics('train', 'recall', model_out_argmax, labels)
            self._report.update_metrics('train', 'f1_score', model_out_argmax, labels)
            self._report.update_metrics('train', 'auroc', model_out.softmax(dim=1), labels)  

            epoch_loss += loss.item()

        epoch_loss /= epoch_step

        metrics_values = self._report.compute_metrics('train')
        self._report.add_row('Training_Results', [
            self._run_id,
            epoch, 
            epoch_loss,
            metrics_values['accuracy'],
            metrics_values['precision'],
            metrics_values['recall'],
            metrics_values['f1_score'],
            metrics_values['auroc']
        ])



    def _validate(self, epoch_val:int) -> None:
            print("Validation")
            self._model.eval()

            self._report.reset_metrics('validation')
            epoch_loss = 0
            step = 0
            with torch.no_grad():
                for batch_data in self._validation_data:
                    step += 1
                    inputs, labels = batch_data[0].to(self._device), batch_data[1].to(self._device)
                    model_out = self._model(inputs)
                    model_out_argmax = model_out.argmax(dim=1)
                    self._report.update_metrics('validation', 'accuracy', model_out_argmax, labels)
                    self._report.update_metrics('validation', 'precision', model_out_argmax, labels)
                    self._report.update_metrics('validation', 'recall', model_out_argmax, labels)
                    self._report.update_metrics('validation', 'f1_score', model_out_argmax, labels)
                    self._report.update_metrics('validation', 'auroc', model_out.softmax(dim=1), labels) 

                    loss = self._loss_function(model_out, labels)
                    epoch_loss += loss.item()

            epoch_loss /= step

            metrics_values = self._report.compute_metrics('validation')
            self._report.add_row('Validation_Results', [
                self._run_id,
                epoch_val, 
                epoch_loss,
                metrics_values['accuracy'],
                metrics_values['precision'],
                metrics_values['recall'],
                metrics_values['f1_score'],
                metrics_values['auroc']
            ])
            
            # Save the best model based on f1 score
            metric = metrics_values['f1_score']
            if metric > self._best_metric:
                self._best_metric = metric
            print(f'Best Metric: {self._best_metric}')


    def _load_model(self, model_path:str) -> None:
        checkpoint = torch.load(model_path)
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                self._model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            self._last_epoch = checkpoint.get('epoch', 0) + 1
            self._best_metric = checkpoint.get('best_metric', -1)
        else:
            self._model.load_state_dict(checkpoint)


    def train(self, run_id:int, epoch_num:int, model_path:str = None) -> None:
        self._best_metric = -1 # F1 score
        self._last_epoch = 1

        if model_path:
            self._load_model(model_path)

        torch.cuda.empty_cache()
        self._scaler = GradScaler()
        self._run_id = run_id

        for epoch in range(self._last_epoch, self._last_epoch + epoch_num):
            self._run_epoch(epoch)
            if (epoch) % self._validation_interval == 0:
                self._validate(epoch)
                self._save_model(epoch, self._best_metric)

       

    def _save_model(self, epoch_val:int, best_metric:float) -> None:
        checkpoint = {
            'model_state_dict': self._model.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict(),
            'epoch': epoch_val,
            'best_metric': best_metric
        }
        torch.save(
            checkpoint, 
            os.sep.join([self._save_model_path, f'{self._run_id}_{epoch_val}.pth'])
        )



def main(run_id: int = -1, batch_size: int = 4, num_workers: int = 0, epoch_num: int = 5, validation_interval: int = 1, model_path: str = None): 
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
    model = resnet50(
        spatial_dims=3,
        n_input_channels=1,
        num_classes=num_classes,
        pretrained=pretrained,
    )
    
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
    report  = Report(num_classes=5)
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
        'resnet50', pretrained, 'Adam', learning_rate, betas, weight_decay, 'CrossEntropyLoss', 
        validation_interval, training_duration.total_seconds()
    ])

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--epoch_num', type=int, default=5)
    parser.add_argument('--validation_interval', type=int, default=1)
    parser.add_argument('--model_path', type=str, default=None)
    args = parser.parse_args()

    main(
        run_id=args.run_id, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        epoch_num=args.epoch_num, 
        validation_interval=args.validation_interval, 
        model_path=args.model_path
    )
