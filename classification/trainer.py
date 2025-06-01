import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch
from torch.amp import GradScaler

from report import Report

import monai 
from monai.data import DataLoader
from utils import make_file_dir


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
        report: Report,
        scheduler     
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
        self._scheduler = scheduler

        self._report.init_metrics('train')
        self._report.init_metrics('validation')


    def _run_epoch(self, epoch:int) -> None:
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
            self._scheduler.step(epoch_loss)

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

        model_path = os.sep.join([self._save_model_path, f'{self._run_id}_{epoch_val}.pth'])
        make_file_dir(model_path)

        torch.save(
            checkpoint, 
            os.sep.join([self._save_model_path, f'{self._run_id}_{epoch_val}.pth'])
        )