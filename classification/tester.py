import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch

from report import Report

import monai 
from monai.data import DataLoader



class Tester():

    def __init__(self, 
                model:monai.networks.nets, 
                loss_function:torch.nn.Module, 
                test_data:DataLoader, 
                device:torch.device,
                save_model_path:str,
                report: Report     
    ) -> None: 
        self._model = model
        self._loss_function = loss_function
        self._test_data = test_data
        self._device = device
        self._save_model_path = save_model_path
        self._report = report

        self._report.init_metrics('test')


    def _test(self, run_id:int) -> None:
        self._load_model(self._save_model_path)
        self._model

        self._model.eval()
        self._report.reset_metrics('test')
        epoch_loss = 0
        step = 0
        ground_truth_labels = torch.tensor([], dtype=torch.int64).to(self._device)
        predicted_labels = torch.tensor([], dtype=torch.int64).to(self._device) 
        with torch.no_grad():
            for batch_data in self._test_data:
                step += 1
                inputs, labels = batch_data[0].to(self._device), batch_data[1].to(self._device)
                model_out = self._model(inputs)
                model_out_argmax = model_out.argmax(dim=1)
                self._report.update_metrics('test', 'accuracy', model_out_argmax, labels)
                self._report.update_metrics('test', 'precision', model_out_argmax, labels)
                self._report.update_metrics('test', 'recall', model_out_argmax, labels)
                self._report.update_metrics('test', 'f1_score', model_out_argmax, labels)
                self._report.update_metrics('test', 'auroc', model_out.softmax(dim=1), labels) 

                # collect data for confusion matrix
                ground_truth_labels = torch.cat((ground_truth_labels, labels))
                predicted_labels = torch.cat((predicted_labels, model_out_argmax))


                loss = self._loss_function(model_out, labels)
                epoch_loss += loss.item()
        epoch_loss /= step

        metrics_values = self._report.compute_metrics('test')
        self._report.add_row('test_results', [
            run_id,
            epoch_loss,
            metrics_values['test'],
            metrics_values['test'],
            metrics_values['test'],
            metrics_values['test'],
            metrics_values['test']
        ])
        self._report.save_confusion_matrix(predicted_labels, ground_truth_labels, f'conf_mat_{run_id}.pt')


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
        else:
            self._model.load_state_dict(checkpoint)