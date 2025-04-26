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
                report: Report     
    ) -> None: 
        self._model = model
        self._loss_function = loss_function
        self._test_data = test_data
        self._device = device
        self._report = report

        self._report.init_metrics('test')


    def test(self, run_id:int, epoch:int, model_path:str) -> None:
        self._load_model(model_path)
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
            epoch,
            epoch_loss,
            metrics_values['accuracy'],
            metrics_values['precision'], 
            metrics_values['recall'], 
            metrics_values['f1_score'], 
            metrics_values['auroc']
        ])
        self._report.save_confusion_matrix(predicted_labels, ground_truth_labels, f'conf_mat_{run_id}.csv')


    def _load_model(self, model_path:str) -> None:
        checkpoint = torch.load(model_path)
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                self._model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self._model.load_state_dict(checkpoint)