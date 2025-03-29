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
        self._model.to(self._device)

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

    
    """
    Save confusion matrix
    """

    

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
