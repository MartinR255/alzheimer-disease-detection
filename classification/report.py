import os
import pandas as pd

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch
from torcheval.metrics import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
    MulticlassAUROC
)

from torcheval.metrics.functional import multiclass_confusion_matrix

class Report:

    def __init__(self, num_classes, root_path):
        self._num_classes = num_classes
        self._root_path = root_path
        self._tables = {}
        self._metrics = {}


    def _create_metrics(self):
        """
        Create evaluation metrics for multiclass classification.
        
        Args:
            None
            
        Returns:
            dict: Dictionary containing initialized metrics:
                - 'accuracy': MulticlassAccuracy
                - 'precision': MulticlassPrecision
                - 'recall': MulticlassRecall
                - 'f1_score': MulticlassF1Score
                - 'auroc': MulticlassAUROC
        """
        accuracy = MulticlassAccuracy(num_classes=self._num_classes, average='macro')
        precision = MulticlassPrecision(num_classes=self._num_classes, average='macro')
        recall = MulticlassRecall(num_classes=self._num_classes, average='macro')
        f1_score = MulticlassF1Score(num_classes=self._num_classes, average='macro')
        auroc = MulticlassAUROC(num_classes=self._num_classes, average='macro')

        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1_score, 'auroc': auroc}


    def init_metrics(self, group_name: str):
        """
        Initialize metrics objects.
        MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassAUROC
        
        Args:
            group_names (str): Metrics group name.
            
        Returns:
            None
        """
        self._metrics[group_name] = self._create_metrics()


    def update_metrics(self, group_name: str, metric_name: str, predicted_value: torch.Tensor, ground_truth: torch.Tensor) -> None:
        """
        Update a specific metric.

        Args:
            group_name (str): The metric group name (e.g., 'train', 'validation').
            metric_name (str): The specific metric to update ('accuracy', 'precision', 'recall', 'f1_score', 'auroc').
            predicted_value (torch.Tensor): The predicted values from the model.
            ground_truth (torch.Tensor): The actual ground truth values.

        Returns:
            None
        """
        self._metrics[group_name][metric_name].update(predicted_value, ground_truth)


    def compute_metrics(self, group_name: str) -> dict:
        """
        Compute all metrics for a given metric group and return the results.

        Args:
            group_name (str): The metric group name (e.g., 'train', 'val').

        Returns:
            dict: A dictionary containing the computed values for each metric 
            ('accuracy', 'precision', 'recall', 'f1_score', 'auroc').
            
        """
        computations = {}
        for metric_name, metric in self._metrics[group_name].items():
            computations[metric_name] = metric.compute().item()


    def reset_metrics(self, group_name: str):
        """
        Reset all metrics with specific group name to their initial state.
        
        Args:
            group_name (str): Metrics group name.
            
        Returns:
            None
        """
        for metric in self._metrics[group_name].values():
            metric.reset()
    

    def create_table(self, table_name, header, file_path=None):
        """
        Create a new pandas DataFrame table with the specified name and header columns.
        
        Args:
            table_name (str): The name of the table to create
            header (list): List of column names for the table
            file_path (str): The path where the CSV file will be saved
        """        
        if not(os.path.exists(file_path)):
            pd.DataFrame(columns=header).to_csv(file_path, index=False)

        self._tables[table_name] = {
            'table': pd.DataFrame(columns=header),
            'file_path': file_path
        }

        
    def add_row(self, table_name, row_data):
        """
        Add a new row of data to the specified table.
        
        Args:
            table_name (str): The name of the table to add the row to
            row_data (list): List of data values to add to the row
        """
        if table_name not in self._tables:
            raise ValueError(f"Table '{table_name}' does not exist.")
        
        new_line = pd.DataFrame([row_data], columns=self._tables[table_name]['table'].columns)

        new_line.to_csv(self._tables[table_name]['file_path'], mode='a', header=False, index=False)


    def save_confusion_matrix(self, predicted_labels, ground_truth_labels, file_name:str):
        file_path = os.sep.join([self._root_path, 'confusion_matrices', file_name]) 
        conf_matrix = multiclass_confusion_matrix(predicted_labels, ground_truth_labels, self._num_classes)
        torch.save(conf_matrix, file_path)
        
        
    