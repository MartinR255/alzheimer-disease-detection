import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from torch.utils.tensorboard import SummaryWriter


class Report:

    def __init__(self):
        self._tables = {}
    

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
        # self._tables[table_name]['table'] = pd.concat([self._tables[table_name]['table'], new_line])

        new_line.to_csv(self._tables[table_name]['file_path'], mode='a', header=False, index=False)
        
        
    