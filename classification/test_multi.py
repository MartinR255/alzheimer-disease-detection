import os
import re
import argparse
import pandas as pd
from resnet_3d_test import main as resnet_test
from densenet_3d_test import main as densenet_test

from utils.utils import load_yaml_config

resnets = [ 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet18p', 'resnet34p', 'resnet50p', 'resnet101p']
densenets = ['densenet121', 'densenet121', 'densenet169', 'densenet201']

def get_models_paths(path:str):
    models_paths = {}
    file_pattern = re.compile(r'^(\d+)_(\d+)\.pth$') 

    for file_name in os.listdir(path):
        run_id, epoch = file_pattern.match(file_name).groups()
        model_path = os.sep.join([path, file_name])
        models_paths.setdefault(run_id, []).append({'path': model_path, 'epoch': epoch})
   
    return models_paths


def test_models(data_config, models_paths, row_data):
    row_id = row_data['ID']

    model_run_data = models_paths[str(row_id)]
    for model_data in model_run_data:
        train_config = {
            'batch_size': row_data['Batch Size'],
            'num_workers': 0,
            'model' : {
                'name': row_data['Network Type'],
                'num_classes': row_data['Num Classes'],
                'spatial_dims': row_data['Spatial Dims'],
                'n_input_channels': row_data['Input Channels'],
                'load_model_path': model_data['path'],
                'dropout_rate_fc': row_data['Dropout Rate fc'],
                'dropout_rate_relu': row_data['Dropout Rate relu']
            },
            'epoch' : model_data['epoch'],
            'loss': {
                "name": row_data['Loss Function']
            }
        }
     
        if row_data['Network Type'] in resnets:
            resnet_test(
                run_id=row_id,
                data_config=data_config,
                train_config=train_config
            )
        elif row_data['Network Type'] in densenets:
            densenet_test(
                run_id=row_id,
                data_config=data_config,
                train_config=train_config
            )



def main(data_config:str, train_params:str, models_folder:str):
    data_config = load_yaml_config(data_config)
    models_paths = get_models_paths(models_folder)
    
    df = pd.read_csv(train_params)
    grouped = df.groupby('Network Type')
    for _, group_df in grouped:
        for _, row in group_df.iterrows():
            row_data = row.to_dict()
            test_models(data_config, models_paths, row_data)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_config', type=str, required=True)
    parser.add_argument('--train_params', type=str, required=True)
    parser.add_argument('--models_folder', type=str, required=True)
    args = parser.parse_args()

    main(args.data_config, args.train_params, args.models_folder)
