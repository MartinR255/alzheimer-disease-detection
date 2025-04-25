import os
import re
import argparse
import pandas as pd
from resnet_3d_test import main as resnet_test
from densenet_3d_test import main as densenet_test

from utils.utils import load_yaml_config

def get_models_paths(path:str):
    models_paths = {}
    file_pattern = re.compile(r'^(\d+)_(\d+)\.txt$') # pth

    for file_name in os.listdir(path):
        run_id, epoch = file_pattern.match(file_name).groups()
        model_path = os.sep.join([path, file_name])
        models_paths.setdefault(run_id, []).append({'path': model_path, 'epoch': epoch})
   
    return models_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_config', type=str, required=True)
    parser.add_argument('--train_params', type=str, required=True)
    parser.add_argument('--models_folder', type=str, required=True)
    args = parser.parse_args()

    data_config = load_yaml_config(args.data_config)
    models_paths = get_models_paths(args.models_folder)
    
    

    resnets = [ 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet18p', 'resnet34p', 'resnet50p', 'resnet101p']
    densenets = ['densenet121', 'densenet121', 'densenet169', 'densenet201']


    df = pd.read_csv(args.train_params)
    grouped = df.groupby('Network Type')
    for group_name, group_df in grouped:
        
        for index, row in group_df.iterrows():
            # print(f"Row Index: {index}, Row Data: {row.to_dict()}")
            row_data = row.to_dict()
            row_id = row_data['ID']
            
            if not(str(row_id) in  models_paths.keys()):
                continue
            model_run_data = models_paths[str(row_id)]
            for model_data in model_run_data:
                train_config = {
                    'batch_size': row_data['Batch Size'],
                    'num_workers': 0,
                    'num_classes': row_data['Num Classes'],
                    'model' : row_data['Network Type'],
                    'load_model_path': model_data['path'],
                    'epoch' : model_data['epoch'],
                    'loss': row_data['Loss Function']
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


    

    # main(
    #     run_id=args.run_id, 
    #     data_config_file_path=args.data_config,
    #     train_config_file_path=args.test_config
    # )