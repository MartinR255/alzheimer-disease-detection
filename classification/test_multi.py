import os
import argparse
import pandas as pd
from resnet_3d_test import main as resnet_test

from utils.utils import load_yaml_config

def get_models_paths(self):
    ...


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_config', type=str, required=True)
    parser.add_argument('--train_params', type=str, required=True)
    parser.add_argument('--models_folder', type=str, required=True)
    args = parser.parse_args()

    data_config = load_yaml_config(args.data_config)

    models_folder = args.models_folder
    get_models_paths()

    df = pd.read_csv(args.train_params)
    grouped = df.groupby('Network Type')
    for group_name, group_df in grouped:
        print(f"Group: {group_name}")
        # print(group_df)
    
        # call validation for group
        for index, row in group_df.iterrows():
            print(f"Row Index: {index}, Row Data: {row.to_dict()}")
            train_config = {
                'batch_size': row['Batch Size'],
                'num_workers': 0,
                'num_classes': row['Num Classees'],
                'model' : row['Network Type'],
                'load_model_path': '',
                'loss': row['Loss Function']

            }

            # resnet_test(
            #     run_id=row['ID'],
            #     data_config=data_config,
            #     train_config=train_config
            # )

    

    # main(
    #     run_id=args.run_id, 
    #     data_config_file_path=args.data_config,
    #     train_config_file_path=args.test_config
    # )