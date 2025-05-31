import os
import argparse
from test_script import main as test_model

from utils.utils import load_yaml_config


def main(data_configs_folder:str): 
    for filename in os.listdir(data_configs_folder):
        run_id = filename.split('_')[0]
        
        data_config_file_path = os.sep.join([data_configs_folder, filename])
        data_config = load_yaml_config(data_config_file_path)
        
        train_config_file_path = os.sep.join([data_config['save_eval_logs_path'], 'train_configs', f'{run_id}_train_config.yaml'])
        train_config = load_yaml_config(train_config_file_path)

        models_path = os.sep.join([data_config['save_eval_logs_path'], 'models'])
        for model_filename in os.listdir(models_path):
            if run_id in model_filename:
                model_path = os.sep.join([models_path, model_filename])

                # update/override train_config model parameter to load pretrained model
                train_config['model']['load_model_path'] = model_path
                test_model(run_id, data_config, train_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_configs', type=str, required=True)
    args = parser.parse_args()

    main(args.data_configs)
