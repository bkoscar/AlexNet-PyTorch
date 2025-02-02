from src.training import Trainer
import argparse
import yaml
import wandb
import torch
from datetime import datetime


def get_gpu_info():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(device)
        total_memory = torch.cuda.get_device_properties(device).total_memory / 1e9  # en GB
        return {
            "device": device,
            "gpu_name": gpu_name,
            "total_memory_GB": total_memory
        }
    else:
        return {
            "device": "cpu",
            "gpu_name": "No GPU"
        }



def load_hyperparameters(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def run_train(args):
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d_%H-%M") 
    hyperparameters_config = load_hyperparameters(args.config_path)
    # Inicializar wandb
    wandb.init(project="AlexNet CIFAR100")
    # wandb.config.update(hyperparameters_config)
    wandb.config.update({
        "hyperparameters_config": hyperparameters_config,
        "dataset_root": args.dataset_root,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "current_time": current_time,
        "checkpoint_dir": hyperparameters_config['paths']['checkpoint_dir'],
        "best_model_dir": hyperparameters_config['paths']['best_model_dir'],
        "checkpoint_interval": hyperparameters_config['checkpoint_interval'],
        **get_gpu_info() 
    })

    trainer = Trainer(
        config=wandb.config
    )
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR100 training")
    parser.add_argument("--dataset_root", type=str, default="./data", help="Root directory of CIFAR100 dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for dataloader")
    parser.add_argument("--pin_memory", type=bool, default=False, help="Pin memory for dataloader")
    parser.add_argument("--config_path", type=str, default="./config/hyperparameters.yaml", help="Path to the configuration file")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for optimizer")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs for training")
    args = parser.parse_args()
    run_train(args)