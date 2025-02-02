import os
import torch
import wandb
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
from src.model import AlexNet
from src.data import CIFAR100Dataset
from src.utils import calculate_topk_accuracy


class Trainer:
    def __init__(self, config):
        self.config = config
        # Determinar el dispositivo (GPU o CPU)
        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        # Cargar el dataset
        self.dataset = CIFAR100Dataset(
            dataset_root=config.dataset_root,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory
        )
        self.train_loader, self.test_loader = self.dataset.get_dataloader()
        # Crear el modelo y moverlo al dispositivo (GPU o CPU)
        self.model = AlexNet(config=self.config.hyperparameters_config).to(self.device)
        # Definir el criterio de pérdida y el optimizador
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.learning_rate, momentum=0.9, weight_decay=0.000005)
        # self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=0.0005)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size = 10, gamma=0.1)
        # Crear carpetas para checkpoints y mejores modelos
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.best_model_dir, exist_ok=True)
        self.best_top1_acc = 0.0  # Mejor top-1 accuracy

    def save_checkpoint(self, epoch):
        """Guardar un checkpoint durante el entrenamiento."""
        checkpoint_path = os.path.join(self.config.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch}")

    def save_best_model(self):
        """Guardar el mejor modelo según el top-1 accuracy."""
        best_model_path = os.path.join(self.config.best_model_dir, "best_model.pth")
        torch.save(self.model.state_dict(), best_model_path)
        print(f"Best model saved with top-1 accuracy: {self.best_top1_acc:.4f}")
        # Log best model to wandb
        wandb.log({"best_model_top1_accuracy": self.best_top1_acc})

    def train(self):
        self.model.train()
        for epoch in range(self.config.epochs):
            running_loss = 0.0
            correct_1 = 0
            correct_5 = 0
            total = 0
            # Usar tqdm para mostrar una barra de progreso
            for inputs, labels in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.epochs}", dynamic_ncols=True):
                # Mover los datos al dispositivo (GPU o CPU)
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # Calcular top-1 y top-5 accuracy
                topk_acc = calculate_topk_accuracy(outputs, labels, top_k=(1, 5))
                correct_1 += topk_acc[0] * inputs.size(0)
                correct_5 += topk_acc[1] * inputs.size(0)
                total += inputs.size(0)
                running_loss += loss.item()
            # Cálculo de las métricas para la época
            epoch_loss = running_loss / len(self.train_loader)
            epoch_top1_acc = correct_1 / total
            epoch_top5_acc = correct_5 / total
            # Logear las métricas en wandb
            wandb.log({
                "epoch": epoch + 1,
                "loss": epoch_loss,
                "learning_rate": self.optimizer.param_groups[0]['lr'],
                "top1_accuracy": epoch_top1_acc,
                "top5_accuracy": epoch_top5_acc,
            })
            print(f"Epoch [{epoch+1}/{self.config.epochs}], Loss: {epoch_loss:.4f}, Top-1 Accuracy: {epoch_top1_acc:.4f}, Top-5 Accuracy: {epoch_top5_acc:.4f}")
            # Guardar checkpoint
            if (epoch + 1) % self.config.checkpoint_interval == 0:
                self.save_checkpoint(epoch)
            self.test()  # Realizar evaluación después de cada época
            # Actualizar el scheduler
            self.scheduler.step()

    def test(self):
        self.model.eval()  # Establecer el modelo en modo evaluación
        test_loss = 0.0
        correct_1 = 0
        correct_5 = 0
        total = 0
        with torch.no_grad():  # No necesitamos gradientes durante la evaluación
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                # Calcular top-1 y top-5 accuracy
                topk_acc = calculate_topk_accuracy(outputs, labels, top_k=(1, 5))
                correct_1 += topk_acc[0] * inputs.size(0)
                correct_5 += topk_acc[1] * inputs.size(0)
                total += inputs.size(0)
                test_loss += loss.item()
        # Calcular la pérdida total y las métricas de precisión
        test_loss = test_loss / len(self.test_loader)
        test_top1_acc = correct_1 / total
        test_top5_acc = correct_5 / total
        # Logear las métricas de evaluación en wandb
        wandb.log({
            "test_loss": test_loss,
            "test_top1_accuracy": test_top1_acc,
            "test_top5_accuracy": test_top5_acc,
        })
        print(f"Test Loss: {test_loss:.4f}, Test Top-1 Accuracy: {test_top1_acc:.4f}, Test Top-5 Accuracy: {test_top5_acc:.4f}")
          # Guardar el mejor modelo en base al test top-1 accuracy
        if test_top1_acc > self.best_top1_acc:
            self.best_top1_acc = test_top1_acc
            self.save_best_model()
