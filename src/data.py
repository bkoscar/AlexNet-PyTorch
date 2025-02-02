import torch 
import torchvision 
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class CIFAR100Dataset:
    
    def __init__(self, dataset_root: str, batch_size: int = 2, num_workers: int = 0, pin_memory: bool = False):
        self.dataset_root = dataset_root 
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.transform = transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
                                            ])

    def get_dataset(self):
        train_dataset = torchvision.datasets.CIFAR100(root=self.dataset_root, train=True, download=True, transform=self.transform)
        test_dataset = torchvision.datasets.CIFAR100(root=self.dataset_root, train=False, download=True, transform=self.transform)
        return train_dataset, test_dataset

    def get_dataloader(self):
        train_dataset, test_dataset = self.get_dataset()
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=self.pin_memory)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)
        return train_loader, test_loader
    
    def imshow(self, img):
        # Desnormalizar la imagen
        mean = torch.tensor([0.5071, 0.4867, 0.4408]).view(3, 1, 1)
        std = torch.tensor([0.2675, 0.2565, 0.2761]).view(3, 1, 1)
        img = img * std + mean  # Desnormalización
        # Clipping para evitar valores fuera de [0,1]
        img = torch.clamp(img, 0, 1)
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))  # Cambiar de (C, H, W) a (H, W, C)
        plt.show()

    
    def visualice_data(self):
        train_loader, _ = self.get_dataloader()
        train_dataset, test_dataset = self.get_dataset()
        class_names = train_dataset.classes
        class_to_idx = train_dataset.class_to_idx
        dataiter = iter(train_loader)
        images, labels = next(dataiter)
        self.imshow(torchvision.utils.make_grid(images))
        for label in labels:
            print(f"Índice: {label.item()}, Clase: {class_names[label.item()]}")
    
    def dataset_info(self):
        train_dataset, test_dataset = self.get_dataset()
        class_names = train_dataset.classes
        class_to_idx = train_dataset.class_to_idx
        print(f"Total de clases: {len(class_names)}")
        print(f"Total de imágenes en el conjunto de entrenamiento: {len(train_dataset)}")
        print(f"Total de imágenes en el conjunto de prueba: {len(test_dataset)}")
        print(f"Total de imágenes en CIFAR-100: {len(train_dataset) + len(test_dataset)}")
