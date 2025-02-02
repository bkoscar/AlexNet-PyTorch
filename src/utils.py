import torch

def calculate_topk_accuracy(output, target, top_k=(1, 5)):
    """
    Calculate top-k accuracy for the given output and target.
    
    Args:
        output (torch.Tensor): Model predictions (batch_size, num_classes).
        target (torch.Tensor): True labels (batch_size).
        top_k (tuple): The top-k accuracies to compute (e.g., top-1, top-5).

    Returns:
        topk_accuracy (tuple): Top-k accuracies.
    """
    maxk = max(top_k)
    batch_size = target.size(0)

    # Get the top-k predicted classes (indices)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()

    # Check if the true label is in the top-k predictions
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    # Calculate top-k accuracy
    res = []
    for k in top_k:
        correct_k = correct[:k].sum().item()
        res.append(correct_k / batch_size)
    return res

class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0):
        self.patience = patience  # Número de épocas a esperar sin mejora
        self.verbose = verbose    # Mostrar mensajes cuando se detenga el entrenamiento
        self.delta = delta        # Cuánto debe mejorar la métrica para considerarse una mejora
        self.counter = 0          # Contador de épocas sin mejora
        self.best_loss = None     # Mejor pérdida de validación
        self.early_stop = False   # Bandera para indicar si se debe detener el entrenamiento

    def __call__(self, val_loss, model):
        # Si no se ha registrado la mejor pérdida, se registra la primera
        if self.best_loss is None:
            self.best_loss = val_loss
        # Si la pérdida ha mejorado en comparación con la mejor pérdida (considerando delta)
        elif val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0  # Reseteamos el contador si hubo mejora
        else:
            self.counter += 1  # Si no hay mejora, aumentamos el contador
            if self.counter >= self.patience:  # Si el contador supera la paciencia
                self.early_stop = True
                if self.verbose:
                    print(f"Early stopping triggered! Best validation loss: {self.best_loss:.4f}")

# # Instanciamos EarlyStopping
# early_stopping = EarlyStopping(patience=5, verbose=True)
