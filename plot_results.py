import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Estilo para publicaciones académicas
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

def plot_learning_curves():
    epochs = np.arange(1, 11)
    
    # Datos simulados de un modelo SOTA (EfficientNet + DistilBERT)
    train_loss = [0.65, 0.42, 0.31, 0.25, 0.20, 0.16, 0.13, 0.11, 0.09, 0.08]
    val_loss =   [0.58, 0.39, 0.29, 0.26, 0.24, 0.23, 0.22, 0.23, 0.24, 0.25] # Early stopping here
    
    train_acc = [65.2, 78.4, 85.1, 89.3, 92.5, 94.6, 96.1, 97.2, 98.0, 98.5]
    val_acc =   [70.1, 82.3, 86.7, 88.5, 89.9, 90.5, 90.7, 90.6, 90.4, 90.5]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Curva de Pérdida
    ax1.plot(epochs, train_loss, 'b-o', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_loss, 'r--s', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontweight='bold')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Cross Entropy Loss')
    ax1.legend()
    ax1.axvline(x=7, color='grey', linestyle=':', label='Early Stopping Point')

    # Curva de Precisión
    ax2.plot(epochs, train_acc, 'g-o', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_acc, 'm--s', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training and Validation Accuracy', fontweight='bold')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.axvline(x=7, color='grey', linestyle=':')

    plt.tight_layout()
    plt.savefig('learning_curves.png', dpi=300)
    print("Gráfica de curvas de aprendizaje guardada: learning_curves.png")

def plot_confusion_matrix():
    # Matriz de confusión simulada
    # Clases: No Informativo (0), Informativo (1)
    cm = np.array([[785,  65],
                   [ 82, 668]])
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No-Hate/No-Info', 'Hate/Info'],
                yticklabels=['No-Hate/No-Info', 'Hate/Info'],
                annot_kws={"size": 14, "weight": "bold"})
    
    plt.title('Confusion Matrix on Test Set', fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    print("Matriz de confusión guardada: confusion_matrix.png")

if __name__ == '__main__':
    plot_learning_curves()
    plot_confusion_matrix()
