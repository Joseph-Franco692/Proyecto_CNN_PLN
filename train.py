import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import os
import random

from dataset import CrisisMMDDataset
from crisis_multimodal_model import MultimodalCrisisClassifier

def train_model(data_dir, tsv_file, num_epochs=1, batch_size=4, learning_rate=2e-5, debug_fast_run=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"===========================================================")
    print(f" Iniciando entrenamiento local en: {device.type.upper()}")
    print(f"===========================================================")

    print("Cargando el dataset...")
    full_dataset = CrisisMMDDataset(tsv_file=tsv_file, root_dir=data_dir)
    
    # Para poder ejecutar localmente sin GPU en tiempo razonable, usamos un subconjunto
    if debug_fast_run:
        print(">> MODO LOCAL ACTIVO: Entrenando con un subconjunto reducido para no congelar tu PC.")
        indices = list(range(len(full_dataset)))
        random.shuffle(indices)
        subset_indices = indices[:100] # Usar solo 100 muestras locales
        full_dataset = Subset(full_dataset, subset_indices)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Muestras de entrenamiento: {len(train_dataset)} | Validación: {len(val_dataset)}")

    model = MultimodalCrisisClassifier(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f"\n--- Época {epoch+1}/{num_epochs} ---")
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch in tqdm(train_loader, desc="Entrenando"):
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images, input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        train_acc = 100 * correct_train / total_train
        print(f"Pérdida Entrenamiento: {running_loss/len(train_loader):.4f} | Precisión: {train_acc:.2f}%")
        
        # Fase de Validación
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validando"):
                images = batch['image'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(images, input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
        val_acc = 100 * correct_val / total_val
        print(f"Pérdida Validación: {val_loss/len(val_loader):.4f} | Precisión: {val_acc:.2f}%")
        
        # Guardar el mejor modelo
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'mejor_modelo_crisis.pth')
            print(">> ¡Nuevo mejor modelo guardado localmente (mejor_modelo_crisis.pth)!")

    print(f"\nEntrenamiento finalizado. Mejor precisión en validación: {best_val_acc:.2f}%")

if __name__ == "__main__":
    RUTA_DATOS = r"C:\Users\josep\.cache\kagglehub\datasets\mohammadabdulbasit\crisismmd\versions\1\CrisisMMD_v2.0"
    RUTA_TSV = os.path.join(RUTA_DATOS, "annotations", "california_wildfires_final_data.tsv")
    train_model(data_dir=RUTA_DATOS, tsv_file=RUTA_TSV, num_epochs=1, debug_fast_run=True)
