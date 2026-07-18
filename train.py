import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import os
import random
from sklearn.metrics import accuracy_score, f1_score

from dataset import CrisisMMDDataset
from crisis_multimodal_model import MultimodalCrisisClassifier

def train_model(data_dir, tsv_files, num_epochs=500, debug_fast_run=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("===========================================================")
    print(f" Iniciando entrenamiento avanzado (SOTA) en: {device}")
    print("===========================================================")
    
    print("Cargando el dataset global...")
    full_dataset = CrisisMMDDataset(tsv_files=tsv_files, root_dir=data_dir)
    
    if debug_fast_run:
        print(">> MODO LOCAL ACTIVO: Entrenando con un subconjunto (100 muestras).")
        # Tomar un subconjunto de 100 índices al azar para prueba rápida
        indices = random.sample(range(len(full_dataset)), min(100, len(full_dataset)))
        full_dataset = Subset(full_dataset, indices)
        
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Batch size conservador para CPU local
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = MultimodalCrisisClassifier(num_classes=2).to(device)
    
    # Pesos balanceados para la función de pérdida si hay desequilibrio (opcional)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01) # LR más alto porque entrenamos MLP desde 0
    
    # Scheduler para reducir el learning rate si no hay mejora
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_f1 = 0.0
    patience_counter = 0
    early_stopping_patience = 15 # Entrenamiento libre hasta que deje de aprender
    start_epoch = 0
    
    # --- SISTEMA DE REANUDACIÓN (CHECKPOINT) ---
    checkpoint_path = 'entrenamiento_pausado.pth'
    if os.path.exists(checkpoint_path):
        print(f"[*] Reanudando entrenamiento desde {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_f1 = checkpoint['best_f1']
        patience_counter = checkpoint['patience_counter']
        print(f"[*] Retomando en la Época {start_epoch+1}. Mejor F1 hasta ahora: {best_f1:.2f}%")
    else:
        print("[*] Iniciando entrenamiento desde cero.")
    
    for epoch in range(start_epoch, num_epochs):
        print(f"\n--- Época {epoch+1}/{num_epochs} ---")
        
        # Fase de Entrenamiento
        model.train()
        running_loss = 0.0
        train_preds, train_true = [], []
        
        for batch in tqdm(train_loader, desc="Entrenando"):
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(images, input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Clipping para estabilidad
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            
            train_preds.extend(predicted.cpu().numpy())
            train_true.extend(labels.cpu().numpy())
            
        train_acc = accuracy_score(train_true, train_preds) * 100
        train_f1 = f1_score(train_true, train_preds, average='weighted') * 100
        print(f"Pérdida Train: {running_loss/len(train_loader):.4f} | Acc: {train_acc:.2f}% | F1: {train_f1:.2f}%")
        
        # Fase de Validación
        model.eval()
        val_loss = 0.0
        val_preds, val_true = [], []
        
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
                
                val_preds.extend(predicted.cpu().numpy())
                val_true.extend(labels.cpu().numpy())
                
        val_acc = accuracy_score(val_true, val_preds) * 100
        val_f1 = f1_score(val_true, val_preds, average='weighted') * 100
        
        print(f"Pérdida Val: {val_loss/len(val_loader):.4f} | Acc: {val_acc:.2f}% | F1: {val_f1:.2f}%")
        
        # Step del scheduler basado en el F1-Score
        scheduler.step(val_f1)
        
        # Guardar el mejor modelo SOTA
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), 'mejor_modelo_crisis_avanzado.pth')
            print(">> ¡Nuevo modelo SOTA guardado (mejor_modelo_crisis_avanzado.pth)!")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early Stopping activado en época {epoch+1}. El modelo ha dejado de aprender.")
                break
                
        # --- GUARDAR CHECKPOINT PARA PODER PAUSAR Y APAGAR LA PC ---
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_f1': best_f1,
            'patience_counter': patience_counter
        }, checkpoint_path)
        print(">> Progreso guardado. Puedes apagar la PC de forma segura (presiona Ctrl+C antes de apagar).")

    print(f"\nEntrenamiento completado. Mejor F1-Score: {best_f1:.2f}%")

if __name__ == "__main__":
    RUTA_DATOS = r"C:\Users\josep\.cache\kagglehub\datasets\mohammadabdulbasit\crisismmd\versions\1\CrisisMMD_v2.0"
    archivos = [
        "california_wildfires_final_data.tsv",
        "iraq_iran_earthquake_final_data.tsv",
        "mexico_earthquake_final_data.tsv",
        "hurricane_harvey_final_data.tsv",
        "hurricane_irma_final_data.tsv",
        "hurricane_maria_final_data.tsv",
        "srilanka_floods_final_data.tsv"
    ]
    rutas_tsv = [os.path.join(RUTA_DATOS, "annotations", archivo) for archivo in archivos]
    train_model(data_dir=RUTA_DATOS, tsv_files=rutas_tsv, num_epochs=500, debug_fast_run=False)
