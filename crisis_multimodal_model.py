import torch
import torch.nn as nn
from torchvision import models
from transformers import DistilBertModel

class MultimodalCrisisClassifier(nn.Module):
    def __init__(self, num_classes, freeze_pretrained=False):
        """
        Arquitectura híbrida que fusiona CNN (Imágenes) y PLN (Texto).
        Ideal para ejecutar en Google Colab con soporte GPU.
        
        Args:
            num_classes (int): Número de categorías de salida (ej. niveles de gravedad).
            freeze_pretrained (bool): Si es True, congela los pesos base de ResNet y BERT.
        """
        super(MultimodalCrisisClassifier, self).__init__()
        
        # -------------------------------------------------------------
        # RAMA DE VISIÓN (CNN) - ResNet50
        # -------------------------------------------------------------
        # Cargamos ResNet-50 preentrenada en ImageNet
        self.image_model = models.resnet50(pretrained=True)
        
        if freeze_pretrained:
            for param in self.image_model.parameters():
                param.requires_grad = False
                
        # Modificamos la última capa para extraer características (ignoramos clasificación original)
        num_ftrs_img = self.image_model.fc.in_features
        self.image_model.fc = nn.Identity() 
        
        # -------------------------------------------------------------
        # RAMA DE LENGUAJE (PLN) - DistilBERT
        # -------------------------------------------------------------
        # Cargamos DistilBERT preentrenado (versión ligera y rápida)
        self.text_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        if freeze_pretrained:
            for param in self.text_model.parameters():
                param.requires_grad = False
                
        num_ftrs_txt = self.text_model.config.hidden_size # Generalmente 768 para DistilBERT
        
        # -------------------------------------------------------------
        # CAPAS DE FUSIÓN Y CLASIFICACIÓN
        # -------------------------------------------------------------
        # Dimensión total tras concatenar imagen y texto
        combined_features = num_ftrs_img + num_ftrs_txt # 2048 + 768 = 2816
        
        self.classifier = nn.Sequential(
            nn.Linear(combined_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5), # Regularización para evitar sobreajuste
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
            # Nota: La activación Softmax/Sigmoid se aplica usualmente junto con la función de pérdida (CrossEntropyLoss) en PyTorch
        )

    def forward(self, images, input_ids, attention_mask):
        """
        Flujo de los datos (Forward pass) a través de la red.
        """
        # Extraer características visuales (Salida: [Batch, 2048])
        img_features = self.image_model(images)
        
        # Extraer características semánticas del token [CLS] (Salida: [Batch, 768])
        txt_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        txt_features = txt_outputs.last_hidden_state[:, 0, :] 
        
        # Fusión temprana: Concatenación de características
        combined = torch.cat((img_features, txt_features), dim=1)
        
        # Pasar por capas densas de clasificación
        logits = self.classifier(combined)
        
        return logits

# =====================================================================
# EJEMPLO DE USO Y CONFIGURACIÓN DEL ENTORNO (Colab/Jupyter)
# =====================================================================
if __name__ == "__main__":
    # 1. Definir dispositivo (GPU si está disponible, sino CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Ejecutando en el entorno con dispositivo: {device}")
    
    # 2. Instanciar el modelo (ejemplo para 3 clases de emergencia)
    model = MultimodalCrisisClassifier(num_classes=3).to(device)
    
    # 3. Definir Función de Pérdida
    criterion = nn.CrossEntropyLoss()
    
    # 4. Definir Optimizador (AdamW es ideal para arquitecturas con Transformers)
    # Tasa de aprendizaje pequeña (2e-5) para no destruir los pesos preentrenados
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    
    print("\nResumen del modelo cargado con éxito. Listo para el bucle de entrenamiento.")
    # (El bucle de entrenamiento se implementaría aquí iterando sobre los DataLoaders de CrisisMMD)
