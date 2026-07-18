import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import DistilBertTokenizer

class CrisisMMDDataset(Dataset):
    def __init__(self, tsv_files, root_dir, max_length=128):
        """
        Args:
            tsv_files (list of string): Lista de rutas a los archivos TSV con anotaciones.
            root_dir (string): Directorio principal de las imágenes.
            max_length (int): Longitud máxima para el tokenizador de BERT.
        """
        self.root_dir = root_dir
        self.max_length = max_length
        
        # Cargar y concatenar múltiples TSVs
        dfs = []
        for tsv in tsv_files:
            df = pd.read_csv(tsv, sep='\t')
            dfs.append(df)
        self.data_frame = pd.concat(dfs, ignore_index=True)
        
        # Filtrar muestras unimodales si faltan textos o imágenes
        self.data_frame = self.data_frame.dropna(subset=['tweet_text', 'image_path', 'image_info'])
        
        # Mapear etiquetas a enteros (Clasificación de Informatividad)
        # 'informative' -> 1, 'not_informative' -> 0
        self.label_map = {'informative': 1, 'not_informative': 0}
        
        # Filtrar solo aquellas filas que tengan las etiquetas válidas
        self.data_frame = self.data_frame[self.data_frame['image_info'].isin(self.label_map.keys())]
        self.data_frame.reset_index(drop=True, inplace=True)
        
        self.root_dir = root_dir
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.max_length = max_length
        
        # Transformaciones de imagen estándar (Data Augmentation)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 1. Cargar y transformar imagen
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx]['image_path'])
        try:
            image = Image.open(img_name).convert('RGB')
        except Exception as e:
            # En caso de imagen corrupta, crear un tensor vacío o manejar el error
            image = Image.new('RGB', (224, 224))
            
        image = self.transform(image)

        # 2. Obtener y tokenizar texto
        text = str(self.data_frame.iloc[idx]['tweet_text'])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        # 3. Obtener etiqueta
        label_str = self.data_frame.iloc[idx]['image_info']
        label = self.label_map[label_str]

        return {
            'image': image,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Ejemplo de uso:
if __name__ == '__main__':
    # dataset = CrisisMMDDataset(tsv_file='ruta_al_tsv.tsv', root_dir='ruta_a_las_imagenes')
    print("Módulo dataset.py listo.")
