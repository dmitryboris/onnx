import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms

FAKE_DATASET_SIZE = 3200

class CustomImageDataset(Dataset):
    """Кастомный датасет для работы с папками классов"""
    
    def __init__(self, root_dir, transform=None, target_size=(224, 224)):
        """
        Args:
            root_dir (str): Путь к папке с классами
            transform: Аугментации для изображений
            target_size (tuple): Размер для ресайза изображений
        """
        self.root_dir = root_dir
        self.target_size = target_size
        
        # Получаем список классов (папок)
        self.classes = sorted([d for d in os.listdir(root_dir) 
                              if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        # Собираем все пути к изображениям
        self.images = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    img_path = os.path.join(class_dir, img_name)
                    image = Image.open(img_path).convert('RGB')
                    image = transform(image)
                    self.images.append(image)
                    self.labels.append(class_idx)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        return image, label
    
    def get_class_names(self):
        """Возвращает список имен классов"""
        return self.classes

class RandomImageDataset(Dataset):
    def __init__(self, target_size=(3, 224, 224)):
        self.target_size = target_size
        images = [torch.randn(self.target_size) for _ in range(FAKE_DATASET_SIZE)]
        labels = [torch.randint(0, 1000, (1,)) for _ in range(FAKE_DATASET_SIZE)]
        self.images = images
        self.labels = labels

    def __len__(self):
        return FAKE_DATASET_SIZE
    
    def __getitem__(self, idx):
        image, label = self.images[idx], self.labels[idx]
        return image, label

if __name__ == '__main__':
    dataset = RandomImageDataset()
    print(dataset[0])
    dl = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=True)
    for batch in dl:
        print(batch[0].shape)