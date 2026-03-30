import os
import sqlite3
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class Dataset(Dataset):
    # Thay vì truyền db_path, ta truyền thẳng dataframe đã được cắt (Train/Val/Test) vào đây
    def __init__(self, metadata_df, lake_dir, label_encoder, transform=None):
        # reset_index để đảm bảo index chạy từ 0 đến len(df), tránh lỗi khi gọi __getitem__
        self.metadata = metadata_df.reset_index(drop=True)
        self.lake_dir = lake_dir
        self.transform = transform
        self.label_encoder = label_encoder
        
        # Biến đổi nhãn text thành số
        self.labels = self.label_encoder.transform(self.metadata['Label_Unicode'])
        self.file_cache = {}

    def load_image(self, image_id, source_file):
        if source_file == 'User_Upload':
            file_path = os.path.join(self.lake_dir, f"{image_id}.npy")
            return np.load(file_path)

        if source_file not in self.file_cache:
            npz_path = os.path.join(self.lake_dir, f"{source_file}_images_lake.npz")
            self.file_cache[source_file] = np.load(npz_path)['images']

        img_index = int(image_id.split('_')[-1])
        return self.file_cache[source_file][img_index]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_np = self.load_image(row['Image_ID'], row['Source_File'])
        
        if self.transform:
            img_tensor = self.transform(img_np)
        else:
            img_tensor = torch.from_numpy(img_np).float().unsqueeze(0) / 255.0

        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        return img_tensor, label_tensor

class DataLoaderManager:
    def __init__(self, db_path, lake_dir, batch_size=64):
        self.db_path = db_path
        self.lake_dir = lake_dir
        self.batch_size = batch_size
        self.transform = self.get_transforms()

    def get_transforms(self):
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def fetch_full_metadata(self):
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT Image_ID, Source_File, Label_Unicode FROM Image_Metadata", conn)
        conn.close()
        return df

    def create_dataloaders(self, train_ratio=0.8, val_ratio=0.1):
        df_full = self.fetch_full_metadata()

        # 1. Khởi tạo LabelEncoder học trên TOÀN BỘ nhãn để đảm bảo không sót chữ nào
        le = LabelEncoder()
        le.fit(df_full['Label_Unicode'])

        test_ratio = 1.0 - train_ratio - val_ratio
        val_test_ratio = val_ratio + test_ratio

        df_train, df_temp = train_test_split(
            df_full, 
            test_size=val_test_ratio, 
            random_state=42, 
            stratify=df_full['Label_Unicode']
        )

        relative_val_ratio = val_ratio / val_test_ratio
        df_val, df_test = train_test_split(
            df_temp, 
            train_size=relative_val_ratio, 
            random_state=42, 
            stratify=df_temp['Label_Unicode']
        )

        print(f"Stratified Split completed:")
        print(f"  - Train: {len(df_train)} samples")
        print(f"  - Val:   {len(df_val)} samples")
        print(f"  - Test:  {len(df_test)} samples")

        train_dataset = Dataset(df_train, self.lake_dir, le, self.transform)
        val_dataset = Dataset(df_val, self.lake_dir, le, self.transform)
        test_dataset = Dataset(df_test, self.lake_dir, le, self.transform)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader, le