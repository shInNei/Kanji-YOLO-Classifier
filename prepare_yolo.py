import os
import sqlite3
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def create_yolo_dataset(db_path='data_warehouse.db', lake_dir='Data_Lake', output_dir='YOLO_Dataset'):
    # 1. Đọc Metadata
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT Image_ID, Source_File, Label_Unicode FROM Image_Metadata", conn)
    conn.close()

    print("Đang chia tập dữ liệu theo tỉ lệ 80% Train - 10% Val - 10% Test...")
    
    # Bước 1: Tách 80% Train và 20% còn lại (rem)
    df_train, df_rem = train_test_split(
        df, train_size=0.8, random_state=42, stratify=df['Label_Unicode']
    )
    
    # Bước 2: Tách 20% còn lại thành Val (10%) và Test (10%)
    df_val, df_test = train_test_split(
        df_rem, test_size=0.5, random_state=42, stratify=df_rem['Label_Unicode']
    )
    
    # Gán nhãn split
    df_train = df_train.copy(); df_train['split'] = 'train'
    df_val = df_val.copy(); df_val['split'] = 'val'
    df_test = df_test.copy(); df_test['split'] = 'test'
    
    df_full = pd.concat([df_train, df_val, df_test])

    # Tạo cấu trúc thư mục YOLO (3 tập)
    print("Đang tạo cấu trúc thư mục Train/Val/Test...")
    for split in ['train', 'val', 'test']:
        for label in df_full['Label_Unicode'].unique():
            os.makedirs(os.path.join(output_dir, split, label), exist_ok=True)

    # 2. Xả nén và "Quét sơn trắng" tẩy nhiễu
    unique_files = df_full['Source_File'].unique()
    print(f"Bắt đầu xử lý {len(df_full)} ảnh với thông số Left-22, Border-8...")

    BORDER_SIZE = 8   
    LEFT_PAINT = 22   
    WHITE_COLOR = 255 

    for source_file in tqdm(unique_files, desc="Processing Files"):
        npz_path = os.path.join(lake_dir, f"{source_file}_images_lake.npz")
        if not os.path.exists(npz_path): continue
            
        data = np.load(npz_path)['images']
        subset = df_full[df_full['Source_File'] == source_file]
        
        for _, row in subset.iterrows():
            img_id = row['Image_ID']
            label = row['Label_Unicode']
            split = row['split']
            
            idx = int(img_id.split('_')[-1])
            img_array = data[idx].copy()
            
            # --- THỰC THI QUÉT SƠN TRẮNG CHUẨN V3 ---
            # Quét 3 viền (Trên, Dưới, Phải)
            img_array[0:BORDER_SIZE, :] = WHITE_COLOR      
            img_array[-BORDER_SIZE:, :] = WHITE_COLOR     
            img_array[:, -BORDER_SIZE:] = WHITE_COLOR     
            
            # Quét lề trái sâu (Diệt lem lề)
            img_array[:, 0:LEFT_PAINT] = WHITE_COLOR      
            
            # Lưu ra PNG
            save_path = os.path.join(output_dir, split, label, f"{img_id}.png")
            Image.fromarray(img_array).save(save_path)

    print(f"✅ Hoàn tất! Dataset sạch đã nằm trong '{output_dir}'.")

if __name__ == '__main__':
    create_yolo_dataset()