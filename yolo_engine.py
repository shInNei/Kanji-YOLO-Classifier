import os
import shutil
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from ultralytics import YOLO

class YoloDataLoader:
    def __init__(self, lake_dir='Data_Lake', output_dir='YOLO_Temp_Dataset'):
        self.lake_dir = lake_dir
        self.output_dir = output_dir

    def prepare_dataset(self, df_metadata):
        import re 
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
            
        df_metadata = df_metadata.copy()
        forbidden_map = str.maketrans('<>:"/\\|?*', '＜＞：”／＼｜？＊')
        df_metadata['Label_Unicode'] = df_metadata['Label_Unicode'].str.translate(forbidden_map)
        
        df_metadata['Label_Unicode'] = df_metadata['Label_Unicode'].apply(
            lambda x: re.sub(r'[\x00-\x1F\x7F]', '', str(x)).strip()
        )
        df_metadata.loc[df_metadata['Label_Unicode'] == '', 'Label_Unicode'] = 'Unknown'
            
        print("Đang chia tập dữ liệu theo tỉ lệ 80-10-10...")

        class_counts = df_metadata['Label_Unicode'].value_counts()
        min_samples = class_counts.min()

        if min_samples < 3:
            print(f"Phát hiện có nhãn chỉ chứa {min_samples} ảnh. Tự động chuyển sang chia ngẫu nhiên...")
            df_train, df_rem = train_test_split(df_metadata, train_size=0.8, random_state=42)
            df_val, df_test = train_test_split(df_rem, test_size=0.5, random_state=42)
        else:
            print("Dữ liệu đủ tiêu chuẩn, đang chia theo tỉ lệ Stratified Split...")
            df_train, df_rem = train_test_split(df_metadata, train_size=0.8, random_state=42, stratify=df_metadata['Label_Unicode'])
            df_val, df_test = train_test_split(df_rem, test_size=0.5, random_state=42, stratify=df_rem['Label_Unicode'])
        
        df_train['split'] = 'train'; df_val['split'] = 'val'; df_test['split'] = 'test'
        df_full = pd.concat([df_train, df_val, df_test])

        for split in ['train', 'val', 'test']:
            for label in df_full['Label_Unicode'].unique():
                os.makedirs(os.path.join(self.output_dir, split, label), exist_ok=True)

        BORDER_SIZE = 8   
        WHITE_COLOR = 255 

        unique_files = df_full['Source_File'].unique()
        for source_file in tqdm(unique_files, desc="Xuất ảnh cho YOLO"):
            subset = df_full[df_full['Source_File'] == source_file]
            
            if source_file.startswith('ETL9'):
                current_border = 8      # ETL9 cần viền 8px
                current_left_paint = 22 # ETL9 tẩy nhiễu gáy sách 22px
            else:
                current_border = 0      # ETL6 KHÔNG CẦN viền (tránh cắt chữ)
                current_left_paint = 0  # ETL6 KHÔNG CÓ nhiễu
            
            if source_file == 'User_Upload':
                for _, row in subset.iterrows():
                    img_id = row['Image_ID']
                    img_array = np.load(os.path.join(self.lake_dir, f"{img_id}.npy"))
                    self._process_and_save_image(img_array, img_id, row['split'], row['Label_Unicode'], current_border, current_left_paint, WHITE_COLOR)
            else:
                npz_path = os.path.join(self.lake_dir, f"{source_file}_images_lake.npz")
                if not os.path.exists(npz_path): continue
                data = np.load(npz_path)['images']
                
                for _, row in subset.iterrows():
                    img_id = row['Image_ID']
                    idx = int(img_id.split('_')[-1])
                    img_array = data[idx].copy()
                    self._process_and_save_image(img_array, img_id, row['split'], row['Label_Unicode'], current_border, current_left_paint, WHITE_COLOR)

        return self.output_dir

    def _process_and_save_image(self, img_array, img_id, split, label, border, left_paint, color):
        if border > 0:
            img_array[0:border, :] = color      
            img_array[-border:, :] = color     
            img_array[:, -border:] = color     
        if left_paint > 0:
            img_array[:, 0:left_paint] = color      
            
        save_path = os.path.join(self.output_dir, split, label, f"{img_id}.png")
        Image.fromarray(img_array).save(save_path)

    def cleanup(self):
        # Xóa toàn bộ ảnh rác sau khi train xong
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
            print("Đã dọn dẹp thư mục ảnh tạm YOLO.")

class YoloManager:
    def __init__(self, base_model_version='yolo26n-cls.pt'):
        self.current_model_path = base_model_version

    def train(self, data_dir, epochs=10, imgsz=64, batch=128):
        print(f"\n BẮT ĐẦU TRAIN VỚI MÔ HÌNH: {self.current_model_path}...")
        
        # 1. KHỞI TẠO LẠI MODEL TRONG MỖI LẦN TRAIN
        model = YOLO(self.current_model_path)

        results = model.train(
            data=data_dir,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,           
            device=0,
            workers=2       
        )
        
        try:
            run_dir = results.save_dir 
            best_model_path = os.path.join(run_dir, 'weights', 'best.pt')
            
            if os.path.exists(best_model_path):
                self.current_model_path = best_model_path
                print(f"Đã ghi nhớ Model mới cho đợt Train sau: {self.current_model_path}")
        except Exception as e:
            print(f"Không thể cập nhật model mới. Giữ nguyên model cũ. Lỗi: {e}")

        print("ĐÃ TRAIN XONG")
        return results