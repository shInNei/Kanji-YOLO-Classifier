import os
import struct
import numpy as np
import pandas as pd
import sqlite3
import csv
import glob
from PIL import Image
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

class DataWarehouseManager:
    def __init__(self, db_url=None):
        load_dotenv()
        if db_url is None:
            db_url = os.getenv("DB_URL")
        if db_url is None:
            raise ValueError("Không tìm thấy DB_URL trong file .env!")
            
        self.engine = create_engine(db_url)

    def upsert_metadata(self, df):
        df['is_trained'] = 0 
        
        with self.engine.connect() as conn:
            # 1. Đẩy data vào bảng tạm
            df.to_sql('temp_table', conn, if_exists='replace', index=False)
            
            # 2. Chuyển data từ bảng tạm sang bảng chính bằng cú pháp PostgreSQL
            upsert_query = text("""
                INSERT INTO "Image_Metadata" ("Image_ID", "Label_Unicode", "Label_JIS", "Source_File", "Pixel_Mean", is_trained)
                SELECT "Image_ID", "Label_Unicode", "Label_JIS", "Source_File", "Pixel_Mean", is_trained 
                FROM temp_table
                ON CONFLICT ("Image_ID") DO NOTHING;
            """)
            conn.execute(upsert_query)
            conn.commit()

    def get_untrained_count(self):
        with self.engine.connect() as conn:
            result = conn.execute(text('SELECT COUNT(*) FROM "Image_Metadata" WHERE is_trained = 0'))
            return result.fetchone()[0]

    def mark_as_trained(self):
        with self.engine.connect() as conn:
            conn.execute(text('UPDATE "Image_Metadata" SET is_trained = 1 WHERE is_trained = 0'))
            conn.commit()

    def fetch_all_metadata(self):
        # Pandas làm việc trực tiếp với SQLAlchemy engine
        return pd.read_sql_query('SELECT * FROM "Image_Metadata"', self.engine)

    def close(self):
        self.engine.dispose()

class DataLakePipeline:
    def __init__(self, input_dir, lake_dir, db_manager):
        self.input_dir = input_dir
        self.lake_dir = lake_dir
        self.db = db_manager
        os.makedirs(self.lake_dir, exist_ok=True)

    def process_binary_etl9g(self, file_path):
        file_name = os.path.basename(file_path)
        record_size = 8199
        struct_fmt = '>2H8sI4B4H2B30x8128s11x'
        images_list = []
        metadata_list = []
        count = 0

        with open(file_path, 'rb') as f:
            f.seek(record_size)
            while True:
                record = f.read(record_size)
                if not record or len(record) < record_size: break
                
                extracted_data = struct.unpack(struct_fmt, record)
                jis_code, image_data = extracted_data[1], extracted_data[14]
                try:
                    unicode_char = (jis_code | 0x8080).to_bytes(2, 'big').decode('euc_jp')
                except Exception:
                    unicode_char = 'Unknown'

                raw_data = np.frombuffer(image_data, dtype=np.uint8)
                img_flat = np.zeros(128 * 127, dtype=np.uint8)
                img_flat[0::2] = (raw_data >> 4) & 0x0F
                img_flat[1::2] = raw_data & 0x0F
                img_array = img_flat.reshape((127, 128))
                img_inverted = (255 - (img_array * 17)).astype(np.uint8)
                
                pixel_mean = round(np.mean(img_inverted), 2)
                image_id = f"{file_name}_{count:06d}"
                
                images_list.append(img_inverted)
                metadata_list.append({'Image_ID': image_id, 'Label_Unicode': unicode_char, 
                                      'Label_JIS': hex(jis_code), 'Source_File': file_name, 'Pixel_Mean': pixel_mean})
                count += 1

        if images_list:
            np.savez_compressed(os.path.join(self.lake_dir, f"{file_name}_images_lake.npz"), images=np.array(images_list, dtype=np.uint8))
            self.db.upsert_metadata(pd.DataFrame(metadata_list))
        return count
        
    def process_binary_etl6(self, file_path):
        import os
        import struct
        import numpy as np
        import pandas as pd
        from PIL import Image

        file_name = os.path.basename(file_path)
        record_size = 2052
        struct_fmt = '>H2B8sI4B4H2B6x2016s' 
        images_list = []
        metadata_list = []
        count = 0

        # Bảng map Romaji 2-byte sang Katakana chuẩn của định dạng M-Type (ETL6)
        # Bao gồm đúng 46 ký tự Katakana theo đặc tả của database
        M_TYPE_KATAKANA_MAP = {
            ' A': 'ア', ' I': 'イ', ' U': 'ウ', ' E': 'エ', ' O': 'オ',
            'KA': 'カ', 'KI': 'キ', 'KU': 'ク', 'KE': 'ケ', 'KO': 'コ',
            'SA': 'サ', 'SI': 'シ', 'SU': 'ス', 'SE': 'セ', 'SO': 'ソ',
            'TA': 'タ', 'TI': 'チ', 'TU': 'ツ', 'TE': 'テ', 'TO': 'ト',
            'NA': 'ナ', 'NI': 'ニ', 'NU': 'ヌ', 'NE': 'ネ', 'NO': 'ノ',
            'HA': 'ハ', 'HI': 'ヒ', 'HU': 'フ', 'HE': 'ヘ', 'HO': 'ホ',
            'MA': 'マ', 'MI': 'ミ', 'MU': 'ム', 'ME': 'メ', 'MO': 'モ',
            'YA': 'ヤ', 'YU': 'ユ', 'YO': 'ヨ',
            'RA': 'ラ', 'RI': 'リ', 'RU': 'ル', 'RE': 'レ', 'RO': 'ロ',
            'WA': 'ワ', 'WO': 'ヲ', ' N': 'ン'
        }

        with open(file_path, 'rb') as f:
            while True:
                record = f.read(record_size)
                if not record or len(record) < record_size: break
                
                extracted_data = struct.unpack(struct_fmt, record)
                
                # Trích xuất 2 byte mã ký tự
                byte1 = extracted_data[1]
                byte2 = extracted_data[2]
                
                try:
                    # Kết hợp 2 byte thành chuỗi Romaji (VD: 'K' + 'A' = 'KA')
                    romaji_str = chr(byte1) + chr(byte2)
                except Exception:
                    continue
                
                #Chỉ lấy những chuỗi nằm trong bảng Katakana
                if romaji_str in M_TYPE_KATAKANA_MAP:
                    unicode_char = M_TYPE_KATAKANA_MAP[romaji_str]
                else:
                    # Bỏ qua các dữ liệu số (0-9) và chữ (A-Z) hoặc ký hiệu
                    continue

                # Giải mã ảnh 4-bit
                image_data = extracted_data[15]
                raw_data = np.frombuffer(image_data, dtype=np.uint8)
                img_flat = np.zeros(64 * 63, dtype=np.uint8)
                img_flat[0::2] = (raw_data >> 4) & 0x0F
                img_flat[1::2] = raw_data & 0x0F
                
                img_array = img_flat.reshape((63, 64))
                img_scaled = (img_array * 17).astype(np.uint8) 
                img_inverted = (255 - img_scaled).astype(np.uint8)

                resample_method = Image.Resampling.BILINEAR if hasattr(Image, 'Resampling') else Image.BILINEAR
                img_resized = np.array(Image.fromarray(img_inverted).resize((128, 127), resample_method))

                pixel_mean = round(np.mean(img_resized), 2)
                image_id = f"{file_name}_{count:06d}"
                
                images_list.append(img_resized)
                metadata_list.append({
                    'Image_ID': image_id, 
                    'Label_Unicode': unicode_char, 
                    'Label_JIS': romaji_str, # Lưu chuỗi Romaji để dễ theo dõi trong DB
                    'Source_File': file_name, 
                    'Pixel_Mean': pixel_mean
                })
                count += 1

        if images_list:
            np.savez_compressed(os.path.join(self.lake_dir, f"{file_name}_images_lake.npz"), images=np.array(images_list, dtype=np.uint8))
            self.db.upsert_metadata(pd.DataFrame(metadata_list))
        
        print(f"👉 File {file_name}: Đã trích xuất thành công {count} ảnh Katakana.")
        return count
    
    def process_raw_images(self, file_path, label):
        # Xử lý ảnh lẻ người dùng upload
        file_name = os.path.basename(file_path)
        img = Image.open(file_path).convert('L').resize((128, 127)) # Chuyển về Grayscale và resize
        img_array = np.array(img, dtype=np.uint8)
        
        image_id = f"img_{file_name.split('.')[0]}"
        pixel_mean = round(np.mean(img_array), 2)
        
        # Lưu npy cho ảnh lẻ
        np.save(os.path.join(self.lake_dir, f"{image_id}.npy"), img_array)
        
        df = pd.DataFrame([{
            'Image_ID': image_id, 'Label_Unicode': label, 'Label_JIS': 'N/A', 
            'Source_File': 'User_Upload', 'Pixel_Mean': pixel_mean
        }])
        self.db.upsert_metadata(df)
        return 1

    def run_ingestion(self):
        # Quét thư mục input để tìm file mới
        processed_count = 0
        for root, _, files in os.walk(self.input_dir):
            for file in files:
                file_path = os.path.join(root, file)
                
                if file.startswith('ETL9'):
                    processed_count += self.process_binary_etl9g(file_path)
                    os.remove(file_path)
                elif file.startswith('ETL6'):
                    processed_count += self.process_binary_etl6(file_path)
                    os.remove(file_path)
                elif file.endswith(('.png', '.jpg', '.jpeg')):
                    label = os.path.basename(root) 
                    processed_count += self.process_raw_images(file_path, label)
                    os.remove(file_path)
                    
        return processed_count