import os
import struct
import numpy as np
from PIL import Image
import csv

class DataLakePipeline:
    def __init__(self, input_dir, output_dir, file_prefix, record_size=8199, struct_fmt='>2H8sI4B4H2B30x8128s11x'):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.file_prefix = file_prefix
        self.record_size = record_size
        self.struct_fmt = struct_fmt
        self.csv_path = os.path.join(self.output_dir, f'{self.file_prefix.lower()}_metadata_master.csv')

    def setup_environment(self):
        os.makedirs(self.output_dir, exist_ok=True)
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Image_ID', 'Label_Unicode', 'Label_JIS', 'Source_File', 'Pixel_Mean'])

    def extract_record(self, record):
        return struct.unpack(self.struct_fmt, record)

    def transform_record(self, extracted_data):
        jis_code = extracted_data[1]
        image_data = extracted_data[14]

        try:
            unicode_char = (jis_code | 0x8080).to_bytes(2, 'big').decode('euc_jp')
        except Exception:
            unicode_char = 'Unknown'

        # Đọc ảnh bằng Numpy
        raw_data = np.frombuffer(image_data, dtype=np.uint8)
        img_flat = np.zeros(128 * 127, dtype=np.uint8)
        img_flat[0::2] = (raw_data >> 4) & 0x0F
        img_flat[1::2] = raw_data & 0x0F
        img_array = img_flat.reshape((127, 128))
        img_inverted = (255 - (img_array * 17)).astype(np.uint8)
        
        margin = 8
        img_inverted[:margin, :] = 0
        img_inverted[-margin:, :] = 0
        img_inverted[:, :margin] = 0
        img_inverted[:, -margin:] = 0
        
        pixel_mean = round(np.mean(img_inverted), 2)
        return unicode_char, hex(jis_code), img_inverted, pixel_mean

    def load_data(self, file_name, images, metadata):
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(metadata)

        npz_path = os.path.join(self.output_dir, f"{file_name}_images_lake.npz")
        np.savez_compressed(npz_path, images=np.array(images, dtype=np.uint8))

    def process_file(self, file_name):
        file_path = os.path.join(self.input_dir, file_name)
        images_list = []
        metadata_list = []
        count = 0

        with open(file_path, 'rb') as f:
            f.seek(self.record_size)
            
            while True:
                record = f.read(self.record_size)
                if not record or len(record) < self.record_size:
                    break
                    
                extracted_data = self.extract_record(record)
                unicode_char, hex_jis, img_array, pixel_mean = self.transform_record(extracted_data)
                
                image_id = f"{file_name}_{count:06d}"
                images_list.append(img_array)
                metadata_list.append([image_id, unicode_char, hex_jis, file_name, pixel_mean])
                
                count += 1

        self.load_data(file_name, images_list, metadata_list)
        return count

    def run(self):
        self.setup_environment()
        
        try:
            files = sorted([f for f in os.listdir(self.input_dir) if f.startswith(self.file_prefix)])
        except FileNotFoundError:
            print(f"Error: Directory '{self.input_dir}' not found.")
            return

        print(f"Found {len(files)} source files. Starting pipeline...")
        total_processed = 0

        for file_name in files:
            print(f"Processing {file_name}...")
            processed_count = self.process_file(file_name)
            total_processed += processed_count
            print(f"Successfully processed {processed_count} records from {file_name}.")

        print(f"Pipeline finished. Total records processed: {total_processed}")

if __name__ == '__main__':
    pipeline = DataLakePipeline(
        input_dir='ETL9G', 
        output_dir='Data_Lake', 
        file_prefix='ETL9G'
    )
    pipeline.run()