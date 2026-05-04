import time
from dotenv import load_dotenv
from datetime import datetime
from data_manager import DataWarehouseManager, DataLakePipeline
from yolo_engine import YoloDataLoader, YoloManager

load_dotenv()
INPUT_DIR = 'Landing_Zone'
LAKE_DIR = 'Data_Lake'
# DB_PATH = 'data_warehouse.db'

TRIGGER_RECORDS_THRESHOLD = 0  # Train khi có 10 sample mới
CHECK_INTERVAL_SECONDS = 10     # Mỗi 1 tiếng kiểm tra 1 lần

def generate_report(db, new_records):
    print("\n" + "="*40)
    print("📊 BÁO CÁO HỆ THỐNG DỮ LIỆU")
    print(f"Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Số lượng bản ghi vừa ingest: {new_records}")
    print(f"Số lượng dữ liệu chờ Train (Untrained): {db.get_untrained_count()}")
    
    # Optional Metric: Lấy histogram cơ bản của Pixel_Mean
    df = db.fetch_all_metadata()
    if not df.empty:
        print(f"Tổng dung lượng Data Lake hiện tại: {len(df)} records")
        mean_avg = df['Pixel_Mean'].mean()
        print(f"Trung bình cường độ sáng (Pixel Mean Average): {mean_avg:.2f}")
    print("="*40 + "\n")

def run_continuous_pipeline():
    print("Khởi động AI Auto-Train Pipeline...")
    # db = DataWarehouseManager(db_path=DB_PATH)
    db = DataWarehouseManager()
    data_lake = DataLakePipeline(input_dir=INPUT_DIR, lake_dir=LAKE_DIR, db_manager=db)
    
    yolo_loader = YoloDataLoader(lake_dir=LAKE_DIR)
    yolo_model = YoloManager('yolo26n-cls.pt')

    try:
        while True:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Đang kiểm tra dữ liệu mới...")
            
            # 1. Thu thập dữ liệu
            new_records = data_lake.run_ingestion()
            if new_records > 0:
                print(f"Đã thu thập {new_records} bản ghi mới vào Data Lake.")
            
            untrained_count = db.get_untrained_count()
            
            # 2. Kiểm tra điều kiện Trigger
            if untrained_count >= TRIGGER_RECORDS_THRESHOLD:
                print(f"🔥 Đạt ngưỡng kích hoạt ({untrained_count} >= {TRIGGER_RECORDS_THRESHOLD}). Bắt đầu quá trình Train...")
                
                # Fetch TOÀN BỘ dữ liệu để train (cả cũ lẫn mới) hoặc bạn có thể chỉnh query để train dữ liệu mới
                full_metadata = db.fetch_all_metadata()
                
                # Tiền xử lý: Xuất ảnh ra thư mục YOLO
                yolo_dataset_dir = yolo_loader.prepare_dataset(full_metadata)
                
                # Train Model
                yolo_model.train(data_dir=yolo_dataset_dir, epochs=10, imgsz=64, batch=128)
                
                # Dọn dẹp ảnh tạm để tiết kiệm ổ cứng
                yolo_loader.cleanup()
                
                # Đánh dấu dữ liệu đã train
                db.mark_as_trained()
                
                generate_report(db, new_records)
            else:
                print(f"Chưa đủ dữ liệu để train (Hiện tại: {untrained_count}/{TRIGGER_RECORDS_THRESHOLD}).")

            # Ngủ một khoảng thời gian trước khi quét lại
            time.sleep(CHECK_INTERVAL_SECONDS)
            
    except KeyboardInterrupt:
        print("\nĐã nhận lệnh dừng hệ thống an toàn.")
    finally:
        db.close()

if __name__ == '__main__':
    run_continuous_pipeline()