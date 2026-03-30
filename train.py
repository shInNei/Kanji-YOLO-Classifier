from ultralytics import YOLO

if __name__ == '__main__':
    # Tải mô hình YOLO26 Nano Classification mới nhất!
    model = YOLO('yolo26n-cls.pt')

    # Bắt đầu Train
    # YOLO sẽ tự động lưu model tốt nhất vào: runs/classify/train/weights/best.pt
    results = model.train(
        data='YOLO_Dataset', # Thư mục vừa xả nén ở Bước 1
        epochs=10,           # Chạy thử 10 Epoch
        imgsz=64,
        batch=128,           
        device=0             
    )