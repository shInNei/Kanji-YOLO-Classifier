npm create vite@latest my-japanese-app -- --template react


DataLake.py - Đọc vào folder chứa mã nhị phân và đưa nó về dạng npz và dạng npy (npz là cho các file nhị phân nhiều ảnh, npy là cho ảnh đơn) - Incomplete
DataLakeManager.py - Tạo db để nhận dạng SQL - Incomplete
dataLoader.py (cũ)- Load và tạo train/test/val - code cũ rồi
prepare_yolo.py (mới)- TIỀN XỬ LÝ và tạo train/test/val cho Yolo dưới dạng ảnh - code này chưa có class, train yolo bắt buộc phải đưa về dạng ảnh cấu trúc như trong file
train.py - chạy train yolo - thông số đang hard code

Công việc cần thực hiện:
- 1 file chung để chạy liên tục, khi mình bỏ dữ liệu mới tự động chạy từ đầu tới lúc train luôn rồi chạy lại be, dữ liệu mới có thể là ảnh (ảnh size bất kì)
- Note: cái npy và npz là nó dùng để lưu trữ trong datalake (nhẹ hơn ảnh, gọn), còn sau đó cái dataloader nó sẽ lấy thông tin từ datalake nó tiền xử lý và chia train test val dưới dạng ảnh cho yolo, khả năng là train xong yolo nên xóa cái thư mục ảnh đi.
- class cho dataloader chưa có merge với cái load của yolo, class model để ẩn cái yolo
- SQL chưa có test
- cái datalake chạy xong có thể có 1 số metric cho dữ liệu như histogram (optional)
- BÁO CÁO