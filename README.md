npm create vite@latest my-japanese-app -- --template react


DataLake.py - Đọc vào folder chứa mã nhị phân và đưa nó về dạng npz và dạng npy (npz là cho các file nhị phân nhiều ảnh, npy là cho ảnh đơn) - Incomplete
DataLakeManager.py - Tạo db để nhận dạng SQL - Incomplete
dataLoader.py (cũ)- Load và tạo train/test/val - code cũ rồi
prepare_yolo.py (mới)- TIỀN XỬ LÝ và tạo train/test/val cho Yolo dưới dạng ảnh - code này chưa có class, train yolo bắt buộc phải đưa về dạng ảnh cấu trúc như trong file
train.py - chạy train yolo - thông số đang hard code

Công việc cần thực hiện:
- 1 file chung để chạy liên tục, khi mình bỏ dữ liệu mới tự động chạy từ đầu tới lúc train luôn rồi chạy lại be, dữ liệu mới có thể là ảnh (ảnh size bất kì)
- cái npz và npy là định dạng cũ để chạy reset mà t bỏ xài yolo xài ảnh rồi, nên là có thể là prepare_yolo.py 1 phần tiền xử lý là bỏ qua bên DataLake, cho cái datalake nó tiền xử lý + ra ảnh luôn khỏi npy npz, 1 phần là đưa qua dataLoader là để .
- class cho dataloader chưa có merge với cái load của yolo, class model để ẩn cái yolo
- SQL chưa có test
- cái datalake chạy xong có thể có 1 số metric cho dữ liệu như histogram (optional)
- BÁO CÁO