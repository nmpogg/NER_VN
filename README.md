# Vietnamese NER Demo

Ứng dụng demo nhận diện thực thể tiếng Việt sử dụng mô hình PhoBERT.

## Cài đặt

1. Cài đặt các dependencies:
```bash
pip install -r requirements.txt
```

2. Đảm bảo bạn có file mô hình `phobert_ner_model.pth` trong thư mục `models/`

## Chạy ứng dụng

1. Khởi động API server:
```bash
python api.py
```

2. Khởi động giao diện web (từ thư mục view):
```bash
npm install
npm run dev
```

3. Truy cập ứng dụng tại http://localhost:3000

## Cách sử dụng

1. Nhập văn bản tiếng Việt vào ô input
2. Nhấn nút "Phân tích" để nhận diện các thực thể
3. Kết quả sẽ hiển thị với các thực thể được đánh dấu màu sắc khác nhau

## Các loại thực thể được nhận diện

- PERSON_NAME (PER): Tên người
- LOCATION (LOC): Địa điểm
- ORGANIZATION (ORG): Tổ chức
- DATE: Ngày tháng 