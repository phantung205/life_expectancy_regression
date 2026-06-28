#  Life Expectancy Prediction (Machine Learning - Regression)

Dự án Machine Learning dự đoán **Tuổi thọ trung bình (Life Expectancy)** dựa trên các yếu tố kinh tế – xã hội và y tế.
Bài toán thuộc loại **Supervised Learning – Regression** , giao diện Wed sự dụng Flask.

---

## 1. Chức năng

- Phân tích dữ liệu (EDA)
- Tiền xử lý dữ liệu
- Huấn luyện mô hình hồi quy
- Lưu model (.pkl)
- Xuất báo cáo kết quả
- Dự đoán dữ liệu mới

---

## 2. Yêu cầu

- Python >= 3.8
- pip

---

## 3. Cấu trúc thư mục

```text
life_expectancy/
├── data
│   ├── processed
│   └── raw
├── models
├── reports
│   ├── edu
│   ├── parameter
│   └── results
├── results
├── src
├── templates
├── uploads
└── validation
```

---

## 4. Dataset

### 4.1 Tải dữ liệu

- Kaggle – WHO Life Expectancy Dataset:  
 https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who

Sau khi tải về, đặt file vào:
```
data/raw/Life Expectancy Data.csv
```

---

### 4.2 Biến mục tiêu
- Life expectancy

---

### 4.3 Numerical features
- Year  
- Adult Mortality  
- Alcohol  
- Hepatitis B  
- Measles  
- BMI  
- under-five deaths  
- Polio  
- Total expenditure  
- Diphtheria  
- HIV/AIDS  
- Income composition of resources  
- Schooling  

---

### 4.4 Nominal
- Country  

---

### 4.5 Ordinal
- Status  

---

### 4.6 Cột loại bỏ
- infant deaths  
- percentage expenditure  
- GDP  
- Population  
- thinness 1-19 years  
- thinness 5-9 years  

---

## 5. Cài đặt

### 5.1 Tạo môi trường ảo

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Linux / macOS:

```bash
source venv/bin/activate
```

---

### 5.2 Cài thư viện

```bash
pip install -r requirements.txt
```

---

## 6. chỉnh cấu hình tham số mặc định
```text
config.py
```

---

## 7. Train model

```bash
# RandomForestRegressor
python -m src.train -m RandomForestRegressor

# LinearRegression
python -m src.train -m LinearRegression

# Ridge
python -m src.train -m Ridge
```

### 7.1 Model sau khi train sẽ được lưu tại:

```
models/
```

---

## 8. chạy docker container 
### 8.1 build docker image
```bash
docker build -t life_expectancy .
```
### 8.2 vào trong container train 
```bash
docker run -it  --rm -v ${PWD}/data/raw:/life_expectancy/data/raw  -v ${PWD}/models:/life_expectancy/models -v ${PWD}/uploads:/life_expectancy/uploads -v ${PWD}/results:/life_expectancy/results  life_expectancy bash
```
- sau khi vào trong docker chạy các lệnh train model như phần 7

### 8.3 nếu đã có checkpoint thì chạy app luôn trong container
```bash
docker run  --rm -p 5000:5000  -v ${PWD}/models:/life_expectancy/models -v ${PWD}/uploads:/life_expectancy/uploads -v ${PWD}/results:/life_expectancy/results  life_expectancy 
```

---

## 9. Đánh giá mô hình

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score

---

## 10. Báo cáo

- EDA:
```
reports/edu/report_life_expectancy.html
```

- Kết quả huấn luyện:
```
reports/results/
```

---

## 11. Chạy ứng dụng wed, test

```bash
python app.py
```
Mặc định ứng dụng chạy tại:

http://127.0.0.1:5000

---

## 👤 Tác giả

Phan Tùng  
GitHub: https://github.com/phantung205