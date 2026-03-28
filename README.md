#  Life Expectancy Prediction (Machine Learning - Regression)

Dự án Machine Learning dự đoán **Tuổi thọ trung bình (Life Expectancy)** dựa trên các yếu tố kinh tế – xã hội và y tế.
Bài toán thuộc loại **Supervised Learning – Regression**.

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
├── requirements.txt
├── README.md
├── data/
│   ├── raw/
│   │   └── Life Expectancy Data.csv
│   └── processed/
├── models/
├── reports/
│   ├── edu/
│   └── results/
└── src/
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

### 7.2 Model sau khi train sẽ được lưu tại:

```
models/
```

---

## 8. Đánh giá mô hình

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score

---

## 9. Báo cáo

- EDA:
```
reports/edu/report_life_expectancy.html
```

- Kết quả huấn luyện:
```
reports/results/
```

---

## 10. Dự đoán dữ liệu mới

```bash
python src/inference.py
```

---

## 👤 Tác giả

Phan Tùng  
GitHub: https://github.com/phantung205