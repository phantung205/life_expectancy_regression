# 🌍 Life Expectancy Prediction (Machine Learning - Regression)

Dự án Machine Learning dự đoán **Tuổi thọ trung bình (Life Expectancy)** dựa trên các yếu tố kinh tế – xã hội và y tế.

Bài toán thuộc loại **Supervised Learning – Regression**.

🔗 GitHub Repository:  
https://github.com/phantung205

---

## 🚀 Chức năng

- Phân tích dữ liệu (EDA)
- Tiền xử lý dữ liệu
- Huấn luyện mô hình hồi quy
- Lưu model (.pkl)
- Xuất báo cáo kết quả
- Dự đoán dữ liệu mới

---

## 🖥️ Yêu cầu

- Python >= 3.8
- pip

---

## 📂 Cấu trúc thư mục

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

## 📊 Dataset

🔗 Kaggle – WHO Life Expectancy Dataset:  
https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who

Sau khi tải về, đặt file vào:

```
data/raw/Life Expectancy Data.csv
```

---

### 🎯 Biến mục tiêu
- Life expectancy

---

### 🔢 Numerical features
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

### 🏷️ Nominal
- Country  

---

### 🔠 Ordinal
- Status  

---

### ❌ Cột loại bỏ
- infant deaths  
- percentage expenditure  
- GDP  
- Population  
- thinness 1-19 years  
- thinness 5-9 years  

---

## ⚙️ Cài đặt

### 1️⃣ Tạo môi trường ảo

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

### 2️⃣ Cài thư viện

```bash
pip install -r requirements.txt
```

---

## 🧠 Train model

```bash
python src/train.py
```

Model sau khi train sẽ được lưu tại:

```
models/
```

---

## 📊 Đánh giá mô hình

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score

---

## 📈 Báo cáo

- EDA:
```
reports/edu/report_life_expectancy.html
```

- Kết quả huấn luyện:
```
reports/results/
```

---

## 🔁 Dự đoán dữ liệu mới

```bash
python src/inference.py
```

---

## ⚙️ Tham số cấu hình mặc định

- random_state = 42
- test_size = 0.2

---

## 👤 Tác giả

Phan Tùng  
GitHub: https://github.com/phantung205