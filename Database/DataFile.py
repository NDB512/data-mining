import pandas as pd
from sklearn.datasets import make_classification

# Tạo dữ liệu mẫu: 1000 mẫu, 10 đặc trưng, 2 lớp (mất cân bằng)
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_classes=2, weights=[0.9, 0.1], random_state=42)

# Tạo DataFrame từ dữ liệu
columns = [f"feature_{i}" for i in range(X.shape[1])]  # Đặt tên cột: feature_0, feature_1, ..., feature_9
df = pd.DataFrame(X, columns=columns)
df["target"] = y  # Thêm cột mục tiêu (target)

# Lưu DataFrame thành file CSV
df.to_csv("data_sample.csv", index=False)
print("File 'data_sample.csv' đã được tạo!")