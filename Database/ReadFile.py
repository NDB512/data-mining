import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import RandomOverSampler

# Đọc dữ liệu từ file CSV
df = pd.read_csv("data_sample.csv")

# Tách đặc trưng (X) và nhãn (y)
X = df.drop("target", axis=1).values  # Tất cả các cột trừ cột target
y = df["target"].values  # Cột target

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# In phân bố lớp trong y_train để kiểm tra mất cân bằng
print("Phân bố lớp trong y_train:", np.bincount(y_train))

# Chuẩn hóa dữ liệu với StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Cân bằng dữ liệu với RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_train_balanced, y_train_balanced = ros.fit_resample(X_train_scaled, y_train)

# Kiểm tra lại phân bố lớp sau khi cân bằng
print("Phân bố lớp sau khi cân bằng:", np.bincount(y_train_balanced))

# Khởi tạo các mô hình
models = {
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Extra Trees": ExtraTreesClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000)
}

# K-Means (phân cụm, không phải phân loại)
kmeans = KMeans(n_clusters=2, random_state=42)

# Định nghĩa tham số cho GridSearchCV
param_grids = {
    "Naive Bayes": {},
    "Decision Tree": {
        "criterion": ["gini", "entropy"],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10]
    },
    "Random Forest": {
        "criterion": ["gini", "entropy"],
        "n_estimators": [50, 100],
        "max_depth": [None, 10],
        "min_samples_split": [2, 5]
    },
    "Extra Trees": {
        "criterion": ["gini", "entropy"],
        "n_estimators": [50, 100],
        "max_depth": [None, 10],
        "min_samples_split": [2, 5]
    },
    "KNN": {
        "n_neighbors": [3, 5, 7],
        "weights": ["uniform", "distance"]
    },
    "Logistic Regression": {
        "C": [0.1, 1, 10],
        "solver": ["lbfgs", "liblinear"]
    }
}

# Tinh chỉnh tham số và huấn luyện mô hình
best_models = {}
for name, model in models.items():
    print(f"\nTinh chỉnh tham số cho {name}...")
    grid = GridSearchCV(model, param_grids[name], cv=5, scoring="f1", n_jobs=-1)
    grid.fit(X_train_balanced, y_train_balanced)
    best_models[name] = grid.best_estimator_
    print(f"Tham số tốt nhất cho {name}: {grid.best_params_}")

# Đánh giá các mô hình
for name, model in best_models.items():
    print(f"\nĐánh giá {name}:")
    y_pred = model.predict(X_test_scaled)
    print(classification_report(y_test, y_pred))

# Phân cụm với K-Means
kmeans.fit(X_train_balanced)
y_pred_kmeans = kmeans.predict(X_test_scaled)
print("\nKết quả phân cụm với K-Means (không phải phân loại, chỉ để tham khảo):")
print(classification_report(y_test, y_pred_kmeans))

# Đánh giá tầm quan trọng đặc trưng với permutation_importance
for name, model in best_models.items():
    print(f"\nTầm quan trọng đặc trưng cho {name}:")
    result = permutation_importance(model, X_test_scaled, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    for i in result.importances_mean.argsort()[::-1]:
        print(f"Đặc trưng {i}: {result.importances_mean[i]:.3f} +/- {result.importances_std[i]:.3f}")