import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve

%matplotlib inline

df = pd.read_csv("Downloads/online_shoppers_intention.csv")

print(df.head())

print(df.info())

# Đếm số lượng và tính tỷ lệ phần trăm
target_counts = df['Revenue'].value_counts()

print("Số lượng lớp trong Revenue:\n", target_counts)


# Vẽ biểu đồ tròn
plt.figure(figsize=(6, 6))

plt.pie(target_counts, labels=['Không mua hàng (False)', 'Mua hàng (True)'], autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightcoral'])

plt.title('Phân bố của Cột Mục tiêu (Revenue)', fontsize=14)

plt.show()

# Tính tỷ lệ mua hàng theo Weekend

revenue_by_weekend = df.groupby('Weekend')['Revenue'].mean() * 100

# Vẽ biểu đồ thanh
plt.figure(figsize=(7, 5))

revenue_by_weekend.plot(kind='bar', color=['lightgreen', 'darkgreen'])

plt.title('Tỷ lệ Mua hàng (%) theo Cuối tuần/Ngày thường', fontsize=14)

plt.xlabel('Cuối tuần (False: Ngày thường, True: Cuối tuần)', fontsize=12)

plt.ylabel('Tỷ lệ Mua hàng (%)', fontsize=12)

plt.xticks(rotation=0)

plt.grid(axis='y', linestyle='--')

plt.show()

# Tách biến mục tiêu và đặc trưng
X = df.drop('Revenue', axis=1)

y = df['Revenue']

# Tách tập huấn luyện và kiểm tra

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Định nghĩa các loại cột
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

categorical_features = ['Month', 'VisitorType', 'OperatingSystems', 'Browser', 'Region', 'TrafficType']

binary_features = ['Weekend']

# Định nghĩa các bước tiền xử lý trong ColumnTransformer

preprocessor = ColumnTransformer(

    transformers=[
    
        ('num', StandardScaler(), numerical_features),
        
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        
        ('bin', 'passthrough', binary_features) # Weekend đã là boolean, sẽ được pipeline xử lý thành số 0/1.
        
    ],
    
    remainder='passthrough' # Giữ nguyên các cột khác (nếu có)
    
)

# Định nghĩa Pipeline

logreg_pipeline = Pipeline(steps=[

    ('preprocessor', preprocessor),
    
    ('classifier', LogisticRegression(random_state=42, solver='liblinear'))
    
])

# Huấn luyện mô hình

logreg_pipeline.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred_logreg = logreg_pipeline.predict(X_test)

y_prob_logreg = logreg_pipeline.predict_proba(X_test)[:, 1]

# 1. Tính toán các độ đo chính
print("--- Đánh giá Mô hình Logistic Regression ---")

print(f"Accuracy: {accuracy_score(y_test, y_pred_logreg):.4f}")

print(f"Precision (True): {precision_score(y_test, y_pred_logreg):.4f}")

print(f"Recall (True): {recall_score(y_test, y_pred_logreg):.4f}")

print(f"F1-Score (True): {f1_score(y_test, y_pred_logreg):.4f}")

print(f"ROC AUC Score: {roc_auc_score(y_test, y_prob_logreg):.4f}")

# 2. Vẽ Ma trận Nhầm lẫn (Confusion Matrix)

cm = confusion_matrix(y_test, y_pred_logreg)

plt.figure(figsize=(6, 6))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,

            xticklabels=['Không mua (False)', 'Mua (True)'],
            
            yticklabels=['Không mua (False)', 'Mua (True)'])

plt.title('Ma trận Nhầm lẫn - Logistic Regression', fontsize=14)

plt.xlabel('Dự đoán')

plt.ylabel('Thực tế')

plt.show()

# 3. Vẽ đường cong ROC

fpr, tpr, _ = roc_curve(y_test, y_prob_logreg)

plt.figure(figsize=(7, 5))

plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {roc_auc_score(y_test, y_prob_logreg):.4f})')

plt.plot([0, 1], [0, 1], 'r--', label='Đường Cơ sở')

plt.xlabel('False Positive Rate (FPR)')

plt.ylabel('True Positive Rate (TPR)')

plt.title('Đường cong ROC')

plt.legend()

plt.grid(linestyle='--')

plt.show()

# Định nghĩa lại Pipeline với Random Forest

rf_pipeline = Pipeline(steps=[

    ('preprocessor', preprocessor),
    
    ('classifier', RandomForestClassifier(random_state=42))
])

# Định nghĩa lưới siêu tham số để tìm kiếm

param_grid = {

    'classifier__n_estimators': [100, 200],  # Số lượng cây
    
    'classifier__max_depth': [10, 20, None], # Chiều sâu tối đa của cây
    
    'classifier__min_samples_split': [5, 10], # Số lượng mẫu tối thiểu để phân tách một nút
    
    'classifier__min_samples_leaf': [3, 5]    # Số lượng mẫu tối thiểu ở lá
}

# Thiết lập Grid Search

grid_search = GridSearchCV(

    estimator=rf_pipeline,
    
    param_grid=param_grid,
    
    scoring='f1', # Tối ưu hóa F1-Score cho lớp True
    
    cv=5, # Cross-validation 5 lần
    
    verbose=1,
    
    n_jobs=-1
)

# Tiến hành tinh chỉnh

grid_search.fit(X_train, y_train)

# In ra các siêu tham số tốt nhất

print("Các siêu tham số tốt nhất (Best Parameters):")

print(grid_search.best_params_)

# Lấy mô hình tốt nhất

best_rf_model = grid_search.best_estimator_

# Dự đoán trên tập kiểm tra

y_pred_best_rf = best_rf_model.predict(X_test)

# Tính toán F1-Score của mô hình đã tinh chỉnh

final_f1_score = f1_score(y_test, y_pred_best_rf)

print("\n--- Đánh giá Mô hình Random Forest đã Tinh chỉnh ---")

print(f"Accuracy: {accuracy_score(y_test, y_pred_best_rf):.4f}")

print(f"Precision (True): {precision_score(y_test, y_pred_best_rf):.4f}")

print(f"Recall (True): {recall_score(y_test, y_pred_best_rf):.4f}")

print(f"F1-Score (True): {final_f1_score:.4f}")

# YÊU CẦU ĐỀ BÀI: Kiểm tra kết quả

GIA_TRI_KY_VONG = 0.85

if final_f1_score >= GIA_TRI_KY_VONG:

    print(f"\n✅ ĐẠT: F1-Score ({final_f1_score:.4f}) >= Giá trị kỳ vọng ({GIA_TRI_KY_VONG})")

else:

    print(f"\n❌ CHƯA ĐẠT: F1-Score ({final_f1_score:.4f}) < Giá trị kỳ vọng ({GIA_TRI_KY_VONG})")
