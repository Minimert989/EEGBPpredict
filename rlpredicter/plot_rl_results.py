

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

# === 예측 결과 불러오기 ===
preds = np.load("preds.npy")
y_test = np.load("y_test.npy")

target_names = [
    "BP1_systole", "BP1_diastole", "Pulse1",
    "BP2_systole", "BP2_diastole", "Pulse2",
    "BP3_systole", "BP3_diastole",
    "Age", "Gender", "Smoking", "Drug"
]

# === 타겟별 산점도 시각화 및 저장 ===
for i, name in enumerate(target_names):
    plt.figure(figsize=(6, 5))
    plt.scatter(y_test[:, i], preds[:, i], alpha=0.6, label="Pred vs True")
    plt.plot([y_test[:, i].min(), y_test[:, i].max()],
             [y_test[:, i].min(), y_test[:, i].max()], 'r--', label="Ideal")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(f"{name} Prediction")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{name}_scatter.png")
    plt.close()