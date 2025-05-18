

import os
import numpy as np
from stable_baselines3 import SAC
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from env_biosignal import BioSignalEnv

# === 1. 데이터 불러오기
X = np.load("X_eeg_features.npy")
y = np.load("Y_targets.npy")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 2. 환경 초기화
env = BioSignalEnv(X_train, y_train)

# === 3. 로그 디렉토리 설정
log_dir = "./logs/sac_biosignal/"
os.makedirs(log_dir, exist_ok=True)

# === 4. SAC 모델 학습
model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
model.learn(total_timesteps=100_000)

# === 5. 예측
preds = []
for obs in X_test:
    action, _ = model.predict(obs)
    preds.append(action)
preds = np.array(preds)

# === 6. 성능 출력
print("\n[SAC 기반 예측 성능]")
for i in range(y.shape[1]):
    mae = mean_absolute_error(y_test[:, i], preds[:, i])
    r2 = r2_score(y_test[:, i], preds[:, i])
    print(f"Target {i}: MAE = {mae:.2f}, R² = {r2:.2f}")

# === 7. 예측 결과 저장
np.save("preds.npy", preds)
np.save("y_test.npy", y_test)