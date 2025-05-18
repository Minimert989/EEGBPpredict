

import numpy as np
from stable_baselines3 import PPO
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from env_biosignal import BioSignalEnv

# === 1. 데이터 불러오기 (X: EEG feature, y: 생체지표)
X = np.load("X_eeg_features.npy")
y = np.load("Y_targets.npy")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 2. 강화학습 환경 생성
env = BioSignalEnv(X_train, y_train)

# === 3. PPO 에이전트 초기화 및 학습
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)

# === 4. 평가
preds = []
for obs in X_test:
    action, _ = model.predict(obs)
    preds.append(action)
preds = np.array(preds)

# === 5. 성능 평가 출력
print("\n[강화학습 예측 결과 평가]")
for i in range(y.shape[1]):
    mae = mean_absolute_error(y_test[:, i], preds[:, i])
    r2 = r2_score(y_test[:, i], preds[:, i])
    print(f"Target {i}: MAE = {mae:.2f}, R² = {r2:.2f}")