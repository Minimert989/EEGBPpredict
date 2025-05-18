import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from env_biosignal import BioSignalEnv

# === 1. 데이터 불러오기
X = np.load("X_eeg_features.npy")
y = np.load("Y_targets.npy")

# === 2. 정규화
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 3. 강화학습 환경 정의
env = BioSignalEnv(X_train, y_train)

# === 4. 로그 디렉토리 설정
log_dir = "./logs/ppo_biosignal/"
os.makedirs(log_dir, exist_ok=True)

# === 5. 모델 설정 및 학습
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
model.learn(total_timesteps=100_000)

# === 6. 평가
preds = []
for obs in X_test:
    action, _ = model.predict(obs)
    preds.append(action)
preds = np.array(preds)

# === 7. 출력
print("\n[TensorBoard 연동 PPO 예측 성능]")
for i in range(y.shape[1]):
    mae = mean_absolute_error(y_test[:, i], preds[:, i])
    r2 = r2_score(y_test[:, i], preds[:, i])
    print(f"Target {i}: MAE = {mae:.2f}, R² = {r2:.2f}")

# === 8. 저장
np.save("preds.npy", preds)
np.save("y_test.npy", y_test)