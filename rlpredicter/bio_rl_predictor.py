


import gym
from gym import spaces
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from stable_baselines3 import PPO

# === 1. 데이터 로딩 (예시: numpy 로딩 가정, 필요 시 변경)
X = np.load("X_eeg_features.npy")   # shape = (samples, 11)
y = np.load("Y_targets.npy")        # shape = (samples, 12)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 2. 강화학습 환경 정의
class BioSignalEnv(gym.Env):
    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y
        self.index = 0
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(X.shape[1],), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=200, shape=(y.shape[1],), dtype=np.float32)

    def reset(self):
        self.index = np.random.randint(len(self.X))
        return self.X[self.index]

    def step(self, action):
        true = self.y[self.index]
        reward = -np.mean(np.abs(action - true))  # MAE 기반 reward
        done = True
        return self.X[self.index], reward, done, {}

# === 3. PPO 학습
env = BioSignalEnv(X_train, y_train)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)

# === 4. 평가
preds = []
for i in range(len(X_test)):
    obs = X_test[i]
    action, _ = model.predict(obs)
    preds.append(action)
preds = np.array(preds)

# === 5. 성능 출력
print("\n[RL 기반 예측 평가 결과]")
for i in range(y.shape[1]):
    mae = mean_absolute_error(y_test[:, i], preds[:, i])
    r2 = r2_score(y_test[:, i], preds[:, i])
    print(f"Target {i}: MAE = {mae:.2f}, R² = {r2:.2f}")