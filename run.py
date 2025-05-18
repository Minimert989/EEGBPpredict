def parse_float(value):
    try:
        return float(value)
    except:
        if isinstance(value, str):
            value = value.replace('–', '-').replace('—', '-').replace('~', '-').strip().lower()

            # 나이 범위 처리 (예: "20-25", "60–65" 등 포함)
            if '-' in value:
                parts = value.split('-')
                if len(parts) == 2:
                    try:
                        return (float(parts[0]) + float(parts[1])) / 2
                    except:
                        return np.nan

            # 흡연 여부 처리 (모든 관측된 값 포함)
            if value in ['non-smoker', 'never smoker', 'no']:
                return 0.0
            if value in ['smoker', 'current smoker', 'yes', 'occasional smoker', 'smoker?']:
                return 1.0

        return np.nan
import os
import mne
import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# === 설정 ===
eeg_root = "EEG_Preprocessed"
bp_csv = "Behavioural_Data_MPILMBB_LEMON/Medical_LEMON/Blood Pressure/Blood_Pressure_LEMON.csv"
hormone_csv = "Behavioural_Data_MPILMBB_LEMON/META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv"
selected_condition = "EC"

bp_df = pd.read_csv(bp_csv, index_col=0)
bp_df = bp_df[~bp_df.index.isna()]
bp_df.index = bp_df.index.astype(str)
hormone_df = pd.read_csv(hormone_csv, index_col=0)
hormone_df.index = hormone_df.index.astype(str)

print("=== 디버그 시작 ===")
print(f"bp_df 인덱스 샘플: {bp_df.index[:5]}")
print(f"bp_df 컬럼 샘플: {bp_df.columns[:5]}")
print(f"hormone_df 컬럼 샘플: {hormone_df.columns[:5]}")
print("===")

# === 예측할 타겟 열 설정 ===
targets = [
    "BP1_left_systole",
    "BP1_left_diastole",
    "pulse1_left",
    "BP2_left_systole",
    "BP2_left_diastole",
    "pulse2_left",
    "BP3_left_systole",
    "BP3_left_diastole",
    "Age",
    "Gender_ 1=female_2=male",
    "Smoking_num_(Non-smoker=1, Occasional Smoker=2, Smoker=3)",
    "DRUG_0=negative_1=Positive"
]

# === feature 및 label 수집 ===
X, y = [], []

for subject in os.listdir(eeg_root):
    subj_path = os.path.join(eeg_root, subject)
    inner_path = os.path.join(subj_path, subject)
    if not os.path.isdir(inner_path):
        print(f"[{subject}] 내부 디렉토리 없음")
        continue
    if not os.path.isdir(subj_path): continue
    try:
        eeg_files = [f for f in os.listdir(inner_path) if f.endswith(f"{selected_condition}.set")]
        if not eeg_files:
            print(f"[{subject}] .set 파일 없음")
            continue
        eeg_file = eeg_files[0]
        eeg_path = os.path.join(inner_path, eeg_file)

        raw = mne.io.read_raw_eeglab(eeg_path, preload=True, verbose=False)
        raw.set_eeg_reference('average', projection=True)  # 평균 기준 전환
        raw.apply_proj()  # 실제 적용
        raw_data = raw.get_data() * 1e6  # Volt -> μV 단위 변환
        raw.filter(1., 40., verbose=False)
        psd = raw.compute_psd(fmin=1, fmax=40, n_fft=256, verbose=False)
        freqs = psd.freqs
        psd_data = psd.get_data()

        alpha = psd_data[:, (freqs >= 8) & (freqs <= 12)].mean()
        beta = psd_data[:, (freqs >= 13) & (freqs <= 30)].mean()
        theta = psd_data[:, (freqs >= 4) & (freqs <= 7)].mean()

        # 추가 feature 계산
        delta = psd_data[:, (freqs >= 1) & (freqs < 4)].mean()
        gamma = psd_data[:, (freqs >= 30) & (freqs <= 45)].mean()
        alpha_beta_ratio = alpha / (beta + 1e-12)
        theta_alpha_ratio = theta / (alpha + 1e-12)

        # ID 정리
        sid = subject

        age = parse_float(hormone_df.loc[sid, 'Age']) if sid in hormone_df.index and 'Age' in hormone_df.columns else np.nan
        gender = parse_float(hormone_df.loc[sid, 'Gender_ 1=female_2=male']) if sid in hormone_df.index and 'Gender_ 1=female_2=male' in hormone_df.columns else np.nan
        smoking_num = parse_float(hormone_df.loc[sid, 'Smoking_num_(Non-smoker=1, Occasional Smoker=2, Smoker=3)']) if sid in hormone_df.index and 'Smoking_num_(Non-smoker=1, Occasional Smoker=2, Smoker=3)' in hormone_df.columns else np.nan
        drug = parse_float(hormone_df.loc[sid, 'DRUG_0=negative_1=Positive']) if sid in hormone_df.index and 'DRUG_0=negative_1=Positive' in hormone_df.columns else np.nan

        # 생체지표 통합
        row_bp = bp_df.loc[[sid]] if sid in bp_df.index else pd.DataFrame()

        if row_bp.empty:
            continue

        label_row = []
        for target in targets:
            if target in row_bp.columns:
                value = row_bp[target].values[0]
            elif target in hormone_df.columns:
                value = hormone_df.loc[sid, target] if sid in hormone_df.index else np.nan
            else:
                value = np.nan
            try:
                label_row.append(parse_float(value))
            except:
                label_row.append(np.nan)

        print(f"[{sid}] α={alpha:.8e}, β={beta:.8e}, θ={theta:.8e} → y={label_row}")
        print(f"psd_data.shape = {psd_data.shape}, mean PSD = {psd_data.mean():.8e}")

        if np.any(pd.isnull(label_row)): continue

        feature_vector = [
            alpha, beta, theta,
            delta, gamma,
            alpha_beta_ratio, theta_alpha_ratio,
            age, gender, smoking_num, drug
        ]
        feature_vector = [parse_float(f) if not pd.isnull(f) else np.nan for f in feature_vector]

        if np.any(pd.isnull(label_row)) or np.any(pd.isnull(feature_vector)):
            print(f"[{sid}] 누락된 값 있음 → feature: {feature_vector}, label: {label_row}")
            continue
        X.append(feature_vector)
        y.append(label_row)

    except Exception as e:
        print(f"[{subject}] 에러 발생:", e)
        continue

if len(X) == 0 or len(y) == 0:
    raise ValueError("사용 가능한 EEG + 생체지표 샘플이 0개입니다. 경로, ID 매칭, 열 이름 등을 확인하세요.")
else:
    print(f"총 유효 샘플 수: {len(X)}")

# === 모델 훈련 및 평가 ===
X, y = np.array(X), np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# === 평가 출력 ===
mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
r2 = r2_score(y_test, y_pred, multioutput='raw_values')

print("\n[모델 평가 결과]")
for i, target in enumerate(targets):
    print(f"{target}: MAE = {mae[i]:.2f}, R^2 = {r2[i]:.2f}")

# === 예측 시각화 ===
import matplotlib.pyplot as plt

for i, target in enumerate(targets):
    plt.figure(figsize=(6, 5))
    plt.scatter(y_test[:, i], y_pred[:, i], color="orange", alpha=0.6, label="Predicted vs True")
    plt.plot([y_test[:, i].min(), y_test[:, i].max()],
             [y_test[:, i].min(), y_test[:, i].max()], 'r--', label="Ideal")
    plt.xlabel("True Value")
    plt.ylabel("Predicted Value")
    plt.title(f"{target}: Prediction Scatter Plot")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === 저장 ===
np.save("X_eeg_features.npy", X)
np.save("Y_targets.npy", y)
print("✅ 'X_eeg_features.npy'와 'Y_targets.npy' 저장 완료.")