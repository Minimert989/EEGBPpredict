conda- eeg glucose

# ✅ 프로젝트 정보: EEG 기반 생체 신호 예측 모델 + 강화학습 최적화 기록

## 1. 원본 데이터셋 준비

- 데이터 출처: MPI-LEMON (EEG 및 행동/생체 데이터 포함)
- 데이터 구성:
  - EEG 시계열 데이터 (`EEG_Preprocessed` 폴더, `.fdt`, `.set`)
  - 행동 및 생리 정보 (`Behavioural_Data_MPILMBB_LEMON`)
  - 혈압/맥박 정보 CSV (`Blood_Pressure_LEMON.csv`)
  - 메타 정보 CSV (`META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv`)

## 2. 데이터 가공
- `run.py`, `look.py` 작성
- EEG 스펙트럼 -> PSD 계산 (α, β, θ 대역별 값 추출)
- 혈압/맥박 병합: BP1, BP2, BP3 (좌측) + pulse1, pulse2
- 타겟 변수 정리:
  - 총 12개: BP1_sys/dia, pulse1, BP2_sys/dia, pulse2, BP3_sys/dia, age, gender, smoking_num, drug
- 결측치 제거, 잘못된 sample 필터링
- 최종 유효 샘플 수: 145개

## 3. 머신러닝 예측
- 모델: `MLPRegressor`
- 모든 변수에 대해 R² 및 MAE 측정
- 예측 성능 일부 항목 양호, 일부 미흡

## 4. 강화학습 기반 예측 모델 구축
- 폴더: `rlpredicter/`
- 환경: `env_biosignal.py`
- 알고리즘:
  - PPO (Stable-Baselines3)
  - SAC (Stable-Baselines3)
- 입력: X_eeg_features.npy / 타겟: Y_targets.npy
- 보상함수: `-np.sum((action - true) ** 2)` 초기버전

## 5. 강화학습 튜닝 및 평가
- TensorBoard 기록 저장됨: `logs/`
- 성능 미흡한 타겟 분석 후 가중치 튜닝 or 단일 타겟 실험 준비
- 현재 문제점: pulse2 등 특정 지표가 예측되지 않고 평균값에 수렴함 → reward 개선 예정

## 6. 사용한 패키지 목록 (conda env: eeg-glucose)
- `numpy`, `pandas`, `matplotlib`, `mne`
- `scikit-learn`, `gymnasium`, `shimmy`, `stable-baselines3`, `torch`, `tensorboard`
