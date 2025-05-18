

import mne
import matplotlib.pyplot as plt

# EEG .set 파일 경로 지정
eeg_path = "EEG_Preprocessed/sub-032301/sub-032301/sub-032301_EC.set"

# 1. .set 파일 로드
raw = mne.io.read_raw_eeglab(eeg_path, preload=True, verbose=True)

# 2. 기본 정보 출력
print(raw)
print(f"채널 수: {len(raw.ch_names)}, 샘플 수: {raw.n_times}, 총 시간: {raw.times[-1]:.2f}초")
print("채널 이름:", raw.ch_names)

# 3. 원시 데이터 가져오기
data = raw.get_data()  # shape = (n_channels, n_times)

# 4. 1번 채널 데이터 시각화 (10초 분량)
sfreq = int(raw.info['sfreq'])  # sampling frequency
plt.plot(raw.times[:sfreq * 10], data[0, :sfreq * 10])
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (uV)")
plt.title(f"EEG Signal - {raw.ch_names[0]}")
plt.grid(True)
plt.tight_layout()
plt.show()