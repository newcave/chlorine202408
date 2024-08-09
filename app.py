import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image  # Image 모듈 임포트

# 사용자 입력 받기
st.title("잔류 염소 농도 예측 모델링 (EPA & Two-phase)")

# 이미지 불러오기
try:
    im = Image.open("AI_Lab_logo.jpg")
    st.sidebar.image(im, caption="AI Lab Logo")  # 사이드바에 이미지 표시
except FileNotFoundError:
    st.sidebar.write("Logo image not found.")  # 이미지가 없을 때의 처리

st.sidebar.header("모델 인풋 설정")

DOC = st.sidebar.slider("DOC (mg/L)", 0.0, 10.0, 5.0)
NH3 = st.sidebar.slider("surrogate var (mg/L)", 0.0, 5.0, 0.5)
Cl0 = st.sidebar.slider("현재농도 Cl0 (mg/L)", 0.0, 5.0, 1.5)
Temp = st.sidebar.slider("Temperature (°C)", 0.0, 35.0, 20.0)
max_time = st.sidebar.slider("최대예측시간 (hrs)", 1, 24, 4)

# EPA 모델에서 k1, k2 계산
k1_EPA = np.exp(-0.442 + 0.889 * np.log(DOC) + 0.345 * np.log(7.6 * NH3) - 1.082 * np.log(Cl0) + 0.192 * np.log(Cl0 / DOC))
k2_EPA = np.exp(-4.817 + 1.187 * np.log(DOC) + 0.102 * np.log(7.6 * NH3) - 0.821 * np.log(Cl0) - 0.271 * np.log(Cl0 / DOC))

# Two-phase 모델에서 A, k1, k2 계산
A_Two_phase = np.exp(0.168 - 0.148 * np.log(Cl0 / DOC) + 0.29 * np.log(1) - 0.41 * np.log(Cl0) + 0.038 * np.log(1) + 0.0554 * np.log(NH3) + 0.185 * np.log(Temp))
k1_Two_phase = np.exp(5.41 - 0.38 * np.log(Cl0 / DOC) + 0.274 * np.log(NH3) - 1.12 * np.log(Temp) + 0.05 * np.log(1) - 0.854 * np.log(7))
k2_Two_phase = np.exp(-7.13 + 0.864 * np.log(Cl0 / DOC) + 2.63 * np.log(DOC) - 2.55 * np.log(Cl0) + 0.62 * np.log(1) + 0.16 * np.log(1) + 0.48 * np.log(NH3) + 1.03 * np.log(Temp))

# 시간에 따른 농도 계산
time_range = np.linspace(0, max_time, 100)

# EPA 모델
C_EPA = np.where(time_range <= 5,
                 Cl0 * np.exp(-k1_EPA * time_range),
                 Cl0 * np.exp(5 * (k2_EPA - k1_EPA)) * np.exp(-k2_EPA * time_range))

# Two-phase 모델
C_Two_phase = Cl0 * (A_Two_phase * np.exp(-k1_Two_phase * time_range) + (1 - A_Two_phase) * np.exp(-k2_Two_phase * time_range))

# 그래프 그리기
plt.figure(figsize=(10, 6))
plt.plot(time_range, C_EPA, label='EPA Model')
plt.plot(time_range, C_Two_phase, label='Two-phase Model')
plt.xlabel('Time (hrs)')
plt.ylabel('Residual Chlorine (mg/L)')
plt.title('EPA-model-based Prediction of Clorine')
plt.legend()
plt.grid(True)
st.pyplot(plt)
