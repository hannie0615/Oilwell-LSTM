import pandas as pd
import numpy as np

from module import utils
from sklearn.model_selection import train_test_split

# Parameter : Tx -- Time_step, 5초 간격으로 Train 혹은 Prediction, n_feature -- Health Index 특징 개수

Tx = 5
n_feature = 4
batch_size = 64
epochs = 300

# CSV 파일에서 데이터를 읽어와 Training numpy array의 데이터로 변환 하여 모델 Training 준비
X, Y = utils.csv_to_train_data('balanced_data.csv')

# 입력 데이터를 Feature 축을 기준으로 평균과 표준편차를 계산
avg = utils.get_avg(X)
std = utils.get_std(X)

# 입력 데이터 z score normalization
X = utils.z_score_normalization(X, avg, std)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=777, stratify=Y)

# Model load
model = utils.get_model((Tx,n_feature))

# Train
history = utils.train_model(model, X_train, Y_train, X_test, Y_test, batch_size, epochs)

# 훈련된 모델 .h5 형태로 저장
utils.save_model(model, avg, std, "my_model")