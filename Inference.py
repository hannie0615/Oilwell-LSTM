import pandas as pd
import numpy as np

from module import utils
from tensorflow.keras.models import Model, load_model

model, avg, std = utils.load_trained_model("my_model")

# CSV 파일에서 데이터를 읽어와 numpy array의 데이터로 변환 하여 모델 Test 준비
X, Y = utils.csv_to_test_data('test.csv')

# 입력 데이터 z score normalization
X = utils.z_score_normalization(X, avg, std)

# Inference
precision, recall, f1, y_predict = utils.inference(model, X, Y)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Predictive_maintenance
utils.predictive_maintenance(y_predict)