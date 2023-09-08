import pandas as pd 
import numpy as np
import os
import csv
from scipy import stats
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense, Input, TimeDistributed, LSTM, BatchNormalization, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

def csv_to_train_data(filename, segment_sec = 5):
    """
    해양 유전 센서에서 받아진 데이터를 5초 기준으로 하나의 훈련 데이터로 만드는 함수,
    연속된 5초의 데이터가 같은 라벨(정상 혹은 이상진동)일 때, 훈련 데이터로 생성
    
    Argument:
    filename -- 저장된 csv 파일 이름
    
    Returns:
    X -- 5초의 csv train 데이터
    Y -- X와 매핑된 Label
    """

    X = []
    Y = []
    p_pdg = []
    p_tpt = []
    t_tpt = []
    p_mon_ckp = []
    t_jus_ckp = []
    p_jus_ckgl = []
    label = []
    
    with open(filename, 'r') as file:
        #df = pd.read_csv(file)
        csv_reader = csv.DictReader(file)    
        for csv_data in csv_reader:
            for key, val in csv_data.items():
                if key == "P-PDG":
                    p_pdg.append(float(val))
                elif key == "P-TPT":
                    p_tpt.append(float(val))
                elif key == "T-TPT":
                    t_tpt.append(float(val))
                elif key == "P-MON-CKP":
                    p_mon_ckp.append(float(val))
                elif key == "T-JUS-CKP":
                    t_jus_ckp.append(float(val))
                elif key == "P-JUS-CKGL":
                    p_jus_ckgl.append(float(val))
                elif key == "class":
                    if val == '0':
                        label.append(0)
                    elif val == '1':
                        label.append(1)
                    elif val == '101':
                        label.append(1)
                    elif val == '2':
                        label.append(2)
                    elif val == '102':
                        label.append(2)
                    elif val == '3':
                        label.append(3)
                    elif val == '4':
                        label.append(4)
                    elif val == '5':
                        label.append(5)
                    elif val == '105':
                        label.append(5)
                    elif val == '6':
                        label.append(6)
                    elif val == '106':
                        label.append(6)
                    elif val == '7':
                        label.append(7)
                    elif val == '107':
                        label.append(7)
                    elif val == '8':
                        label.append(8)
                    elif val == '108':
                        label.append(8)
        
    ### Data concat ###
    # Simulated = P-PDG, P-TPT, P-MON-CKP, T-JUS-CKP
    # Real = P-PDG, P-TPT, T-TPT, P-MON-CKP, T-JUS-CKP, P-JUS-CKGL
    
    p_pdg                = np.expand_dims(p_pdg, axis = 1)
    p_tpt                = np.expand_dims(p_tpt, axis = 1)
    #t_tpt                = np.expand_dims(t_tpt, axis = 1)
    p_mon_ckp            = np.expand_dims(p_mon_ckp, axis = 1)
    t_jus_ckp            = np.expand_dims(t_jus_ckp, axis = 1)
    #p_jus_ckgl           = np.expand_dims(p_jus_ckgl, axis = 1)

    
    # Label
    label              = np.expand_dims(label, axis = 1)

    # Data Concat
    concat_data        = np.concatenate((p_pdg, p_tpt, p_mon_ckp, t_jus_ckp), axis = 1)
       
    # segment_sec를 간격으로 하나의 데이터 세트로 생성
    X = np.array([ concat_data[i:i+5] for i in range(0, len(concat_data) - segment_sec, segment_sec) ])
        
    # segment_sec를 간격으로  하나의 Label을 생성.
    Y = np.expand_dims([stats.mode(label[i:i+segment_sec])[0][0] for i in range(0, len(concat_data) - segment_sec, segment_sec)], axis=1)
    
    X = np.array(X)
    Y = np.array(Y)
        
    # X와 Y의 위치를 유지하며 random으로 섞기
    idxs = np.arange(X.shape[0])
    np.random.shuffle(idxs)

    Y = Y[idxs]
    X = X[idxs]

    Y = tf.keras.utils.to_categorical(Y)
    Y = Y.reshape(-1,9)
    return X, Y

def csv_to_test_data(filename, segment_sec = 5):
    """
    진동센서에서 받아진 HealthIndex 데이터를 5초 기준으로 하나의 테스트 데이터로 만드는 함수
    
    Argument:
    filename -- 저장된 csv 파일 이름
    
    Returns:
    X -- 5초의 csv test 데이터
    """
    
    X_ = []
    Y_ = []
    p_pdg_ = []
    p_tpt_ = []
    t_tpt_ = []
    p_mon_ckp_ = []
    t_jus_ckp_ = []
    p_jus_ckgl_ = []
    concat_data_ = []
    label_ = []
    
    with open('test.csv', 'r') as file:
        #df = pd.read_csv(file)
        csv_reader = csv.DictReader(file)    
        for csv_data in csv_reader:
            for key, val in csv_data.items():
                if key == "P-PDG":
                    p_pdg_.append(float(val))
                elif key == "P-TPT":
                    p_tpt_.append(float(val))
                elif key == "T-TPT":
                    t_tpt_.append(float(val))
                elif key == "P-MON-CKP":
                    p_mon_ckp_.append(float(val))
                elif key == "T-JUS-CKP":
                    t_jus_ckp_.append(float(val))
                elif key == "P-JUS-CKGL":
                    p_jus_ckgl_.append(float(val))
                elif key == "class":
                    if val == '0':
                        label_.append(0)
                    elif val == '1':
                        label_.append(1)
                    elif val == '101':
                        label_.append(1)
                    elif val == '2':
                        label_.append(2)
                    elif val == '102':
                        label_.append(2)
                    elif val == '3':
                        label_.append(3)
                    elif val == '4':
                        label_.append(4)
                    elif val == '5':
                        label_.append(5)
                    elif val == '105':
                        label_.append(5)
                    elif val == '6':
                        label_.append(6)
                    elif val == '106':
                        label_.append(6)
                    elif val == '7':
                        label_.append(7)
                    elif val == '107':
                        label_.append(7)
                    elif val == '8':
                        label_.append(8)
                    elif val == '108':
                        label_.append(8)
    
    ### Data concat ###
    # Simulated = P-PDG, P-TPT, P-MON-CKP, T-JUS-CKP
    # Real = P-PDG, P-TPT, T-TPT, P-MON-CKP, T-JUS-CKP, P-JUS-CKGL
    
    p_pdg_                = np.expand_dims(p_pdg_, axis = 1)
    p_tpt_                = np.expand_dims(p_tpt_, axis = 1)
    #t_tpt                = np.expand_dims(t_tpt, axis = 1)
    p_mon_ckp_            = np.expand_dims(p_mon_ckp_, axis = 1)
    t_jus_ckp_            = np.expand_dims(t_jus_ckp_, axis = 1)
    #p_jus_ckgl           = np.expand_dims(p_jus_ckgl, axis = 1)
       
    # Real Data Concat
    #concat_data        = np.concatenate((p_pdg, p_tpt, t_tpt, p_mon_ckp, t_jus_ckp, p_jus_ckgl), axis = 1)
    
    # Simulated Data Concat
    concat_data_        = np.concatenate((p_pdg_, p_tpt_, p_mon_ckp_, t_jus_ckp_), axis = 1)
    
    # Label
    label_              = np.expand_dims(label_, axis = 1)
    
    # segment_sec를 간격으로 하나의 데이터 세트로 생성
    X_ = np.array([ concat_data_[i:i+5] for i in range(0, len(concat_data_) - segment_sec, segment_sec) ])

    # segment_sec를 간격으로 하나의 Label을 생성.
    Y_ = np.expand_dims([stats.mode(label_[i:i+segment_sec])[0][0] for i in range(0, len(concat_data_) - segment_sec, segment_sec)], axis=1)

    X_ = np.array(X_)
    Y_ = np.array(Y_)

    Y_ = tf.keras.utils.to_categorical(Y_) # one-hot encoding
    Y_ = Y_.reshape(-1,9)
    
    return X_, Y_

def get_avg(X):
    """
    입력 데이터의 Feature를 기준으로 평균을 계산하는 함수
    
    Argument:
    X -- 입력 데이터
    
    returns: 
    avg -- feature 축을 기준으로 한 평균 벡터
    """
    
    shape = X.shape
    reshape_X = np.reshape(X,(shape[0] * shape[1] ,shape[2]))
    
    avg = np.mean(reshape_X, axis=0)
    
    return avg

def get_std(X):
    """
    입력 데이터의 Feature를 기준으로 표준편차를 계산하는 함수
    
    Argument:
    X -- 입력 데이터
    
    returns: 
    std -- feature 축을 기준으로 한 표준편차 벡터
    """
    
    shape = X.shape
    reshape_X = np.reshape(X,(shape[0] * shape[1] ,shape[2]))
    
    std = np.std(reshape_X, axis=0)
    
    return std

def z_score_normalization(X, avg, std):
    """
    입력 데이터를 z score normalization을 수행하는 함수 
    
    * 데이터 분포에 따른 표준정규분포 정규화
    
    # z score normalization
    X = (X - mu) / sigma
    
    Argument:
    X -- 입력 데이터
    avg -- 입력 데이터의 feature 축을 기준으로 한 평균
    std -- 입력 데이터의 feature 축을 기준으로 한 표준 편차
    
    returns:
    X -- z score normalization이 적용된 입력 데이터
    """
    X = (X - avg) / std
    
    return X
    
def get_model(input_shape):
    """
    Keras의 model graph를 생성해주는 함수
    
    Argument:
    input_shape -- 모델 입력 데이터의 shape
    
    Returns:
    model -- Keras model instance  
    """
    
    lstm_input = Input(shape = (input_shape))
    lstm = TimeDistributed(Dense(20, activation = "relu"))(lstm_input)
    lstm = BatchNormalization()(lstm)
    lstm = LSTM(30, return_sequences = True)(lstm)
    lstm = Dropout(0.5)(lstm)
    lstm = LSTM(20)(lstm)
    lstm = Dropout(0.5)(lstm)
    lstm = Dense(10, activation = "relu")(lstm) 
    lstm = Dense(9, activation = "softmax")(lstm)
        
    model = Model(inputs = lstm_input, outputs = lstm)
        
    opt = Adam(learning_rate=1e-5, beta_1=0.9, beta_2=0.999)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])
    
    return model

def train_model(model, X, Y, X_test, Y_test, batch_size, epochs):   
    """
    model 훈련함수 함수
    
    Argument:
    X -- (N, Tx, n_feature) shape 의 입력 데이터
    Y -- X에 대한 Label 데이터
    batch_size -- 훈련 Batch size
    epochs -- 훈련 반복 횟수
    
    Returns:
    history -- 훈련 history 반환
    """
    
    history = model.fit(X, Y, batch_size = batch_size, epochs=epochs, validation_data=(X_test, Y_test))
    
    return history
    
def save_model(model, avg, std, filename = "my_model"):
    """
    model 저장 함수
    
    Argument:
    model -- 저장할 keras 모델
    avg -- 입력데이터의 feature축을 기준으로 한 평균 벡터
    std -- 입력데이터의 feature축을 기준으로 한 표준편차 벡터
    
    filename -- 저장할 파일 이름
    
    Returns:
    None
    """
    
    # 훈련된 모델과 파라미터를 저장할 폴더 생성
    os.mkdir(filename)
    
    # inference 시 z score normalization에 사용될 평균과 표준편차 저장
    np.save(filename + "/" + filename + "_avg.npy", avg)
    np.save(filename + "/" + filename + "_std.npy", std)
    
    model.save(filename + "/" + filename + ".h5")
    
def load_trained_model(filename = "my_model"):
    """
    훈련된 model을 불러오는 함수
    
    Argument:
    filename -- 저장할 파일 이름
    
    Returns:
    model -- 훈련된 keras 모델
    avg -- 입력데이터의 feature축을 기준으로 한 평균 벡터
    std -- 입력데이터의 feature축을 기준으로 한 표준편차 벡터
    """
    
    model = load_model(filename + "/" + filename + ".h5")
    
    avg = np.load(filename + "/" + filename + "_avg.npy")
    std = np.load(filename + "/" + filename + "_std.npy")
    
    return model, avg, std

def inference(model, X, Y):
    """
    훈련된 model을 사용하여 입력 데이터 X의 precision, recall, f1을 추정하는 함수
    
    Argument:
    model -- 훈련된 keras 모델
    X -- 입력 데이터
    
    Returns:
    """
    
    y_predict = model.predict(X)
    y_predict_binary = (y_predict > 0.5).astype(int)
    
    precision = precision_score(Y, y_predict_binary, average='samples')
    recall = recall_score(Y, y_predict_binary, average='samples')
    f1 = f1_score(Y, y_predict_binary, average='samples')
    
    # print(f"Precision: {precision:.2f}")
    # print(f"Recall: {recall:.2f}")
    # print(f"F1 Score: {f1:.2f}")
    
    return precision, recall, f1, y_predict

def get_train_history(history):
    epochs = len(history.history['loss'])
    texts = []
    
    for epoch in range(epochs):
        text = "Epoch %d/%d - loss: %f - accuracy: %f" % (epoch, epochs, history.history['loss'][epoch], history.history['accuracy'][epoch])
        texts.append(text)
        
    return texts

def predictive_maintenance(y_predict):
    for i in range(len(y_predict)):
        label = y_predict[i].argmax()  # Now, 'label' is an integer and not a list
        confidence = y_predict[i][label]  # 'label' can be used directly as it's an integer
        event = 'Normal'  # Moved inside the loop
        event_location = 'None'
        if label == 0:
            event = "Normal"
            event_info = "None"
        elif label == 1:
            event = "Abrupt_BSW_Increase"
            event_info = "석유 가공 라인에서 유체 구성 변동 Fault 발생"
        elif label == 2:
            event = "Spurious_DHSV_Closure"
            event_info = "생산튜브 내 안전튜브 Falut 발생"
        elif label == 3:
            event = "Severe_Slugging"
            event_info = "파이프라인 내의 유체 유동 불안정 상태 발생"
        elif label == 4:
            event = "Flow_Instability"
            event_info = "유체 유동의 불안정 상태 발생, 원인 파악 바람" 
        elif label == 5:
            event = "Rapid_Productivity_Loss"
            event_info = "생산량의 급속한 감소 발생, 작업 프로세스 점검 요함"
        elif label == 6:
            event = "Quick_PCK_Restriction"
            event_info = "생산량 제한 Fault 발생, PCK 점검 요함"
        elif label == 7:
            event = "PCK_Scaling"
            event_info = "PCK Value의 이상 발생, PCK 조절 요함"
        elif label == 8:
            event = "Hydrates_in_Production_Lines"
            event_info = "생산 라인에서의 수화물 발생, 유체 유동 방해 및 유동성 저하 가능성 높음"

        print("\n")
        print("> 현재 시각 기준 5초 동안의 발생한 Event : " + event)
        print("> Event 예측 신뢰도 : "+ format(confidence*100, ".2f") + "%")
        print("> Event Fault Info. : " + event_info)
        print("\n")









    