import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, Embedding, Add, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import glorot_uniform
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

tf.random.set_seed(2)

esm_tr = np.load('data/esm_train_t33_650M_1024.npy')
esm_ts = np.load('data/esm_test_t33_650M_1024.npy')

esm_str_lab_tr = pd.read_csv('data/esol_tr_str_rem.csv')
esm_str_lab_ts = pd.read_csv('data/esol_ts_str_rem.csv')

esm_str_tr = [eval(i) for i in esm_str_lab_tr.iloc[:, 2]]
esm_str_ts = [eval(i) for i in esm_str_lab_ts.iloc[:, 2]]

esm_str_pad_tr = pad_sequences(esm_str_tr, maxlen=1024, padding='post')
esm_str_pad_ts = pad_sequences(esm_str_ts, maxlen=1024, padding='post')

esm_lab_tr = np.array(esm_str_lab_tr.iloc[:, 1])
esm_lab_ts = np.array(esm_str_lab_ts.iloc[:, 1])

kf = KFold(n_splits=5, random_state=0, shuffle=True)
train_indices, val_indices = [], []

for train_index, val_index in kf.split(esm_lab_tr):
    train_indices.append(train_index)
    val_indices.append(val_index)

early_stopper = EarlyStopping(monitor="val_loss", mode='min', min_delta=0.001, patience=20, verbose=1, restore_best_weights=True)

def create_model():
    inp_str = Input(shape=(1024,))
    emb = Embedding(4, 1280, mask_zero=True)(inp_str)
    emb = tf.math.reduce_mean(emb, axis=0)
    
    inp_seq = Input(shape=(1280,))
    add = Add()([emb, inp_seq])
    mean = tf.math.reduce_mean(inp_seq, axis=1)
    
    lstm = LSTM(500, use_bias=False)(add)
    
    dense1 = Dense(512, activation='relu', use_bias=False,
                   kernel_initializer=glorot_uniform(seed=0),
                   kernel_regularizer=l2(0.001))(inp_seq)
    dense1 = Dropout(0.7)(dense1)
    
    dense2 = Dense(512, activation='relu', use_bias=False,
                   kernel_initializer=glorot_uniform(seed=0),
                   kernel_regularizer=l2(0.001))(dense1)
    dense2 = Dropout(0.7)(dense2)
    
    out = Dense(1, activation='sigmoid')(dense2)
    
    model = Model(inputs=inp_seq, outputs=out)
    model.compile(loss="mean_squared_error", optimizer=Adam(learning_rate=0.001))
    
    return model

def train_model():
    all_predictions = []
    
    for i in range(5):
        print(f'Fold {i+1}')
        
        train_emb, val_emb = esm_tr_mean[train_indices[i]], esm_tr_mean[val_indices[i]]
        train_labels, val_labels = esm_lab_tr[train_indices[i]], esm_lab_tr[val_indices[i]]
        
        model = create_model()
        model.fit(train_emb, train_labels, batch_size=64, epochs=500,
                  validation_split=0.2, verbose=1, callbacks=[early_stopper])
        
        val_predictions = np.squeeze(model.predict(val_emb))
        print(f'Fold {i+1}: {roc_auc_score(val_labels, val_predictions)}')
        
        test_predictions = np.squeeze(model.predict(esm_ts_mean))
        all_predictions.append(test_predictions)
    
    final_predictions = np.mean(all_predictions, axis=0)
    print('Final AUC: ', roc_auc_score(esm_lab_ts, final_predictions))

train_model()
