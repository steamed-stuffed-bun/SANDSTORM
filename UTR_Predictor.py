# Load packages
import sys
#the path you must add your GARDN-SANDSTORM model src folder absolute path
sys.path.insert(1, "/home/nanoribo/SANDSTORM/GARDN-SANDSTORM-main/src")
import tensorflow as tf
import keras as tfk

tfkl = tf.keras.layers
import matplotlib.pyplot as plt
import util
import GA_util
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
print('loaded modules')

from GA_util import create_joint_model
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
from tensorflow.keras.optimizers import Adam
# Set model paramater
epoch_num = 30
iterations =1
latent_dim = 64
learning_rate = 0.001
batch_size = 64
# load input data
data = pd.read_csv('/home/nanoribo/NGSprocessing/CELL_IVT/feature/mRNA_halflife_highquaset_withfeatures.csv')
mask  = ~data['sequence'].str.upper().str.contains('N', na=False)
data = data[mask].copy()


seq_len = len(data['sequence'].iloc[0])
ppm_len = len(data['sequence'].iloc[0])

y = data['half_life'].values  # y is the label corresponding to the sequence data.



from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import StandardScaler
stand = StandardScaler()
stand.fit(y.reshape(-1,1))
y_transformed = stand.transform(y.reshape(-1,1))
est =  KBinsDiscretizer(n_bins=10,encode='ordinal') #Quantile transform
encoded_vals =est.fit_transform(y.reshape(-1,1))
print(encoded_vals.shape)

joint_mse_save = []
joint_r2_save = []
joint_spearman_save = []


utrs = util.one_hot_encode(data[['sequence']])  # Use one hot coding process data
utrs_ppms = GA_util.prototype_ppms_fast(utrs)  # Convert sequence data into a secondary structure matrix.

indices = np.arange(0, utrs.shape[0])


for iteration in range(4):
    utr_train, utr_test, y_train, y_test, indices_train, indices_test = train_test_split(utrs,y_transformed,
                                                                                         indices,
                                                                                         test_size=0.20,
                                                                                         stratify=encoded_vals)
    ppm_train = utrs_ppms[indices_train, :, :]
    ppm_test = utrs_ppms[indices_test, :, :]

    optimizer = Adam(learning_rate=learning_rate)
    joint_model = GA_util.create_SANDSTORM(seq_len=seq_len,
                                           ppm_len=ppm_len,
                                           latent_dim=latent_dim,
                                           internal_activation='relu')

    joint_model.compile(optimizer=optimizer, loss='mse')

    # model train
    hist = joint_model.fit(
        [utr_train, ppm_train], y_train, batch_size=batch_size, validation_data=[[utr_test, ppm_test], y_test],
        epochs=epoch_num)

    # model metrics caculate
    # MSE caculate
    mse = np.min(hist.history['val_loss'])
    joint_mse_save.append(mse)

    joint_predictions = joint_model([utr_test, ppm_test])

    # joint_r2 = r2_score(y_test,joint_predictions)
    # joint_r2_save.append(joint_r2)

    # R2 caculate
    y_pred_np = joint_predictions.numpy().squeeze()
    y_true_np = y_test.squeeze()
    joint_r2 = r2_score(y_true_np, y_pred_np)
    joint_r2_save.append(joint_r2)

    # spearman
    joint_spearman = spearmanr(y_true_np, y_pred_np)[0]
    joint_spearman_save.append(joint_spearman)

data = {
    "MSE": joint_mse_save,
    "R2": joint_r2_save,
    'Spearman': joint_spearman_save}
df = pd.DataFrame(data)
print("The model is complete!")
print("Result is :\n")
print(joint_spearman_save)
df.to_csv("./tmp_result.csv", index=False, encoding='utf-8')
