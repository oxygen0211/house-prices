import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from scipy.stats import skew

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

print(train.describe())

all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))

#log transform the target:
train["SalePrice"] = np.log1p(train["SalePrice"])

#log transform skewed numeric features:
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

all_data = pd.get_dummies(all_data)

all_data = all_data.fillna(all_data.mean())

X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]

X_test.to_csv('data/test-processed.csv', index=False)

print(X_test.shape)

train_y = train.SalePrice

model = keras.Sequential()
model.add(keras.layers.Dense(256, input_dim = X_train.shape[1]))
model.add(keras.layers.Dropout(0.1))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(128))
model.add(keras.layers.Dense(56))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(28))
model.add(keras.layers.Dense(14))
model.add(keras.layers.Dense(8))
model.add(keras.layers.Dense(1, kernel_regularizer=keras.regularizers.l1(0.01)))
model.compile(loss = "mse", optimizer=keras.optimizers.Adam())

x_val = X_train[:200]
partial_x_train = X_train[200:]

y_val = train_y[:200]
partial_y_train = train_y[200:]

hist = model.fit(partial_x_train, partial_y_train, epochs=100, batch_size=20, validation_data = (x_val, y_val))
print(hist.history.keys())
print("train loss min: {}, val loss min: {}, train loss mean: {}, val loss mean: {}".format(np.amin(hist.history['loss']), np.amin(hist.history['val_loss']), np.mean(hist.history['loss']), np.mean(hist.history['val_loss'])))

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
model.save('model_v1.h5');
