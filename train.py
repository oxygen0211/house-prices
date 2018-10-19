import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from tensorflow import keras
from scipy.stats import skew

def qual_conv(q):
    if q=='Po':
        return 0
    if q=='Fa':
        return 1
    if q=='TA':
        return 2
    if q=='Gd':
        return 3
    if q=='Ex':
        return 4
    return 5

def slope_conv(s):
    if s=='Gtl':
        return 0
    if s=='Mod':
        return 1
    if s=='Sev':
        return 2
    return 3

def garage_finish_conv(f):
    if f=='Unf':
        return 0
    if f=='RFn':
        return 1
    if f=='Fin':
        return 2
    return 3

def bsmt_exposure_conv(e):
    if e=='No':
        return 0
    if e=='Mn':
        return 1
    if e=='Av':
        return 2
    if e=='Gd':
        return 3
    return 4

def bsmt_fin_type_conf(t):
    if t=='Unf':
        return 0
    if t=='LwQ':
        return 1
    if t=='Rec':
        return 2
    if t=='BLQ':
        return 3
    if t=='ALQ':
        return 4
    if t=='GLQ':
        return 5
    return 6

def functional_conv(f):
    if f=='Typ':
        return 0
    if f=='Min1':
        return 1
    if f=='Min2':
        return 2
    if f=='Mod':
        return 3
    if f=='Maj1':
        return 4
    if f=='Maj2':
        return 5
    if f=='Sev':
        return 6
    if f=='Sal':
        return 0

def house_style_conv(s):
    if s=='1Story':
        return 0
    if s=='1.5Fin':
        return 1
    if s=='1.5Unf':
        return 2
    if s=='2Story':
        return 3
    if s=='2.5Fin':
        return 4
    if s=='2.5Unf':
        return 5
    if s=='SFoyer':
        return 6
    if s=='SLvl':
        return 7
    return 8

def foundation_conv(f):
    if f=='BrkTil':
        return 0
    if f=='CBlock':
        return 1
    if f=='PConc':
        return 2
    if f=='Slab':
        return 3
    if f=='Stone':
        return 4
    if f=='Wood':
        return 5
    return 6

def mas_ver_type_conv(t):
    if t=='BrkCmn':
        return 0
    if t=='BrkFace':
        return 1
    if t=='CBlock':
        return 2
    if t=='None':
        return 3
    if t=='Stone':
        return 4
    return 5

def addlogs(res, ls):
    m = res.shape[1]
    for l in ls:
        res = res.assign(newcol=pd.Series(np.log(1.01+res[l])).values)
        res.columns.values[m] = l + '_log'
        m += 1
    return res

def addSquared(res, ls):
    m = res.shape[1]
    for l in ls:
        res = res.assign(newcol=pd.Series(res[l]*res[l]).values)
        res.columns.values[m] = l + '_sq'
        m += 1
    return res


train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

train.drop(train[(train['GrLivArea']>4000)].index, inplace=True)
train.drop(train[(train['OverallQual']<5) & (train['SalePrice']>200000)].index, inplace=True)
train.reset_index(drop=True, inplace=True)

# plt.scatter(train['YearRemodAdd'], train['SalePrice'])
# plt.xlabel('YearRemodAdd')
# plt.ylabel('SalePrice')
# plt.show()

cols_to_log=['LotFrontage', 'LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF',
                 'TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea',
                 'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr',
                 'TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF',
                 'EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','YearRemodAdd']

cols_to_square=['YearRemodAdd', 'LotFrontage',
              'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea',
              'GarageCars', 'GarageArea',
              'OverallQual','ExterQual','BsmtQual','GarageQual','FireplaceQu','KitchenQual']

train['ExterQual'] = train['ExterQual'].apply(qual_conv)
train['ExterCond'] = train['ExterCond'].apply(qual_conv)
train['BsmtQual'] = train['BsmtQual'].apply(qual_conv)
train['BsmtCond'] = train['BsmtCond'].apply(qual_conv)
train['KitchenQual'] = train['KitchenQual'].apply(qual_conv)
train['FireplaceQu'] = train['FireplaceQu'].apply(qual_conv)
train['GarageQual'] = train['GarageQual'].apply(qual_conv)
train['PoolQC'] = train['PoolQC'].apply(qual_conv)
train['HeatingQC'] = train['HeatingQC'].apply(qual_conv)

train['LandSlope'] = train['LandSlope'].apply(slope_conv)

train['CentralAir'] = train['CentralAir'].apply(lambda x: 1 if x=='Y' else 0)
train['Street'] = train['Street'].apply(lambda x: 1 if x=='Pave' else 0)
train['PavedDrive'] = train['PavedDrive'].apply(lambda x: 1 if x=='Y' else 0)
train['GarageFinish'] = train['GarageFinish'].apply(garage_finish_conv)
train['BsmtExposure'] = train['BsmtExposure'].apply(bsmt_exposure_conv)
train['BsmtFinType1'] = train['BsmtFinType1'].apply(bsmt_fin_type_conf)
train['BsmtFinType2'] = train['BsmtFinType2'].apply(bsmt_fin_type_conf)
train['HouseStyle'] = train['HouseStyle'].apply(house_style_conv)
train['Foundation'] = train['Foundation'].apply(foundation_conv)
train['MasVnrType'] = train['MasVnrType'].apply(mas_ver_type_conv)

train = addlogs(train, cols_to_log)
train = addSquared(train, cols_to_square)

test['ExterQual'] = test['ExterQual'].apply(qual_conv)
test['ExterCond'] = test['ExterCond'].apply(qual_conv)
test['BsmtQual'] = test['BsmtQual'].apply(qual_conv)
test['BsmtCond'] = test['BsmtCond'].apply(qual_conv)
test['KitchenQual'] = test['KitchenQual'].apply(qual_conv)
test['FireplaceQu'] = test['FireplaceQu'].apply(qual_conv)
test['GarageQual'] = test['GarageQual'].apply(qual_conv)
test['PoolQC'] = test['PoolQC'].apply(qual_conv)
test['HeatingQC'] = test['HeatingQC'].apply(qual_conv)

test['LandSlope'] = test['LandSlope'].apply(slope_conv)

test['CentralAir'] = test['CentralAir'].apply(lambda x: 1 if x=='Y' else 0)
test['Street'] = test['Street'].apply(lambda x: 1 if x=='Pave' else 0)
test['PavedDrive'] = test['PavedDrive'].apply(lambda x: 1 if x=='Y' else 0)
test['GarageFinish'] = test['GarageFinish'].apply(garage_finish_conv)
test['BsmtExposure'] = test['BsmtExposure'].apply(bsmt_exposure_conv)
test['BsmtFinType1'] = test['BsmtFinType1'].apply(bsmt_fin_type_conf)
test['BsmtFinType2'] = test['BsmtFinType2'].apply(bsmt_fin_type_conf)
test['HouseStyle'] = test['HouseStyle'].apply(house_style_conv)
test['Foundation'] = test['Foundation'].apply(foundation_conv)
test['MasVnrType'] = test['MasVnrType'].apply(mas_ver_type_conv)

test = addlogs(test, cols_to_log)
test = addSquared(test, cols_to_square)

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
model.add(keras.layers.Dense(2000, input_dim = X_train.shape[1]))
model.add(keras.layers.Dropout(0.1))
model.add(keras.layers.Dense(1000))
model.add(keras.layers.Dense(500))
model.add(keras.layers.Dense(256))
model.add(keras.layers.Dense(128))
model.add(keras.layers.Dense(56))
model.add(keras.layers.Dense(28))
model.add(keras.layers.Dense(14))
model.add(keras.layers.Dense(8))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(1, kernel_regularizer=keras.regularizers.l1(0.01)))
model.compile(loss = "mse", optimizer=keras.optimizers.Adam())

x_val = X_train[:200]
partial_x_train = X_train[200:]

y_val = train_y[:200]
partial_y_train = train_y[200:]

run_epochs = int(os.environ['EPOCHS'])

tensorBoardCallback = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=100, write_graph=True, write_images=True)

checkpoint_epochs = int(os.environ['CHECKPOINT_EPOCHS'])
checkpoint_path = "checkpoints/cp-{epoch:04d}.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    # Save weights, every 5-epochs.
    period=checkpoint_epochs)

print('Training with {} epochs, saving weights all {} epochs.'.format(run_epochs, checkpoint_epochs))

if 'CHECKPOINT_FILE' in os.environ and os.environ['CHECKPOINT_FILE']:
    reload_checkpoint = os.environ['CHECKPOINT_FILE']
    print('Loading stored weights from {}'.format(reload_checkpoint))
    model.load_weights(reload_checkpoint)

hist = model.fit(partial_x_train, partial_y_train, epochs=run_epochs, batch_size=20, validation_data = (x_val, y_val), callbacks=[tensorBoardCallback, cp_callback])
print(hist.history.keys())
print("train loss min: {}, val loss min: {}, train loss mean: {}, val loss mean: {}".format(np.amin(hist.history['loss']), np.amin(hist.history['val_loss']), np.mean(hist.history['loss']), np.mean(hist.history['val_loss'])))

# plt.plot(hist.history['loss'])
# plt.plot(hist.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
model.save('model_v1.h5');
