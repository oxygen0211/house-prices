import tensorflow as tf
import pandas as pd
from tensorflow import keras
from scipy.stats import skew
import numpy as np

model = keras.models.load_model('model_v1.h5')
test = pd.read_csv('./data/test.csv')
X_test = pd.read_csv('./data/test-processed.csv')

print('Predicting prices...')
model.summary()

pred = model.predict(X_test);

denorm_predict = np.exp(pred)
denorm_predict = np.reshape(denorm_predict, -1)
submission = pd.DataFrame({'Id': test['Id'], 'SalePrice': denorm_predict})
submission.loc[submission['SalePrice'] <= 0, 'SalePrice'] = 0
submission.to_csv('submission.csv', index=False)
