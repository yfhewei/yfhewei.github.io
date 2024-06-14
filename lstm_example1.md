
### about the problem
* how to classify the dish by it's name.

### code example
***
* install the necessary package

```python
%pip install h5py==2.10.0
#安装HDF5文件的库
%pip install pandas==1.1.0
#安装pandas
%pip install scikit-learn==0.24.2
#安装sklearn
%pip install tensorflow-gpu==1.15.0
#安装支持GPU的tf
%pip install tokenizers
#安装文本分词的工具库
```
***
* import the related package

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#确保所有基于CUDA的库只会看到并使用编号为0的GPU
import tensorflow as tf
print(tf.__version__)
from tensorflow.python.client import device_lib
device_lib.list_local_devices()
[tf.test.is_gpu_available(), tf.test.is_gpu_available(cuda_only=True)]

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder   #
from tensorflow.keras import Input, Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Attention
from tensorflow.keras.callbacks import EarlyStopping

import pandas as pd
import numpy as np
import re
```

* get the dataset

```python
dish_raw = pd.read_csv('dish_raw.csv')[['cat_name','customer_id','dish_name']]
#len(dish_raw)
#dish_raw.head()
```

* proprocess

```python
#去除一些特殊字符噪音
def clean_dish_name(text): 
    stopwords = ['哦','呀','呢','毫升','升','听','罐','瓶','一','份','大份','中份','小份','一']
    # 去掉（）和【】内的内容
    pattern_parentheses = r'(\（|\【)(.*?)(\）|\】)'
    cleaned_text1 = re.sub(pattern_parentheses, '', text)
    
    # 只保留中文部分
    pattern_chinese = r'[^\u4e00-\u9fa5]' 
    cleaned_text = re.sub(pattern_chinese, '', cleaned_text1)

    #分词
    seglist = [char for char in cleaned_text]
    words = []
    for seg in seglist:
        if seg not in stopwords:
            words.append(seg)
    
    return ' '.join(words)
dish_raw['clean_dish_name'] = dish_raw['dish_name'].apply(lambda x:clean_dish_name(x))
```

* modeling

```python
#tokenizer
MAX_NB_WORDS = 8000
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(dish_raw[dish_raw['clean_dish_name'].notna()]['clean_dish_name'].values)
word_index = tokenizer.word_index

#label encoding
le = LabelEncoder()
le.fit(dish_raw['cat_name'])

#train/test split
custDF = dish_raw.groupby(['cat_name','customer_id'],as_index=False)['clean_dish_name'].count()
custDF['weight'] = 1/custDF['clean_dish_name']
custDF.columns=['cat_name','customer_id','dish_cnt','weight']
custDF = custDF.sample(frac=1, random_state=42).reset_index(drop=True)

train = custDF[:40000]
test = custDF[50000:55000]

#build the model
MAX_LENGTH = 30
NUM_CLASS = 123

dish_train = train[['customer_id','weight']].merge(dish_raw[dish_raw['clean_dish_name'].notna()], how='inner')

X_train = tokenizer.texts_to_sequences(dish_train['clean_dish_name'].values)

# padding
X_train = pad_sequences(X_train, MAX_LENGTH)

#数值化
Y_train = le.transform(dish_train['cat_name'])

W_train = dish_train['weight'].values

model = Sequential()
model.add(Embedding(MAX_NB_WORDS, 180, input_length=MAX_LENGTH))
model.add(LSTM(units=180, dropout=0.2, return_sequences=True))
model.add(Flatten())   # 拉平输出
#LSTM参数return_sequences=True时，需要通过Flatten拉平，如果该参数为False，则不需要Flatten，直接跟Dense即可。
#model.add(Dense(units=540, activation='softmax'))
model.add(Dense(units=NUM_CLASS, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')   #loss='sparse_categorical_crossentropy'

print(model.summary())

# 适配binary_crossentropy loss需要进行one-hot
y_train_onehot = tf.keras.utils.to_categorical(Y_train, NUM_CLASS)

# 设置早停回调函数
early_stop = EarlyStopping(monitor='loss', patience=5)

model.fit(X_train, 
          y_train_onehot,
          sample_weight=W_train, # 对样本重要性进行加权
          callbacks=[early_stop],
          epochs=30, 
          batch_size=20000
         )
```

* predict

```python
dish_testDF = testDF[['customer_id']].merge(dish_raw[dish_raw['clean_dish_name'].notna()], how='inner').sample(frac=0.4, random_state=42)
dish_testDF = dish_testDF.reset_index()
X_test = tokenizer.texts_to_sequences(dish_testDF['clean_dish_name'].values)
# padding
X_test = pad_sequences(X_test, MAX_LENGTH)
Y_test = le.transform(dish_testDF['second_cat_name'])
y_prob = model.predict(X_test)
# y_pred = y_prob.argmax(axis=1)
probDF = pd.DataFrame(y_prob)
dish_reDF = pd.concat([dish_testDF, probDF], axis=1)
```

* evaluate

```python
# dish的topN的召回
# —— 累计概率的阈值
# —— 各个业态的概率分布情况
raw_pred_sampleDF = dish_reDF.drop(columns='index')#.sample(frac=0.2, random_state=42)
raw_reDF = raw_pred_sampleDF.groupby(['clean_dish_name'],as_index=False).mean()
raw_reDF.columns = ['clean_dish_name','customer_id'] +le.classes_.tolist()
dish_cat_predDF = raw_reDF.set_index('clean_dish_name').drop(columns=['customer_id']).stack().to_frame()
dish_cat_predDF = dish_cat_predDF.reset_index()
dish_cat_predDF.columns = ['clean_dish_name','second_cat_name','y_prob']
dis_cat_testDF = raw_pred_sampleDF[['clean_dish_name','second_cat_name','customer_id']].groupby(['clean_dish_name','second_cat_name'])[['customer_id']].count()
dis_cat_testDF['pct'] = dis_cat_testDF['customer_id']/dis_cat_testDF.groupby(['clean_dish_name'])['customer_id'].transform('sum')
dis_cat_testDF = dis_cat_testDF.reset_index()
mrg_reDF = dish_cat_predDF.merge(dis_cat_testDF, how='left')
```
