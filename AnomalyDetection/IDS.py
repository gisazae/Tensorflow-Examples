## Procesar el KDD 2015 Intrusion Detection 
#!/usr/bin/env python
import tensorflow as tf
msg = tf.constant("Test Dataset TF")
sess = tf.Session()
print(sess.run(msg))
x = tf.constant(9)
y = tf.constant(2)
print(sess.run(x + y))
# Based on https://github.com/albahnsen/ML_SecurityInformatics
import pandas as pn
pn.set_option('display.max_columns', 500)
import zipfile
with zipfile.ZipFile('../datasets/UNB_ISCX_NSL_KDD.csv.zip', 'r') as z:
    f = z.open('UNB_ISCX_NSL_KDD.csv')
    data = pn.io.parsers.read_table(f, sep=',')
data.head()
print(data)
y = (data['class'] == 'anomaly').astype(int)
print(y)
y.value_counts()
X = data[['same_srv_rate','dst_host_srv_count']]
print(X)
y = (data['class'] == 'anomaly').astype(int)
print(y)
## Falta aplicar tensores con lineal_regression
## .................