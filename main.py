from Model import Model
import numpy as np
import tensorflow as tf

#3フレームまで全く同じ時系列データdata1, data2
#4フレーム以降は、異なる。
#3フレームまでをmaskはTrueとした。
if __name__ == '__main__':
    data1 = np.random.randn(1, 5, 100, 100, 3)
    mask1 = tf.cast([[True, True, True, False, False]], tf.bool)

    #3フレームまでは同じデータを生成する
    data_l = data1[:,0:3]
    #4フレーム以降のデータを生成
    data_r = np.random.randn(1, 2, 100, 100, 3)
    data2 = np.concatenate([data_l, data_r],1)
    mask2 = tf.cast([[True, True, True, False, False]], tf.bool)
    
    model = Model(5, 100, 100, 3)
    print("data1に対する出力 : ", model(data1, mask1))
    print("data2に対する出力 : ", model(data2, mask2))
