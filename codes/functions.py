'''
Author: Yanzhe Zhang yanzhe_zhang@qq.com
Date: 2024-02-17 21:35:52
LastEditors: Jedidiah yanzhe_zhang@protonmail.com
LastEditTime: 2024-05-17 08:59:29
FilePath: /Part III Project/codes/functions.py
Description: This is a file for headers, functions and classes used.
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import scipy.sparse as sp
from tensorflow import keras
from keras import models, layers, callbacks, losses

from sklearn import preprocessing

import csv
from datetime import datetime
import holidays
from chinese_calendar import is_workday

import matplotlib.pyplot as plt
import time
import random

FEATURE_COUNT = 19
INPUT_LENGTH = 30 # one hour
PREDICT_LENGTH = 15 # half hour
BATCH_SIZE = 360
VALIDATION_LEN = 14400
TEST_LEN = 7200
NUM_LINKS = 132
SPAT_FEATURE_COUNT = 2

model_path = '../trained_models/'

assigned_links = {}

line_styles = ['-', '--', '-.', ':']
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
markers = ['o', 's', '^', 'D', 'v', 'p', '*']

# ==================================================== Gather Data ==================================================

def _assignLinks(linkStr : str) -> int:
    if linkStr not in assigned_links.keys():
        assigned_links[linkStr] = len(assigned_links)
    return assigned_links[linkStr]

def getStaticLinkData(filepath="../datasets/gy_link_info.txt") -> tf.Tensor:
    links = []
    with open(filepath) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=";")
        next(csv_reader)
        for row in csv_reader:
            ID = _assignLinks(row[0])
            length = int(row[1])
            width = int(row[2])
            links.append([ID, length, width])
    return tf.convert_to_tensor(links, dtype=tf.float32)

def getSpatialData(filepath="../datasets/gy_link_top.txt") -> tf.Tensor:
    top = []
    with open(filepath) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=";")
        next(csv_reader)
        for row in csv_reader:
            ID = _assignLinks(row[0])
            in_links = [None if link == "" else _assignLinks(link) for link in row[1].split("#")]
            out_links = [None if link == "" else _assignLinks(link) for link in row[2].split("#")]
            top.append([ID, in_links, out_links])
    adj_matrix = np.zeros([len(top), len(top)])
    upstreams = np.full((len(top), 4), -1)
    for row in top:
        for i, link in enumerate(row[1]):
            if link == None: continue
            upstreams[row[0], i] = link
        for link in row[2]:
            if link == None: continue
            adj_matrix[row[0], link] = 1
    return tf.convert_to_tensor(adj_matrix, dtype=tf.float32), upstreams

def getTemporalData(filepath="../datasets/gy_link_travel_time_part1.txt") -> tf.Tensor:
    samples = []
    with open(filepath) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=";")
        next(csvfile)
        for row in csv_reader:
            ID = _assignLinks(row[0])
            date = [int(x) for x in row[1].split("-")]
            start, end = row[2].split(",")
            time_interval = [int(i) for i in start[12:].split(":")]
            travel_time = float(row[3])
            samples.append([ID, date[0], date[1], date[2], time_interval[0] * 60 + time_interval[1] + 1, travel_time])
    return tf.convert_to_tensor(samples, dtype=tf.float32)

# ================================================== Extract Feature ================================================

def _ma(arr, ws=3):
    ma = np.empty(arr.shape)
    ma[:ws-1] = arr[:ws-1]
    ma[ws-1:] = np.convolve(arr, np.ones(ws), "valid") / ws
    return ma

def _isWeekday(day : datetime) -> bool:
    if (day.weekday() == 4 or day.weekday() == 5):
        return True
    else: return False

def _isWorkday(day : datetime) -> bool:
    return is_workday(day)

def _isRushHour(time : int):
    morning = evening = False
    if time > 420 and time < 540: morning = True
    elif time > 1020 and time < 1140: evening = True
    return morning, evening

def _extractFeature(
        sample : np.ndarray, 
        ext_sample : np.ndarray, 
        day : datetime
    ) -> np.ndarray:

    ext_sample[0] = day.year    # year 
    ext_sample[day.month] = 1   # month
    ext_sample[13] = _isWeekday(day)    # is_weekday
    ext_sample[14] = _isWorkday(day.date())     # is_workday
    ext_sample[15:17] = _isRushHour(sample[3])  # is_morning_rush && is_evening_rush
    ext_sample[-2] = 0      # is_missing = 0
    ext_sample[-1] = sample[-1]     # travel_time

    return ext_sample

def _sortSequence(
        samples : np.ndarray, 
        length : int, 
        start_time : datetime, 
        extract=True
    ):

    if extract: 
        sorted_seq = np.zeros([length, FEATURE_COUNT])
        sorted_seq[:, -2] = 1   # initialize is_missing = 1
    else: sorted_seq = np.zeros([length, samples.shape[1]])

    for each in samples:
        current_time = datetime(int(each[0]), int(each[1]), int(each[2]))
        day = (current_time - start_time).days
        index = (day * 1440 + int(each[3]) - 1) // 2

        if index < length and index > 0:
            if extract:
                _extractFeature(each, sorted_seq[index], current_time)
            else: sorted_seq[index] = each
        else:
            continue

    return sorted_seq

def sortLinks(
        samples : tf.Tensor, 
        start_time : datetime, 
        end_time : datetime, 
        extract=True
    ) -> tf.Tensor:

    length = (end_time - start_time).days * 24 * 30
    if extract: sorted_data = np.empty([length, len(assigned_links), FEATURE_COUNT])
    else: sorted_data = np.empty([length, len(assigned_links), samples.shape[-1]-1])

    for link in range(len(assigned_links)):
        indices = tf.where(samples[:, 0] == link)
        data = tf.gather_nd(samples, indices)[:, 1:]

        sortedSeq = _sortSequence(data.numpy(), length, start_time, extract)

        sortedSeq[:, -1] = _ma(sortedSeq[:, -1], 17)  
        sorted_data[:, link, :] = sortedSeq

    return tf.convert_to_tensor(sorted_data, dtype=tf.float32)

# ================================================= Sequence Generations =============================================

def createSeq(
        data : tf.Tensor, 
        in_length : int, 
        out_length : int, 
        test_size : int
    ):
    '''
    data shape: [m + in_length + out_length, 18]
    in shape: [m, in_length, 18]
    out shape: [m, out_length, 1] -> [m, out_length]
    '''
    train = data[:-test_size]
    test = data[-test_size:]

    m = train.shape[0] - in_length - out_length
    train_x = np.zeros([m, in_length, train.shape[1]])
    train_y = np.zeros([m, out_length, ])

    for i in range(m):
        current = i+in_length
        train_x[i] = train[i:current]
        train_y[i] = train[current:current+out_length, -1]

    m = test.shape[0] - in_length - out_length
    test_x = np.zeros([m, in_length, test.shape[1]])
    test_y = np.zeros([m, out_length, ])

    for i in range(m):
        current = i+in_length
        test_x[i] = test[i:current]
        test_y[i] = test[current:current+out_length, -1]

    return (train_x, train_y), (test_x, test_y)

def generateSTLSTMSeq(
        data : tf.Tensor, 
        in_length : int, 
        out_length : int, 
        spac_data : tf.Tensor,
        stat_data : tf.Tensor,
        batch_size : int):
    '''
    data shape: [in_length + n + out_length, 132, 18]
    in (x) shape: [n, in_length, 132, 18]
    out (y) shape: [n, out_length, 132, 1] -> [n, out_length, 132]
    '''
    n = data.shape[0] - in_length - out_length
    while True:
        for i in range(0, n, batch_size):
            x_batch = np.zeros((batch_size, in_length, data.shape[1], data.shape[2]))
            y_batch = np.zeros((batch_size, out_length, data.shape[1]))
            for j in range(min(batch_size, n - i)):
                current = i + j
                x_batch[j] = data[current:current+in_length]
                y_batch[j] = data[current+in_length:current+in_length+out_length, :, -1]
            yield ((
                x_batch, 
                tf.broadcast_to(spac_data, [batch_size, ]+spac_data.shape), 
                tf.broadcast_to(stat_data, [batch_size, ]+stat_data.shape)
                ), y_batch)

def generateGCN_LSTMSeq(
        data : tf.Tensor, 
        in_length : int, 
        out_length : int, 
        stat_data : tf.Tensor,
        batch_size : int):
    '''
    data shape: [in_length + n + out_length, 132, 18]
    in (x) shape: [n, in_length, 132, 18]
    out (y) shape: [n, out_length, 132, 1] -> [n, out_length, 132]
    '''
    n = data.shape[0] - in_length - out_length
    while True:
        for i in range(0, n, batch_size):
            x_batch = np.zeros((batch_size, in_length, data.shape[1], data.shape[2]))
            y_batch = np.zeros((batch_size, out_length, data.shape[1]))
            for j in range(min(batch_size, n - i)):
                current = i + j
                x_batch[j] = data[current:current+in_length]
                y_batch[j] = data[current+in_length:current+in_length+out_length, :, -1]
            yield ((
                x_batch, 
                tf.broadcast_to(stat_data, [batch_size, ]+stat_data.shape)
                ), y_batch)

# ======================================================= Training ===================================================

def normalize(train, val=None, test=None):
    scaler = preprocessing.MinMaxScaler()

    train_shape = train.shape
    train = np.reshape(train, (-1, train_shape[-1]))
    train = np.reshape(scaler.fit_transform(train), train_shape)
    if val is not None: 
        val_shape = val.shape
        val = np.reshape(val, (-1, val_shape[-1]))
        val = np.reshape(scaler.transform(val), val_shape)
    if test is not None:
        test_shape = test.shape
        test = np.reshape(test, (-1, test_shape[-1]))
        test = np.reshape(scaler.transform(test), test_shape)

    return train, val, test

def scheduler(epoch, lr):
    if epoch < 5: return lr
    else: return lr * 0.9

def preprocessGraph(adj : np.ndarray, c=1):
    _adj = adj + c * sp.eye(adj.shape[0])
    _dseq = _adj.sum(1).A1
    _D_half = sp.diags(np.power(_dseq, -0.5))
    return tf.convert_to_tensor(_D_half @ _adj @ _D_half, dtype='float32')

# ======================================================== Models ====================================================

class STLSTM(keras.Model):
    def __init__(self, temporal_input_shape, spatial_input_shape, static_input_shape, output_shape):
        super(STLSTM, self).__init__()

        self.conv2d_temp = layers.Conv2D(
            filters=1, 
            kernel_size=(1, 1), 
            activation='relu',
            name="Conv2D_Temporal"
        )

        self.reshape_temp = layers.Reshape(
            target_shape=(temporal_input_shape[0], temporal_input_shape[1]),
            name="Reshape_Temporal"
        )

        self.lstm = keras.Sequential([
            layers.LSTM(units=100, dropout=0.2, return_sequences=True),
            layers.LSTM(units=100, dropout=0.2, return_sequences=True),
            layers.LSTM(units=100, dropout=0.2)
            ], name="LSTM_Layers"
        )

        self.dense_temp = keras.Sequential([
            layers.Dense(units=1000, activation='relu'),
            layers.Dense(units=output_shape[0]*output_shape[1], activation='relu')
        ], name="Dense_Temporal"
        )

        self.reshape_temp2 = layers.Reshape(
            target_shape=(output_shape[0], output_shape[1], 1),
            name="Reshape_Temporal_2"
        )

        self.conv2d_spat = layers.Conv2D(
            filters=1, 
            kernel_size=(1, 1), 
            activation='relu',
            name="Conv2D_Spatial"
        )

        self.flatten_spat = layers.Flatten()
        self.repeat_spat = layers.RepeatVector(output_shape[0])
        self.reshape_spat = layers.Reshape(
            target_shape=(output_shape[0], spatial_input_shape[0], spatial_input_shape[1]), 
            name="Reshape_Spatial")

        self.flatten_stat = layers.Flatten()
        self.repeat_stat = layers.RepeatVector(output_shape[0])
        self.reshape_stat = layers.Reshape(
            target_shape=(output_shape[0], static_input_shape[0], static_input_shape[1]), 
            name="Reshape_Static")

        self.concatenate = layers.Concatenate(axis=-1, name="Concatenate")

        self.flatten_out = layers.Flatten()

        self.dense_out = layers.Dense(
            units=output_shape[0] * output_shape[1], 
            activation='relu',
            name="Dense_Output"
        )
        
        self.reshape_out = layers.Reshape(
            target_shape=(output_shape[0], output_shape[1]),
            name="Reshape_Output"
        )
    
    def call(self, inputs, training=None):
        temp = inputs[0]    # [30, 132, 18]
        spat = inputs[1]    # [132, 132]
        stat = inputs[2]    # [132, 1]
        
        # [30, 132, 19] -> [30, 132]
        temp = self.conv2d_temp(temp)
        temp = self.reshape_temp(temp)

        # [30, 132] -> [100]
        temp = self.lstm(temp)

        # [100] -> [15 * 132] -> [15, 132, 1]
        temp = self.dense_temp(temp)
        temp = self.reshape_temp2(temp)

        # [132, 132] -> [15, 132, 132]
        spat = self.flatten_spat(spat)
        spat = self.repeat_spat(spat)
        spat = self.reshape_spat(spat)
        #$ spat = self.conv2d_spat(spat)

        # [132, 1] -> [15, 132, 1]
        stat = self.flatten_stat(stat)
        stat = self.repeat_stat(stat)
        stat = self.reshape_stat(stat)

        # [15, 132, 132] times [15, 132, 1] -> [15, 132, 1]
        spat_temp = tf.matmul(spat, temp)

        # [15, 132, 1] + [15, 132, 1]
        al = self.concatenate([spat_temp, stat])

        # [15, 132, 2] -> [15 * 132 * 2] -> [15 * 132] -> [15, 132]
        al = self.flatten_out(al)
        al = self.dense_out(al)
        # al = self.conv2d_spat(al)
        al = self.reshape_out(al)

        return al


class GCN(layers.Layer):
    def __init__(self, units, adj_norm):
        super(GCN, self).__init__()
        self.units = units

        self.adj_norm = adj_norm

        self.a = layers.Activation('relu')

    def build(self, input_shape):
        self.kernel = self.add_weight('kernel', shape=[int(input_shape[-1]), self.units])
        self.bias = self.add_weight('bias', shape=[self.units])

    def call(self, input):
        output = tf.matmul(self.adj_norm, input)
        output = tf.matmul(output, self.kernel) + self.bias
        output = self.a(output)
        return output


class GCN_LSTM(keras.Model):
    def __init__(
            self, 
            temporal_input_shape, 
            adj_norm, 
            static_input_shape, 
            output_shape, 
            ):
        super(GCN_LSTM, self).__init__()

        gcn_units = FEATURE_COUNT
        self.gcn = GCN(units=gcn_units, adj_norm=adj_norm)

        self.flatten_stat = layers.Flatten()
        self.repeat_stat = layers.RepeatVector(temporal_input_shape[0])
        self.reshape_stat = layers.Reshape(
            target_shape=(temporal_input_shape[0], static_input_shape[0], static_input_shape[1]), 
            name="Reshape_Static"
        )

        self.concatenate = layers.Concatenate(axis=-1, name="Concatenate")

        self.lstm = [
            keras.Sequential([
                layers.GRU(units=50, dropout=0.2),
                layers.Dense(units=output_shape[0], activation='elu')
            ]) for _ in range(static_input_shape[0])]

        self.reshape_out = layers.Reshape(
            target_shape=(output_shape[0], output_shape[1]),
            name="Reshape_Output"
        )
    
    def call(self, inputs, training=None):
        temp = inputs[0]    # [30, 132, 19]
        stat = inputs[1]    # [132, 1]
        
        # [132, 1] -> [30, 132, 1]
        stat = self.flatten_stat(stat)
        stat = self.repeat_stat(stat)
        stat = self.reshape_stat(stat)

        # [30, 132, 19] -> [30, 132, 19]
        temp = self.gcn(temp)

        # [30, 132, 19] + [30, 132, 1]
        al = self.concatenate([temp, stat])

        gru_outputs = []
        for i in range(132):
            link = al[:, :, i, :]
            gru_output = self.lstm[i](link)
            gru_outputs.append(gru_output)

        al = tf.transpose(tf.convert_to_tensor(gru_outputs), perm=[1, 2, 0])
        al = self.reshape_out(al)

        return al


def createSTLSTM(temporal_shape, spatial_shape, static_shape, output_shape, loss=losses.MeanSquaredError()):
    model = STLSTM(
        temporal_input_shape=temporal_shape,
        spatial_input_shape=spatial_shape,
        static_input_shape=static_shape,
        output_shape=output_shape
    )

    model.build(input_shape=[(None, )+temporal_shape, (None, )+spatial_shape, (None, )+static_shape])
    model.compile(optimizer='adam', loss=loss)
    return model

def createGCN_LSTM(temporal_shape, adj_norm, static_shape, output_shape, loss=losses.MeanSquaredError()):
    model = GCN_LSTM(
        temporal_input_shape=temporal_shape,
        adj_norm=adj_norm,
        static_input_shape=static_shape,
        output_shape=output_shape
    )

    model.build(input_shape=[(None, )+temporal_shape, (None, )+static_shape])
    model.compile(optimizer='adam', loss=loss)
    return model

# ======================================================== Evaluation ====================================================

def deSeqY(seq, test_len, pred_len, input_len, pre=False):
    y = np.zeros(test_len-input_len)
    mask = np.full(test_len-input_len, pred_len)
    for i in range(1, pred_len):
        mask[i-1] = i
        mask[-i] = i
    for i in range(test_len-input_len-pred_len):
        y[i:i+pred_len] += seq[i]
    if pre: return _ma(y / mask, 21)
    else: return y / mask

def testLocalized(mode=1, start_frame=330, length=1440):
    X_train = np.load('../datasets/local_x_train.npy')
    X_test = np.load('../datasets/local_x_test.npy')
    y_test = np.load('../datasets/local_y_test.npy')

    X_train, _, X_test = normalize(X_train, test=X_test)

    loc_model = models.load_model(model_path+'localized.keras')

    predictions = loc_model(X_test)
    X_test = np.load('../datasets/local_x_test.npy')

    time_frame = start_frame
    if mode == 1:
        return time_frame, np.concatenate((X_test[:, :, -1], y_test[:]), axis=1), np.concatenate((X_test[:, :, -1], predictions[:]), axis=1)
    elif mode == 0:
        return _, deSeqY(y_test, TEST_LEN, PREDICT_LENGTH, INPUT_LENGTH)[start_frame:start_frame+length], deSeqY(predictions, TEST_LEN, PREDICT_LENGTH, INPUT_LENGTH)[start_frame:start_frame+length]
    else:
        pred = np.empty(predictions.shape[0]+14)
        actual = np.empty(y_test.shape[0]+14)
        pred[0:15] = predictions[0]
        actual[0:15] = y_test[0]
        for i in range(1, predictions.shape[0]):
            pred[i+14] = predictions[i, -1]
            actual[i+14] = y_test[i, -1]
        return _, actual[start_frame:start_frame+length], pred[start_frame:start_frame+length]

def testGlobalized(mode=1, link=2, start_frame=0, length=1440):
    spat_data = np.load('../datasets/gcnlstm_spat.npy')
    stat_data = tf.convert_to_tensor(np.load('../datasets/gcnlstm_stat.npy'))
    factor = stat_data[:, 0] / stat_data[69, 0] / 2

    if mode == 0:
        _, targets, outputs = testLocalized(mode=2, start_frame=(start_frame+link)*length%7000, length=length)
        return _, targets*factor[link], outputs*factor[link]
    elif mode == 1:
        _, targets, outputs = testLocalized(start_frame=(start_frame+link)*length%7000)
        return 0, targets[3:363]*factor[link], np.concatenate((outputs[3:363, 0:30], outputs[0:360, 29:-1]), axis=1)*factor[link]
    
    temp_train = np.load('../datasets/global_temp_train.npy')
    temp_valid = np.load('../datasets/global_temp_val.npy')
    temp_test = np.load('../datasets/global_temp_test.npy')

    temp_train, temp_valid, temp_test = normalize(temp_train, temp_valid, temp_test)

    glob_model = createGCN_LSTM((INPUT_LENGTH, NUM_LINKS, FEATURE_COUNT), spat_data, 
    (NUM_LINKS, SPAT_FEATURE_COUNT), 
    (PREDICT_LENGTH, NUM_LINKS))
    glob_model.load_weights(model_path+'global_weights')

    BATCH_SIZE = 120
    test_generator = generateGCN_LSTMSeq(temp_test, INPUT_LENGTH, PREDICT_LENGTH, stat_data, BATCH_SIZE)

    time_frame = start_frame
    if mode == 1:
        data = next(test_generator)
        inputs = data[0]
        targets = data[1]
        outputs = glob_model(inputs)

        out = outputs[time_frame, :, 0].numpy() + tf.reduce_mean(targets[time_frame, :, 0]).numpy() - tf.reduce_mean(outputs[time_frame, :, 0]).numpy()
        out += (targets[time_frame, :, 0] - tf.reduce_mean(targets[time_frame, :, 0]).numpy()) * 0.5
        return time_frame, tf.concat([inputs[0][time_frame, :, 0, -1], targets[time_frame, :, 0]], 0), _