# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 15:44:00 2018

@author: ziwenruo
"""

import pymongo
"""
X:
    299*299
    
    tick_data
    len:37*37 -1
    dates(next_trading_day - today)#atleast 2
    hangye type
        starts with 0   1
                    6   2
                    3   3
    37*37 repeat 8
    

Y:
    (next_next_trading_day's high - today's close)/today's close
    >= 4%
    -4% < Y < 4%
    <= 4%
"""
import pandas as pd
from pandas.tseries.offsets import CDay
from datetime import date
import numpy as np
import tensorflow as tf
import os
import logging
from collections import namedtuple
import argparse
import pdb


parser = argparse.ArgumentParser()
parser.add_argument('-d','--day', default=date.today().isoformat(), help='target day')
parser.add_argument('mode', help="speicify 'train', 'test' or 'predict'")


LATEST_STOCK_BASIC_DB_REC_DAY = '2018-03-19'
LABELS_FILENAME = 'labels.txt'
_CLASS_NAMES = [
    'sink  ||<-4%',
    'normal||-4%~4%',
    'araise||>4%',
]

def write_label_file(labels_to_class_names, dataset_dir,
                     filename=LABELS_FILENAME):
    """Writes a file with the list of class names.
    
    Args:
      labels_to_class_names: A map of (integer) labels to class names.
      dataset_dir: The directory in which the labels file should be written.
      filename: The filename where the class names are written.
    """
    labels_filename = os.path.join(dataset_dir, filename)
    with tf.gfile.Open(labels_filename, 'w') as f:
      for label in labels_to_class_names:
        class_name = labels_to_class_names[label]
        f.write('%d:%s\n' % (label, class_name))


def has_labels(dataset_dir, filename=LABELS_FILENAME):
    """Specifies whether or not the dataset directory contains a label map file.
    
    Args:
      dataset_dir: The directory in which the labels file is found.
      filename: The filename where the class names are written.
    
    Returns:
      `True` if the labels file exists and `False` otherwise.
    """
    return tf.gfile.Exists(os.path.join(dataset_dir, filename))


def read_label_file(dataset_dir, filename=LABELS_FILENAME):
    """Reads the labels file and returns a mapping from ID to class name.
    
    Args:
      dataset_dir: The directory in which the labels file is found.
      filename: The filename where the class names are written.
    
    Returns:
      A map from a label (integer) to class name.
    """
    labels_filename = os.path.join(dataset_dir, filename)
    with tf.gfile.Open(labels_filename, 'rb') as f:
      lines = f.read().decode()
    lines = lines.split('\n')
    lines = filter(None, lines)
    
    labels_to_class_names = {}
    for line in lines:
      index = line.index(':')
      labels_to_class_names[int(line[:index])] = line[index+1:]
    return labels_to_class_names
    


def initLogging(logFilename):
    
    logging.basicConfig(
            level       = logging.DEBUG,
            format      = "%(message)s",
            #format     = 'LINE %(lineno)-4d %(levelname)-8s %(messages)s',
            datefmt     =  '%m-%d %H:%M',
            filename    = logFilename,
            filemode    = 'w'
            )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)-8s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
SRECORD = namedtuple('REC',"date high close tick")    
#class sRecord():
#    def __init__(self, high=None, close=None, tick=None, date=None):
#        self.high = high
#        self.close = close
#        self.tick = tick
#        self.date = date
#        
#    def __repr__(self):
#        return "<DATE:{}, HIGH:{}, CLOSE{}>".format(self.date, self.high, self.close)
    
class GEN_datasets():    
    
    def __init__(self, split_name, sday=LATEST_STOCK_BASIC_DB_REC_DAY, eday='2016-01-01', data_dir='stock_tfexample', th_size = 30,
                 rief=10000):
        self.fname = "tick"+sday.replace('-',"")+"_%s_{:0>2d}.tfrecord" %(split_name)
        self.data_dir = os.path.join(".", data_dir, split_name)
        self.records_in_each_file = rief
        self.sday = sday
        self.eday = eday
        self.client = pymongo.MongoClient('localhost', 27017)
        self.db = self.client['A-SHARE']
        
        self.db_stock_basics = self.client['STOCK_BASICS'][sday.replace('-', '')]
        self.width = 37
        self.tk_size = self.width*self.width-1
        self.th_size = th_size
        self.count = 0
        if not tf.gfile.Exists(self.data_dir):
            tf.gfile.MakeDirs(self.data_dir)
            
        self._records_gzip = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
        self.record_write_str = "{:<6d}:{} on {} with next day {}, date gap {:2d}, labeled {}, ({})"
        


    def int64_feature(self, values):
        """Returns a TF-Feature of int64s.
        
          Args:
        values: A scalar or list of values.
        
          Returns:
        A TF-Feature.
          """
        if not isinstance(values, (tuple, list)):
            values = [values]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

    
    def bytes_feature(self, values):
        """Returns a TF-Feature of bytes.
        
        Args:
        values: A string.
        
        Returns:
        A TF-Feature.
        """
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))
    
    def float_feature(self, values):
        """Returns a TF-Feature of floats.
        
          Args:
        #    values: A scalar of list of values.
             values: A numpy array 
        
          Returns:
            A TF-Feature.
        """
        return tf.train.Feature(float_list=tf.train.FloatList(value=values))
        
    def _date2str(self, day):
        """type(day) should be pandas._libs.tslib.Timestamp """
        return day.strftime("%Y-%m-%d")
    
    def _str2date(self, dstr):
        #dstr should be of format as 2017-11-07
        return pd.to_datetime(dstr)
    
    def date_gap(self, dstrS, dstrE):
        x = self._str2date(dstrS) - self._str2date(dstrE)
        return x.days
        
    def _get_code_list(self):
        return self.db.collection_names()
    
    def _test_sample_capity(self, code):
        if self.db[code].count() > self.th_size:
            return True
        else:
            return False
        
    def _get_code_vol(self, code):
        code_info = self.db_stock_basics.find_one({'code':code})
        return code_info['outstanding']
    
    def _get_record(self, code, tdate):
        record = self.db[code].find_one({'date':tdate},{'_id':0, 'date':1, 'high':1,'close':1,'tick':1})
        if record and len(record) == 4:
            return SRECORD(**record)               
        
    def _get_tick_fake(self, code):
        col_list = []
        for i in self.db[code].find():
            col_list.append({'date':i['date'],'high':i['high'],'close':i['close']})
        return pd.DataFrame(col_list)
        
    def _processing_tick(self, tick, tick_vol):
        x = tick['vol'].nlargest(self.tk_size)
        tick = tick.loc[x.index.sort_values(ascending=True)]
        #debug purpose
#        print("tick head:\n{}\ntick tail:\n{}".format(tick.head(2), tick.tail(2)))
        tick['price_shift1'] = tick['price'].shift()
        tick['delta_price'] = tick['price'] - tick['price_shift1']       
        tick['delta_vol'] = tick['vol']/(100.0*tick_vol)
#        tick.fillna(value=0, inplace=True)
        return tick.loc[:,['delta_price', 'delta_vol', 'type']]
    
    def _type_y(self, y):
        if y > 0.04:
            return 2
        elif y < -0.04:
            return 0
        else:
            return 1
    def _type_hy(self, code):
        if code.startswith('0'):
            return 1
        elif code.startswith('6'):
            return 0
        else:
            return -1
        
   
    def repeat_and_padding(self, arr, repeat=8, padsize=((1,2),(1,2),(0,0))):
        """
        input: numpy array of shape (W,H,C)
        rebuild the array of size (W,H,C):
            W of repeat times @ axis 0
            H of repeat times @ axis 1
        if padsize = ((1,2),(1,2),(0,0)):
            padding W & H left & top with 1 '0's,
                          right & bottom with 2 '0's
        """
#        print("Before edit: input.shape=", arr.shape)
        arr = arr.repeat(repeat, axis=0)
        arr = arr.repeat(repeat, axis=1)
        arr = np.pad(arr, padsize, mode='constant')
#        print("After  edit: input.shape=", arr.shape)
        return arr

    
    def build_tfexample(self, code, s_record, two_day_later_high):
        feature_type = self._type_hy(code)
        v = self._get_code_vol(code)
        tick = pd.read_json(s_record.tick, orient='index')
        if len(tick) >= self.tk_size:
            tick_processed = self._processing_tick(tick, v)
            if two_day_later_high:
                feature_Y = self._type_y((two_day_later_high-s_record.close)/float(s_record.close))
            else:
                feature_Y = 888
                
            tempdf = pd.DataFrame([feature_type], columns=['type'])
            tick_processed = tick_processed.append(tempdf, ignore_index=True)
            tick_processed.fillna(value=0, inplace=True)
#            print("{} on {} with label:{} and tick_processed.shape {}".format(code, s_record.date, feature_Y, tick_processed.shape))
            tick_processed = tick_processed.values
            # of shape (1369, 3)
            tick_processed = tick_processed.reshape(self.width,self.width,3)
            tick_processed = self.repeat_and_padding(tick_processed)
            # of shape (299,299,3)
            return feature_Y, tf.train.Example(features=tf.train.Features(feature={
                                          'code': self.bytes_feature(code.encode()),
                                          'date': self.bytes_feature(s_record.date.encode()),
                                          'label_type': self.int64_feature(feature_Y),
                                          'tick': self.float_feature(tick_processed.reshape(-1)),}))
            
            
    def _dealwith_tick(self, code, info_tail, gap=2):                 
        daylist = [rec['date'] for rec in self.db[code].find()]
        daylist.sort()
        for i in range(len(daylist)):
            tarday = daylist[i]
            for g in reversed(range(gap+1)):
                try:
                    nextday = daylist[i+g]
                    if self.date_gap(nextday, tarday) == gap:
                        break
                except IndexError:
                    continue
            else:
                continue
            
            rec_tarday = self._get_record(code, tarday)
            rec_nexday = self._get_record(code, nextday)
            
            
            if rec_tarday and rec_nexday:
                try:
                    tY, record_example = self.build_tfexample(code, rec_tarday, rec_nexday.high)
                except TypeError:
                    continue
                    
                if record_example:
                    logging.info(self.record_write_str.format(self.count+1, 
                                                              code, rec_tarday.date, rec_nexday.date, gap, tY, info_tail))
                    yield tY, record_example
        
     
    def gen_predict_data(self, tar_day, clist=None, gap=2):
        nex_day = self._date2str(self._str2date(tar_day) + gap*CDay())
        record_file_name = os.path.join(self.data_dir, self.fname.format(0))                       
        writer = tf.python_io.TFRecordWriter(record_file_name, options=self._records_gzip)
        if clist:
            code_list = clist
        else:
            code_list = self._get_code_list()
            
        count = 0
        clist_length = len(code_list)
        for idx, c in enumerate(code_list, start=1):
            logging.info("{} comes, processing ...<{:>4d}:{}>".format(c, idx, clist_length))
            record = self._get_record(c,tar_day)
            nex_record = self._get_record(c, nex_day)
            if nex_record:
                two_day_later_high = nex_record.high
            else:
                two_day_later_high = None
            if record:
                _, record_example = self.build_tfexample(c, record, two_day_later_high)
            else:
                logging.info("{} skipping, None tick record found".format(c))
                continue
            if record_example:
                writer.write(record_example.SerializeToString())
                count += 1
            else:
                logging.info("{} skipping, None TFrecord formed. Tick length {}".format(c, record.tick.count('datetime')))
                
        writer.close()
        logging.info("Data creating done, totally {} records".format(count))
                
    def gen_stock_tfrecord(self, clist=None, gap=2):
        self.countt0 = 0
        self.countt1 = 0
        self.countt2 = 0
        if not has_labels(self.data_dir):
            labels_to_class_names = dict(zip(range(len(_CLASS_NAMES)), _CLASS_NAMES))
            write_label_file(labels_to_class_names, self.data_dir)

        if clist:
            code_list = clist
        else:
            code_list = self._get_code_list()[1255:]
            
        for idx, c in enumerate(code_list, start=1255):
            print("processing {}...".format(c))
            record_file_t0 = os.path.join(self.data_dir, "tick"+LATEST_STOCK_BASIC_DB_REC_DAY.replace('-',"")+"_t0_%s.tfrecord" % c)
            record_file_t1 = os.path.join(self.data_dir, "tick"+LATEST_STOCK_BASIC_DB_REC_DAY.replace('-',"")+"_t1_%s.tfrecord" % c)
            record_file_t2 = os.path.join(self.data_dir, "tick"+LATEST_STOCK_BASIC_DB_REC_DAY.replace('-',"")+"_t2_%s.tfrecord" % c)
    #        writer = tf.python_io.TFRecordWriter(record_file_name, options=self._records_gzip)
            writer0 = tf.python_io.TFRecordWriter(record_file_t0, options=self._records_gzip)
            writer1 = tf.python_io.TFRecordWriter(record_file_t1, options=self._records_gzip)
            writer2 = tf.python_io.TFRecordWriter(record_file_t2, options=self._records_gzip)
            if self._test_sample_capity(c):
#                self._dealwith_tick(c, writer, "{:4d}|{}".format(idx, len(code_list)))
                for y,rec in self._dealwith_tick(c, "{:4d}|{}".format(idx, len(code_list)), gap=gap):
                    if y == 0:
                        writer0.write(rec.SerializeToString())  
                        self.countt0 += 1
                    elif y == 1:
                        writer1.write(rec.SerializeToString())
                        self.countt1 += 1                        
                    elif y == 2:
                        writer2.write(rec.SerializeToString())
                        self.countt2 += 1                        
#                    writer.write(rec.SerializeToString())
                    self.count += 1
#                    pdb.set_trace()
#                    if not self.count % self.records_in_each_file: 
#                        writer.close()
#                        record_file_count = self.count // self.records_in_each_file
#                        record_file_name = os.path.join(self.data_dir, self.fname.format(record_file_count))
#                        writer = tf.python_io.TFRecordWriter(record_file_name, options=self._records_gzip)
            else:
#                print("%s's data is not long enough" % c)
                logging.info("E: %s's data is not long enough" % c)
            
            writer0.close()  
            writer1.close() 
            writer2.close()
                
#                pass
#        else:
#            writer0.close()  
#            writer1.close() 
#            writer2.close() 
        logging.info("Totally Type0 counts:{}, Type1 counts:{}, Type2 counts:{}".format(self.countt0,self.countt1,self.countt2))
           

            
def read_stock_tfrecord(fname):
    rec_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    record_iterator = tf.python_io.tf_record_iterator(fname, options=rec_opt)
    example = tf.train.Example()
    TFREC = namedtuple('TFREC',"date code label tick") 
    for i in iter(record_iterator):
        example.ParseFromString(i)
        mycode = example.features.feature['code'].bytes_list.value
        mydate = example.features.feature['date'].bytes_list.value
        mylabel = example.features.feature['label_type'].int64_list.value[0]
        mytick = example.features.feature['tick'].float_list.value
        yield TFREC(mydate, mycode, mylabel, mytick)
                    

                        
def main():
    
    args = parser.parse_args()
    
    import time
    tstart = time.time()
    
    lv_se_shi_ping = [
            '000672',
                     '002194',
                     '002460',
                     '300078',
                     '300304',
                     '300458',
                     '300466',
                     '300496',
                     '600139',
                     '600392',
                     '601988',
                     '603616',
                     '603799',
                     '603822']

#    sta_day = '2017-11-07'
#    end_day = '2018-03-19'
    tar_day = args.day    
    if args.mode == 'predict':
        afeature = GEN_datasets('predict{}'.format(tar_day.replace('-', '')))
        logFilename = os.path.join(afeature.data_dir, 'gen_stock_tfrecord.log')
        initLogging(logFilename)
        afeature.gen_predict_data(tar_day=tar_day, clist=lv_se_shi_ping) 
        
    else:
        afeature = GEN_datasets(args.mode, rief=50)  
        logFilename = os.path.join(afeature.data_dir, 'gen_stock_tfrecord.log')
        initLogging(logFilename)        
        afeature.gen_stock_tfrecord(clist=None)

    

    logging.info("time elapsed {}".format(time.time() - tstart))
    
if __name__=='__main__':
    main()


