# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 11:18:12 2018

@author: ziwenruo
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 09:35:34 2018

@author: ziwenruo
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import os
import argparse
import logging
from collections import namedtuple
#import pdb

REC_DB_PATH = 'G:\\workarea\\scripts\\stock_tfexample\\train'
SRECORD = namedtuple('REC',"date high close tick")    

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
    


record_info = pd.DataFrame({
        'name':['code','date','label_type','tick'],
        'type':[tf.string,tf.string,tf.int64,tf.float32],
        'shape':[(),(),(),(299,299,3)],
        'isbyte':[True,True,False,False],
        'length_type':['fixed','fixed','fixed','var'],
        }, columns=['name','type','shape','isbyte','length_type'])

def get_filenames(cate, path=REC_DB_PATH):
    # get all file names 
    files= os.listdir(path) 
    filepaths = [os.path.join(path,file) for file in files if not os.path.isdir(file) and '.tfrecord' in file]
    filepaths = [file for file in filepaths if "_%s_"%cate in file]
    filepaths.sort(key=os.path.getsize, reverse=True)
    return np.array(filepaths)
    
    

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
        

#
#tarfile = ["G:\\workarea\\scripts\\stock_tfexample\\train\\tick20180319_t1_603712.tfrecord",
#           "G:\\workarea\\scripts\\stock_tfexample\\train\\tick20180319_t0_002294.tfrecord",
#           "G:\\workarea\\scripts\\stock_tfexample\\train\\tick20180319_t0_002032.tfrecord",
#           ]
    
def provide_rec(flist):
    for f in flist:
        yield from read_stock_tfrecord(f)
        
def float_feature(values):
    """Returns a TF-Feature of floats.
    
      Args:
    #    values: A scalar of list of values.
         values: A numpy array 
    
      Returns:
        A TF-Feature.
    """
    return tf.train.Feature(float_list=tf.train.FloatList(value=values)) 

def int64_feature(values):
    """Returns a TF-Feature of int64s.
    
      Args:
    values: A scalar or list of values.
    
      Returns:
    A TF-Feature.
      """
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
    """Returns a TF-Feature of bytes.
    
    Args:
    values: A string.
    
    Returns:
    A TF-Feature.
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))
       

#train_inputs = train_input_fn()
#test_input_fn = input_fn_maker('mnist_tfrecord/test/',  'mnist_tfrecord/data_info.csv',batch_size = 512,
#                               padding = padding_info)
#train_input_fn = input_fn_maker('mnist_tfrecord/train/',  'mnist_tfrecord/data_info.csv', shuffle=True, batch_size = 128,
#                               padding = padding_info)
#train_eval_fn = input_fn_maker('mnist_tfrecord/train/',  'mnist_tfrecord/data_info.csv', batch_size = 512,
#                               padding = padding_info)
#sess =tf.InteractiveSession()
#print(train_inputs['code'].eval())
#print(train_inputs['date'].eval())
#print(train_inputs['label_type'].eval())
#x = train_inputs['tick'].eval()
#print(x.shape)
#print(x)



# Set up logging for predictions
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int,default=200, help='the number of records of each (t0,t1,t2) type')
    
    args = parser.parse_args()
    
#    data_path = 'G:\\workarea\\scripts\\stock_tfexample\\train'
    num = args.num
    rec_per_file = 333 # actually 333*3
    db_file_list = [get_filenames(x) for x in ['t0', 't1', 't2']]
    
    save_dir = 'G:\\workarea\\scripts\\tfexample{}'.format(num)
    if not tf.gfile.Exists(save_dir):
        tf.gfile.MakeDirs(save_dir)
    save_file_fmt = "rec012_" + str(num) + "_{:0>2d}.tfrecord"
    initLogging(os.path.join(save_dir, "rec012_%s.log"%(str(num))))
    records_option = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    
    count = 0 
    #x = provide_rec(db_file_list[0])  
    save_file = os.path.join(save_dir, save_file_fmt.format(0))
    writer = tf.python_io.TFRecordWriter(save_file, options=records_option)
    xlist = [provide_rec(i) for i in db_file_list] 
    while count < num:
        for gx in xlist:
            try:
                y = gx.__next__()
                rec = tf.train.Example(features=tf.train.Features(feature={
                                              'code':bytes_feature(np.array(y.code)),
                                              'date':bytes_feature(np.array(y.date)),
                                              'label_type': int64_feature(y.label),
                                              'tick': float_feature(np.array(y.tick)),}))
                logging.info("<{}> {} {} {}".format((count+1),y.date, y.code, y.label))
                writer.write(rec.SerializeToString())
            except StopIteration:
                continue
        else:

            if not count % rec_per_file:
                writer.close()
                file_no = count // rec_per_file
                save_file = os.path.join(save_dir, save_file_fmt.format(file_no))
                writer = tf.python_io.TFRecordWriter(save_file, options=records_option)
            count += 1
    
    writer.close()
    
    

if __name__ == '__main__':
    main()


