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
from collections import namedtuple

REC_DB_PATH = 'G:\\workarea\\scripts\\stock_tfexample\\train'
SRECORD = namedtuple('REC',"date high close tick")    


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
    if cate in ("t0", "t1", "t2"):
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
        


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rtype', help='example: t0|t2|t1')
    parser.add_argument('--path', default=REC_DB_PATH, help='example: t0|t2|t1')
    
    args = parser.parse_args()
    
#    data_path = 'G:\\workarea\\scripts\\stock_tfexample\\train'
    db_file_list = get_filenames(args.rtype)
    
    
    count = 0 
    for y in iter(provide_rec(db_file_list)):
        count += 1
        print("<{}> {} {} {}".format(count, y.date, y.code, y.label))
        
    print("Got {} records finally!!".format(count))
        
    
    
 

if __name__ == '__main__':
    main()


