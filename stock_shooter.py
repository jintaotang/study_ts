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

from nets import inception_v4

parser = argparse.ArgumentParser()
parser.add_argument('--train', action="store_true", help='trainning mode')
parser.add_argument('--predict', action="store_true", help='predicting mode')
parser.add_argument('--eval', action="store_true", help='evaluation mode')
parser.add_argument('--ckrec', action="store_true", help='check_record')
parser.add_argument('--data', default='G:\\workarea\\scripts\\tfexample19500', help='data path')



record_info = pd.DataFrame({
        'name':['code','date','label_type','tick'],
        'type':[tf.string,tf.string,tf.int64,tf.float32],
        'shape':[(),(),(),(299,299,3)],
        'isbyte':[True,True,False,False],
        'length_type':['fixed','fixed','fixed','var'],
        }, columns=['name','type','shape','isbyte','length_type','default'])

def get_filenames(path,shuffle=True):
    # get all file names 
    files= os.listdir(path) 
    filepaths = [os.path.join(path,file) for file in files if not os.path.isdir(file) and '.tfrecord' in file]
    # shuffle
    if shuffle:
        ri = np.random.permutation(len(filepaths))
        filepaths = np.array(filepaths)[ri]
    return filepaths
    
    
def create_parser(data_info):
    
    names = data_info['name']
    types = data_info['type']
    shapes = data_info['shape']
    isbytes = data_info['isbyte']
#    defaults = data_info['default']
    length_types = data_info['length_type']
    
    def cparser(example_proto):
        
        def specify_features():
            specified_features = {}
            for i in np.arange(len(names)):
                # which type
                if isbytes[i]:
                    t = tf.string
                    s = ()
                else:
                    t = types[i]
                    s = shapes[i]
                # has default_value?
#                if defaults[i] == np.NaN:
#                    d = np.NaN
#                else:
#                    d = defaults[i]
                # length varies
                if length_types[i] =='fixed':
                    specified_features[names[i]] = tf.FixedLenFeature(s, t)
                elif length_types[i] =='var':
                    specified_features[names[i]] = tf.VarLenFeature(t)
                else:
                    raise TypeError("length_type is not one of 'var', 'fixed'")
            return specified_features
                        # decode each parsed feature and reshape
        def decode_reshape():
            # store all decoded&shaped features
            final_features = {}
            for i in np.arange(len(names)):
                # exclude shape info
                if 'shape' not in names[i]:
                    # decode
                    if isbytes[i]:
                        # from byte format
#                        decoded_value = tf.decode_raw(parsed_example[names[i]], types[i])
                        decoded_value = parsed_example[names[i]]
                    else:
                        # Varlen value needs to be converted to dense format
                        if length_types[i] == 'var':
                            decoded_value = tf.sparse_tensor_to_dense(parsed_example[names[i]])
                        else:
                            decoded_value = parsed_example[names[i]]

                    if shapes[i]:
                         decoded_value = tf.reshape(decoded_value, shapes[i]) 
                    final_features[names[i]] = decoded_value
            return final_features
        
        
        # create a dictionary to specify how to parse each feature 
        specified_features = specify_features()
        # parse all features of an example
        parsed_example = tf.parse_single_example(example_proto, specified_features)
        final_features = decode_reshape()
        
        
        
        return final_features
    return cparser
            
def get_dataset(paths, data_info, shuffle = False, shuffle_buffer=10000, batch_size = 1, epoch = 1, padding = None):
    
    filenames = paths
    print(data_info)
    print('read dataframe from {}'.format(filenames))
#    data_info['shape']=data_info['shape'].apply(lambda s: [int(i) for i in s[1:-1].split(',') if i !=''])
    shuffle = shuffle
    shuffle_buffer = shuffle_buffer
    batch_size = batch_size
    epoch = epoch
    padding = padding
    dataset = tf.data.TFRecordDataset(filenames, compression_type='GZIP')
   
    parse_function = create_parser(data_info)
    dataset = dataset.map(parse_function)
#    dataset_raw = dataset
    if shuffle:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)
    if padding is not None:
        dataset = dataset.padded_batch(batch_size, padded_shapes = padding)
    else:
        dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(epoch)
    return dataset

def input_fn_maker(path, data_info, shuffle=False, batch_size = 2, epoch = 100, padding = None):
    def input_fn():
        filenames = get_filenames(path=path, shuffle=shuffle)
        dataset=get_dataset(paths=filenames, data_info=data_info, shuffle = shuffle, 
                            batch_size = batch_size, epoch = epoch, padding =padding)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()
    return input_fn


def check_records(fn):
    a = fn()
    mytick = a['tick']
    with tf.Session() as sess:
        atick = mytick.eval()
    return atick[0]


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

def model_fn(features, mode):
    logits, endpoints = inception_v4.inception_v4(features['tick'], num_classes=3)

    predictions = {
        "code":features['code'],
        "tick":features['tick'],
        "logits":logits,
        "classes": tf.argmax(input=logits, axis=1),
        "labels": features['label_type'],
        "probabilities": endpoints['Predictions']
        }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=features['label_type'], logits=logits)
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        accuracy = tf.metrics.accuracy(labels=features['label_type'], predictions=tf.argmax(logits, axis=1))
        # Name the accuracy tensor 'train_accuracy' to demonstrate the
        # LoggingTensorHook.
        tf.identity(accuracy[1], name='train_accuracy')
        train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=0, loss=loss, train_op=train_op)
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=features['label_type'], predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

# Set up logging for predictions
def main():
    
    args = parser.parse_args()
    
#    data_path = 'G:\\workarea\\scripts\\stock_tfexample\\train'
    data_path = args.data
    
    train_input_fn = input_fn_maker(data_path, record_info, batch_size=3, epoch=500)

    test_input_fn = input_fn_maker(data_path, record_info)
    pred_input_fn = input_fn_maker(data_path, record_info, epoch = 1)
    
    tf.logging.set_verbosity(tf.logging.INFO)
    tensors_to_log = {"train_accuracy": 'train_accuracy'}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log,
                                              every_n_secs=15,
    ##                                          every_n_iter=1,
    ##                                          at_end=True
                                              )
    
    
    
    session_config = tf.ConfigProto()
    #session_config.gpu_options.per_process_gpu_memory_fraction = 0.5
    session_config.gpu_options.allow_growth = True
    estimator_config = tf.estimator.RunConfig(session_config=session_config)
    
    stock_shooter = tf.estimator.Estimator(
        model_fn=model_fn, model_dir="G:\\workarea\\lssp19500_inception_v4",config=estimator_config)
    
    #Trainning
    if args.train:        
        stock_shooter.train(
            input_fn=train_input_fn,
            hooks=[logging_hook]
            )
    elif args.eval:
        eval_results = stock_shooter.evaluate(input_fn=test_input_fn)
        print('train set')
        print(eval_results)  
    elif args.predict:
        import pdb
        predicts =list(stock_shooter.predict(input_fn=pred_input_fn))
        pdb.set_trace()
        print()
        print(predicts[0].keys())        
    elif args.ckrec:
        import pdb
        x = check_records(train_input_fn)
        pdb.set_trace()
        print(x)

if __name__ == '__main__':
    main()


