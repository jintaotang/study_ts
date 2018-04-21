# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 18:20:16 2017

@author: ziwenruo
"""
#from glob import glob
#import os
import json
import sys
import pandas as pd
import pprint


import tushare as ts
cons = ts.get_apis()

import pymongo
#from pymongo import MongoClient

#end_day = '2017-11-07'
sta_day = '2016-01-01'
#sta_day = '2018-11-08'
end_day = '2018-04-01'

client = pymongo.MongoClient('localhost', 27017)
db = client['A-SHARE']
db_stock_basics = client['STOCK_BASICS']['2018-03-19'.replace('-', '')]


#create stock basics DB
'''
tar_stock_basics = ts.get_stock_basics(date=end_day)
db_stock_basics.insert(json.loads(tar_stock_basics.to_json(orient='records')))
'''


ava_stock = db.collection_names()

'''
def get_collection_status(collection):
    stemp = pd.Series()
    for c in collection:
        sc = pd.Series([db[c].count()], index=[c])
        stemp = stemp.append(sc, verify_integrity=True)
        
    return stemp

collection_status = get_collection_status(ava_stock)

choosen_col = collection_status[collection_status>365]
def sum_outstanding(s_list):
    stemp = pd.Series()
    for s in s_list:
        sinfo = db_stock_basics.find_one({'code':s})
        sc = pd.Series(sinfo['outstanding'], index=[s])
        stemp = stemp.append(sc, verify_integrity=True)
    
    return stemp

get_outstanding_summary = sum_outstanding(choosen_col.index)


s = '300508'

def do_tick_data_pre(s):
    apd_vol = get_outstanding_summary[s]
    a = db[s].find_one()
    apd = pd.read_json(a.get('tick'), orient='index')
    apd.sort_index(inplace=True)
    apd['price_shift1'] = apd['price'].shift()
    apd['delta_price'] = apd['price'] - apd['price_shift1']       
    apd['delta_vol'] = apd['vol']/(100.0*apd_vol)
    apd.fillna(value=0, inplace=True)
'''
#logFilename = os.path.join(afeature.data_dir, 'gen_stock_tfrecord.log')
#initLogging(logFilename)

def build_index(db_doc, u_key):
    index_dic = db_doc.index_information()
    if u_key + '_1' not in index_dic.keys():
        db_doc.create_index([(u_key, pymongo.ASCENDING)], unique=True)
        
def build_tickdb():
    stock_count = db_stock_basics.count()
    i = 0
    for s_code in db.collection_names():
        k_data = ts.get_k_data(code=s_code, start=sta_day, end=end_day)
        i += 1
        try:
            try:
                build_index(db[s_code], 'date')
            except pymongo.errors.DuplicateKeyError:
#                db[s_code].drop()
                build_index(db[s_code], 'date')
            db[s_code].insert_many(json.loads(k_data.to_json(orient='records')))
           
        except TypeError:
            print('{} passed, maybe the server banned me!'.format(s_code))
        except pymongo.errors.BulkWriteError:
            print('%s records exist!' % s_code)

        for post_d in db[s_code].find({'tick' : {"$exists":False}}):
            day = post_d['date']
            tk = ts.tick(s_code, conn=cons, date=day)
            try:
                db[s_code].update_one({'date':day}, {"$set":{"tick":tk.to_json(orient='index')}})
            except:
                print("WARNING, processing {} {} failed".format(s_code, day))
                continue
            print('processed {} {}...{{{:>4d}:{}}}'.format(s_code, day,i,stock_count))


#        
        
if __name__=='__main__':
    
    build_tickdb()
#for s_code in ava_stock:
#    build_index(db[s_code], 'date')        
    

''' for s_code in ava_stock:
    count = ava_stock.index(s_code)
    status = (100.0*count)/len(ava_stock)
    cur = db[s_code].find({'tick' : {"$exists":False}})
#    print("searching status: {:.2f}%".format(status))
    sys.stdout.write("\rsearching status: {:.2f}%\r".format(status))
    sys.stdout.flush()
    if cur.count():
        for d in cur:
            day = d['date']
            tk = ts.tick(s_code, conn=cons, date=day)
            try:
                db[s_code].update_one({'date': day}, {"$set":{"tick":tk.to_json(orient='index')}})
            except:
                print("WARNING, processing {} {} failed".format(s_code, day))
                break
            print("processed {} {} {:.2f}%".format(s_code, day, status)) '''
    
            
#db_cursor = db_stock_basics.find({}, {'code':1})
#for _ in range(3):
#    i = db_cursor.next()
#    s_code = i['code']
#    k_data = ts.get_k_data(code=s_code, start=sta_day, end=end_day)
#    dbsc = db[s_code]
#    for day in k_data['date']:
#        tick_data = ts.tick(s_code, conn=cons, date=day)
    
        

#code = ['000001', '000002']
#sdata_path = "d:\\workarea\stock_lst"
#
#tick_data_path = os.path.join(sdata_path, 'tick_data_tt')
#
#get_data_file_list = lambda scode: glob(os.path.join(tick_data_path, scode, "*.xlsx"))
#
#
#def data2json(file_path):
#    the_day = os.path.basename(file_path).split('.')[0]
#    if not the_day.startswith('~$'):
#        the_xlsx = pd.read_excel(file_path)
#        the_json = the_xlsx.to_json()
#        return the_day, {'tick_data': the_json}
#
#for scode in code:
#    for xlsxf in get_data_file_list(scode):
#        tdate, tdata = data2json(xlsxf)
#        a_collection = db[scode]
#        if a_collection.find_one({'date':tdate}):
#            a_collection.update_one({'$set' : {'tick_data':tdata}})
#        else:
#            a_collection.insert_one({'date': tdate, 
#                                 'tick_data': tdata})
#        