# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 14:21:22 2018

@author: ziwenruo
"""

import pymongo
import json
import tushare as ts
import pandas as pd
from collections import namedtuple
from datetime import date
cons = ts.get_apis()

# from pymongo import MongoClient

end_day = date.today().isoformat()
sta_day = '2016-01-01'

client = pymongo.MongoClient('localhost', 27017)
db = client['A-SHARE']
db_stock_basics = client['STOCK_BASICS']['2018-03-19'.replace('-', '')]


def dailyiter(d, cols=None):
    if cols is None:
        v = d.values.tolist()
        cols = d.columns.values.tolist()
    else:
        j = [d.columns.get_loc(c) for c in cols]
        v = d.values[:, j].tolist()

    n = namedtuple('SDR', cols)

    for line in iter(v):
        yield n(*line)


# for i in range(0, len(df)):
#    print df.iloc[i]
def compdays(tarday, refday=sta_day):
    day_gap = pd.to_datetime(tarday) - pd.to_datetime(refday)
    return day_gap.days > 0


def create_stockbasicdb(tar_day):
    tar_stock_basics = ts.get_stock_basics(date=tar_day)
    db_stock_basics.insert(
        json.loads(tar_stock_basics.to_json(orient='records')))


def build_index(db_doc, u_key):
    index_dic = db_doc.index_information()
    if u_key + '_1' not in index_dic.keys():
        db_doc.create_index([(u_key, pymongo.ASCENDING)], unique=True)


def build_tickdb(stock_list):

    stock_count = len(stock_list)
    count = 0
    for s_code in stock_list:
        count += 1
        print('Start processing {}...{{{:>4d}:{}}}'.format(
            s_code, count, stock_count))
        # first check wheather indexed
        idxinfo = db[s_code].index_information()
        if not (len(idxinfo) == 2 and 'date_1' in idxinfo.keys()):
            print("No index info found for {}:{}".format(
                s_code, idxinfo.keys()))
            build_index(db[s_code], 'date')

        k_data = ts.get_k_data(code=s_code)
        if not k_data.empty:
            try_count = 0
            for i in range(1, len(k_data) + 1):
                daily_record = k_data.iloc[-i]
                if compdays(daily_record.date) and try_count < 40:
                    if not db[s_code].find_one({
                            'date': daily_record.date
                    }, {'_id': 0}):
                        db[s_code].insert_one(
                            json.loads(daily_record.to_json()))
                        print('{:13s} {}...{{{:>4d}:{}}}'.format(
                            'insert K data ' + s_code, daily_record.date,
                            count, stock_count))
                    else:
                        try_count += 1
                else:
                    break
        else:
            print('skipping, {}, no k-data got'.format(s_code))
            continue

        for post_d in db[s_code].find({'tick': {"$exists": False}}):
            day = post_d['date']
            tk = ts.tick(s_code, conn=cons, date=day)
            try:
                if not tk.empty:
                    db[s_code].update_one(
                        {
                            'date': day
                        }, {"$set": {
                            "tick": tk.to_json(orient='index')
                        }})
                    print('{:13s} {}...{{{:>4d}:{}}}'.format(
                        'processing ...' + s_code, day, count, stock_count))
            except AttributeError:
                print('skipping, {}, no tick-data got on {}'.format(
                    s_code, day))


def tick_data_check(stock_list):
    stock_count = len(stock_list)
    count = 0
    for s_code in stock_list:
        count += 1
        print('Start processing {}...{{{:>4d}:{}}}'.format(
            s_code, count, stock_count))
        for post_d in db[s_code].find():
            day = post_d['date']
            if 'tick' not in post_d.keys() or day not in post_d['tick']:
                tk = ts.tick(s_code, conn=cons, date=day)
                try:
                    if not tk.empty:
                        db[s_code].update_one(
                            {
                                'date': day
                            }, {"$set": {
                                "tick": tk.to_json(orient='index')
                            }})
                        print('{:13s} {}...{{{:>4d}:{}}}**'.format(
                            'processing ...' + s_code, day, count,
                            stock_count))
                except AttributeError:
                    print('skipping, {}, no tick-data got on {}'.format(
                        s_code, day))


def main():
    stock_list = db.collection_names()
    build_tickdb(stock_list)


#    tick_data_check(stock_list)

if __name__ == '__main__':
    main()
