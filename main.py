import collections
import csv
import datetime

import logging

import pandas as pd


logger = logging.getLogger(__name__)
FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT)

logger.setLevel(logging.DEBUG)

translation = dict(pd.read_csv('CAPSULE_TEXT_Translation.csv').values)

def timestamp_mapper(d):
    if d == 'NA':
        return datetime.datetime.max
    else:
        return datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S')

def date_mapper(d):
    if d == 'NA':
        return datetime.datetime.max
    else:
        return datetime.datetime.strptime(d, '%Y-%m-%d')

def int_mapper(d):
    if d == 'NA':
        return None
    else:
        return int(d)

def str_mapper(s):
    if s == '':
        return None
    else:
        return s

mappers = collections.defaultdict(lambda: str_mapper)
mappers.update(
    {
        'DISPEND': timestamp_mapper,
        'DISPFROM': timestamp_mapper,
        'I_DATE': timestamp_mapper,
        'REG_DATE': timestamp_mapper,
        'WITHDRAW_DATE': timestamp_mapper,
        
        'VALIDFROM': date_mapper,
        'VALIDEND': date_mapper,

        'AGE': int,
        'CATALOG_PRICE': int,
        'DISCOUNT_PRICE': int,
        'ITEM_COUNT': int,
        'PRICE_RATE': int,

        'DISPPERIOD': int_mapper,
        'PAGE_SERIAL': int_mapper,
        'PURCHASE_FLG': int_mapper,
        'USABLE_DATE_MON': int_mapper,
        'USABLE_DATE_TUE': int_mapper,
        'USABLE_DATE_WED': int_mapper,
        'USABLE_DATE_THU': int_mapper,
        'USABLE_DATE_FRI': int_mapper,
        'USABLE_DATE_SAT': int_mapper,
        'USABLE_DATE_SUN': int_mapper,
        'USABLE_DATE_HOLIDAY': int_mapper,
        'USABLE_DATE_BEFORE_HOLIDAY': int_mapper,
        'VALIDPERIOD': int_mapper
    }
)

def read_file (filename, item_type_name, index=None):
    logger.debug ('Reading {0}'.format(filename))
    with open(filename) as user_file:
        reader = csv.reader(user_file)
        
        header = next(reader)
        item_mappers = tuple(map(mappers.__getitem__, header))
        
        Type = collections.namedtuple(item_type_name, header)
        globals()[item_type_name] = Type

        if index != None:
            return dict( (o.__getattribute__(index), o)
                         for o in ( Type._make((m(i) for m,i in zip(item_mappers, line))) for line in reader ) )
        else:
            return tuple( Type._make((m(i) for m,i in zip(item_mappers, line))) for line in reader )

users = read_file('user_list.csv', 'Users', 'USER_ID_hash')
coupons = read_file('coupon_list_train.csv', 'Coupons', 'COUPON_ID_hash')

missing_coupon = Coupons(CAPSULE_TEXT=None,
                         GENRE_NAME=None,
                         PRICE_RATE=None,
                         CATALOG_PRICE=None,
                         DISCOUNT_PRICE=None,
                         DISPFROM=None,
                         DISPEND=None,
                         DISPPERIOD=None,
                         VALIDFROM=None,
                         VALIDEND=None,
                         VALIDPERIOD=None,
                         USABLE_DATE_MON=None,
                         USABLE_DATE_TUE=None,
                         USABLE_DATE_WED=None,
                         USABLE_DATE_THU=None,
                         USABLE_DATE_FRI=None,
                         USABLE_DATE_SAT=None,
                         USABLE_DATE_SUN=None,
                         USABLE_DATE_HOLIDAY=None,
                         USABLE_DATE_BEFORE_HOLIDAY=None,
                         large_area_name=None,
                         ken_name=None,
                         small_area_name=None,
                         COUPON_ID_hash='*MISSING*')

purchases = dict ( (k, v._replace(USER_ID_hash=users[v.USER_ID_hash],
                                  COUPON_ID_hash=coupons[v.COUPON_ID_hash]))
                   for k,v in read_file('coupon_detail_train.csv', 'Purchases', 'PURCHASEID_hash').items() )
# Make sure None maps to None
purchases[None] = None    

views = tuple( v._replace(USER_ID_hash=users[v.USER_ID_hash],
                          VIEW_COUPON_ID_hash=coupons.get(v.VIEW_COUPON_ID_hash, missing_coupon),
                          PURCHASEID_hash=purchases[v.PURCHASEID_hash])
               for v in read_file('coupon_visit_train.csv', 'Visits') )

for v in views:
    print(v)

logger.debug('Done.')

def dump(d):
    for k, v in d.items():
        print()
        print('{0}:'.format(k))
        print('      {0}'.format(v))
