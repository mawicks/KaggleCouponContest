import codecs
import collections
import csv
import datetime
import logging
import math

# import pandas as pd


logger = logging.getLogger(__name__)
FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT)

logger.setLevel(logging.DEBUG)

def dump(d):
    for k, v in d.items():
        print()
        print('{0}:'.format(k))
        print('      {0}'.format(v))

# translation = dict(pd.read_csv('CAPSULE_TEXT_Translation.csv').values)

def date_mapper(d):
    if d == 'NA':
        timestamp = datetime.datetime.max
    else:
        try: 
            timestamp = datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S')
        except:
            timestamp = datetime.datetime.strptime(d, '%Y-%m-%d')
                        
    return timestamp.date()

def int_mapper(d):
    if d == 'NA':
        return -999
    else:
        return int(d)

def str_mapper(s):
    if s == '':
        return None
    else:
        return s

mapper = collections.defaultdict(lambda: str_mapper)
mapper.update(
    {
        'DISPEND': date_mapper,
        'DISPFROM': date_mapper,
        'I_DATE': date_mapper,
        'REG_DATE': date_mapper,
        'WITHDRAW_DATE': date_mapper,
        
        'VALIDFROM': date_mapper,
        'VALIDEND': date_mapper,

        'AGE': int,
        'CATALOG_PRICE': int,
        'DISCOUNT_PRICE': int,
        'ITEM_COUNT': int,
        'PRICE_RATE': int,

        'LATITUDE': float,
        'LONGITUDE': float,

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
    logger.info ('Loading {0}...'.format(filename))
    with open(filename, "r") as user_file:
        reader = csv.reader(user_file)
        
        # Explicitly ignore any UTF-8 BOM marks left around as characters
        # Using codecs.open wasn't any better.
        header = tuple(map(lambda s: s.replace('\ufeff',''), next(reader)))
        item_mapper = tuple(map(mapper.__getitem__, header))
        
        Type = collections.namedtuple(item_type_name, header)
        globals()[item_type_name] = Type

        if index != None:
            result = dict( (o.__getattribute__(index), o)
                           for o in ( Type._make((m(i) for m,i in zip(item_mapper, line))) for line in reader ) )
        else:
            result = tuple( Type._make((m(i) for m,i in zip(item_mapper, line))) for line in reader )
    logger.info ('{1:,} {2} objects created from {0}'.format(filename, len(result), item_type_name))
    return result

def earth_distance(coordinate1, coordinate2):
    '''coordinates are (lat,lon) tuples in degrees'''
    lat1, lon1 = map(lambda x: x*math.pi/180.0, coordinate1)
    lat2, lon2 = map(lambda x: x*math.pi/180.0, coordinate2)
    distance = math.acos(math.cos(lat1)*math.cos(lat2) * (math.cos(lon1-lon2)-1)
                         + math.cos(lat1 - lat2))
    return distance*4000

prefecture_locations = read_file('prefecture_locations.csv', 'Location', index='PREF_NAME')

def prefecture_distance(pref1, pref2):
    try:
        loc1 = prefecture_locations[pref1]
        loc2 = prefecture_locations[pref2]
        result = earth_distance((loc1.LATITUDE,loc1.LONGITUDE), (loc2.LATITUDE,loc2.LONGITUDE))
    except:
        result = -99999.0

    return result

user = read_file('user_list.csv', 'User', 'USER_ID_hash')
coupon = read_file('coupon_list_train.csv', 'Coupon', 'COUPON_ID_hash')

missing_coupon = Coupon(CAPSULE_TEXT=None,
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

purchase = dict ( (k, v._replace(USER_ID_hash=user[v.USER_ID_hash],
                                  COUPON_ID_hash=coupon[v.COUPON_ID_hash]))
                   for k,v in read_file('coupon_detail_train.csv', 'Purchase', 'PURCHASEID_hash').items() )

visit = tuple( v._replace(USER_ID_hash=user[v.USER_ID_hash],
                           VIEW_COUPON_ID_hash=coupon.get(v.VIEW_COUPON_ID_hash, missing_coupon),
                           PURCHASEID_hash=purchase.get(v.PURCHASEID_hash, None))
                for v in read_file('coupon_visit_train.csv', 'Visit') )

area = read_file('coupon_area_train.csv', 'Area')


prefecture_set = set()
large_area_name_set = set()
ken_name_set = set()
small_area_name_set = set()
capsule_text_set = set()
genre_name_set = set()

# Purchase
purchase_by_user = {}
for p in purchase.values():
    purchase_by_user.setdefault(p.USER_ID_hash.USER_ID_hash, []).append(p)

# Visit
visit_by_user = {}
for v in visit:
    visit_by_user.setdefault(v.USER_ID_hash.USER_ID_hash, []).append(v)

# Area
area_by_coupon = {}
for a in area:
    area_by_coupon.setdefault(a.COUPON_ID_hash, []).append(a)
    prefecture_set.add(a.PREF_NAME)
    small_area_name_set.add(a.SMALL_AREA_NAME)

# Coupon
for h,c in coupon.items():
    capsule_text_set.add(c.CAPSULE_TEXT)
    genre_name_set.add(c.GENRE_NAME)
    large_area_name_set.add(c.large_area_name)
    ken_name_set.add(c.ken_name)
    small_area_name_set.add(c.small_area_name)

# User
user_history = {}
UserHistory = collections.namedtuple('UserHistory', ['user', 'visit', 'purchase'])
for h,u in user.items():
    user_history[h] = UserHistory(user=u, visit=visit_by_user.get(h, []), purchase=purchase_by_user.get(h, []))
    prefecture_set.add(u.PREF_NAME)

def log_set(logger, s, name):
    logger.info('{0} ({1}): {2}'.format(name, len(s), ', '.join(sorted(map(str, s)))))
        
log_set(logger, prefecture_set, 'Prefecture names')
log_set(logger, large_area_name_set, 'Large area names')
log_set(logger, ken_name_set, 'Ken names')
log_set(logger, capsule_text_set, 'Capsule text')
log_set(logger, genre_name_set, 'Genre names')
log_set(logger, small_area_name_set, 'Small area names')

class CategoryEncoder:
    def __init__ (self, categories):
        enumerated = tuple(enumerate(categories))
        self.mapping = dict((c,i) for i,c in enumerated)
        self.unmapping = tuple(zip(*enumerated))[1]
        print(self.mapping)
        print(self.unmapping)

    def map(self, category):
        return self.mapping[category]

    def unmap(self, value):
        return self.unmapping[int(value)]

prefecture_encoder = CategoryEncoder(prefecture_set)
large_area_name_encoder = CategoryEncoder(large_area_name_set)
capsule_encoder = CategoryEncoder(capsule_text_set)
genre_name_encoder = CategoryEncoder(genre_name_set)
small_area_name_encoder = CategoryEncoder(small_area_name_set)
gender_encoder = CategoryEncoder(('m', 'f'))

class SimpleUserFeatureSet:
    def names(self):
        return ('age', 'gender', 'prefecture', 'days_as_member')

    def map (self, user_history, coupon, date):
        user = user_history.user
        return (user.AGE,
                gender_encoder.map(user.SEX_ID),
                prefecture_encoder.map(user.PREF_NAME),
                (date - user.REG_DATE).days)

class SimpleCouponFeatureSet:
    def names(self):
        return ('capsule_text', 'genre_name',
                'large_area_name', 'ken_name', 'small_area_name',
                'price_rate', 'catalog_price', 'discount_price', 'valid_period',
                'days_on_display', 'display_days_left',
                'days_until_valid', 'days_until_experation')

    def map (self, user_history, coupon, date):
        return (capsule_encoder.map(coupon.CAPSULE_TEXT),
                genre_name_encoder.map(coupon.GENRE_NAME),
                large_area_name_encoder.map(coupon.large_area_name),
                prefecture_encoder.map(coupon.ken_name),
                small_area_name_encoder.map(coupon.small_area_name),
                coupon.PRICE_RATE,
                coupon.CATALOG_PRICE,
                coupon.DISCOUNT_PRICE,
                coupon.VALIDPERIOD,
                (date - coupon.DISPFROM).days,
                (coupon.DISPEND - date).days,
                (coupon.VALIDFROM - date).days,
                (coupon.VALIDEND - date).days)

class JointFeatureSet:
    def names(self):
        return ('distance',)
    
    def map (self, user_history, coupon, date):
        return (prefecture_distance(user_history.user.PREF_NAME, coupon.ken_name))
    
# dump(user_history)

feature_set = SimpleUserFeatureSet()

logger.info('Feature names: {0}'.format(', '.join(feature_set.names())))
logger.info('Features:      {0}'.format(feature_set.map(user_history['280f0cedda5c4b171ee6245889659571'],
                                                        coupon['31a605db6db5ad3fa3b2d4cf69ae3272'],
                                                        datetime.date(year=2012, month=5, day=10))))

feature_set = SimpleCouponFeatureSet()
logger.info('Feature names: {0}'.format(', '.join(feature_set.names())))
logger.info('Features:      {0}'.format(feature_set.map(user_history['280f0cedda5c4b171ee6245889659571'],
                                                        coupon['31a605db6db5ad3fa3b2d4cf69ae3272'],
                                                        datetime.date(year=2012, month=5, day=10))))
feature_set = JointFeatureSet()
logger.info('Feature names: {0}'.format(', '.join(feature_set.names())))
logger.info('Features:      {0}'.format(feature_set.map(user_history['280f0cedda5c4b171ee6245889659571'],
                                                        coupon['31a605db6db5ad3fa3b2d4cf69ae3272'],
                                                        datetime.date(year=2012, month=5, day=10))))

logger.debug('Done.')
