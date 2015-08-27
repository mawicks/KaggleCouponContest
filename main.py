import codecs
import collections
import csv
import datetime
from functools import reduce
import itertools
import logging
import operator
import naive_bayes
import math
import numpy
import random
import sklearn
import sklearn.ensemble
import sklearn.ensemble.weight_boosting
import sys

# Tunable parameters
NPP=1 # Negative training cases per positive cases.
n_positive = 120000 # Number of postive training cases.
# n_estimators = 4000
n_estimators = 4000
# min_samples_leaf = 1 + int(N/4000)
min_samples_leaf = 5
max_features = 9
n_jobs=-1
oob_score=False
seed=12345678

# Random number seeds
random_state = random.Random(seed)
classifier_random_state = numpy.random.RandomState(seed=seed)


# Important constants
train_start_date = datetime.datetime(year=2011, month=7, day=3, hour=0, minute=0)
# train_start_date = datetime.datetime(year=2012, month=1, day=1, hour=0, minute=0)
test_week_start_date = datetime.datetime(year=2012, month=6, day=24)

train_period_in_weeks = (test_week_start_date - train_start_date).days // 7 


class WrappedClassifier:
    """Wrap the brain-dead classifier API to make it look more like a regressor."""
    def __init__(self, classifier):
        self.classifier = classifier

    def fit(self, x, y):
        self.classifier.fit(x, y)
        #        self.oob_score_ = self.classifier.oob_score_

        if hasattr(self.classifier, "oob_decision_function_"):
            self.oob_prediction_ = self.classifier.oob_decision_function_[:,1].reshape(len(y))

        if hasattr(self.classifier, "feature_importances_"):
            self.feature_importances_ = self.classifier.feature_importances_

    def predict(self, x):
        return self.classifier.predict_proba(x)[:,1]

    def score(self, x, y):
        return self.classifier.score(x, y)

regressors = ( 
    ('RandomForestRegressor',
     sklearn.ensemble.RandomForestRegressor(
         random_state=classifier_random_state,
         n_estimators=n_estimators,
         min_samples_leaf=min_samples_leaf,
         max_features=max_features,
         n_jobs=n_jobs,
         oob_score=oob_score
     )),

    ('RandomForestClassifier',
     WrappedClassifier(sklearn.ensemble.RandomForestClassifier(
         random_state=classifier_random_state,
         n_estimators=n_estimators,
         min_samples_leaf=min_samples_leaf,
         max_features=max_features,
         n_jobs=n_jobs,
         criterion='entropy',
         oob_score=oob_score
     ))),

    ('GradientBootingRegressor',
     sklearn.ensemble.gradient_boosting.GradientBoostingRegressor(
         random_state=classifier_random_state,
         n_estimators=n_estimators,
         min_samples_leaf=min_samples_leaf,
         max_features=max_features)),
    
    ('GradientBoostingClassifier',
     WrappedClassifier(sklearn.ensemble.gradient_boosting.GradientBoostingClassifier(
         random_state=classifier_random_state,
         n_estimators=n_estimators,
         min_samples_leaf=min_samples_leaf,
         max_features=max_features))),

    ('AdaboostRegressor',
     sklearn.ensemble.weight_boosting.AdaBoostRegressor(
         n_estimators=n_estimators,
     )),
    
    ('AdaboostClassifier',
     WrappedClassifier(sklearn.ensemble.weight_boosting.AdaBoostClassifier(
         n_estimators=n_estimators,
     ))),

)

# For now, don't use boosting methods
regressors = regressors[0:1]

def week_index(date):
    return int((date - train_start_date).days / 7)

def start_of_week(date):
    return train_start_date + week_index(date)*datetime.timedelta(days=7)

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

def date_mapper_max(d):
    if d == 'NA':
        timestamp = datetime.datetime.max
    else:
        try: 
            timestamp = datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S')
        except:
            timestamp = datetime.datetime.strptime(d, '%Y-%m-%d')
                        
    return timestamp

def date_mapper_min(d):
    if d == 'NA':
        timestamp = datetime.datetime.min
    else:
        try: 
            timestamp = datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S')
        except:
            timestamp = datetime.datetime.strptime(d, '%Y-%m-%d')

    return timestamp

def int_mapper(d):
    if d == 'NA':
        return -999
    else:
        return int(d)

def usable_date_mapper(d):
    # usable_date fields have 0, 1, 2, or NA.  We'll treat this as a two-bit field and represent NA by 3.
    if d == 'NA':
        return 3
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
        'DISPEND': date_mapper_max,
        'DISPFROM': date_mapper_min,
        'I_DATE': date_mapper_max,
        'REG_DATE': date_mapper_min,
        'WITHDRAW_DATE': date_mapper_max,

        'VALIDFROM': date_mapper_min,
        'VALIDEND': date_mapper_max,

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
        'USABLE_DATE_MON': usable_date_mapper,
        'USABLE_DATE_TUE': usable_date_mapper,
        'USABLE_DATE_WED': usable_date_mapper,
        'USABLE_DATE_THU': usable_date_mapper,
        'USABLE_DATE_FRI': usable_date_mapper,
        'USABLE_DATE_SAT': usable_date_mapper,
        'USABLE_DATE_SUN': usable_date_mapper,
        'USABLE_DATE_HOLIDAY': usable_date_mapper,
        'USABLE_DATE_BEFORE_HOLIDAY': usable_date_mapper,
        'VALIDPERIOD': int_mapper
    }
)

def read_file (filename, index=None):
    logger.info ('Loading {0}...'.format(filename))
    with open(filename, "r") as user_file:
        reader = csv.reader(user_file)

        # Explicitly ignore any UTF-8 BOM marks left around as characters
        # Using codecs.open wasn't any better.
        header = tuple(map(lambda s: s.replace('\ufeff',''), next(reader)))

        generator = (dict((h,mapper[h](i)) for h,i in zip(header, line))
                     for line in reader)
        
        if index != None:
            result = collections.OrderedDict(
                (o[index], o)
                for o in generator
            )
        else:
            result = list(generator)
            
    logger.info ('{1:,} records loaded from {0}'.format(filename, len(result)))
    return result

def earth_distance(coordinate1, coordinate2):
    '''coordinates are (lat,lon) tuples in degrees'''
    lat1, lon1 = map(lambda x: x*math.pi/180.0, coordinate1)
    lat2, lon2 = map(lambda x: x*math.pi/180.0, coordinate2)
    distance = math.acos(math.cos(lat1)*math.cos(lat2) * (math.cos(lon1-lon2)-1)
                         + math.cos(lat1 - lat2))
    return distance*4000

def prefecture_distance(pref1, pref2):
    try:
        loc1 = prefecture_locations[pref1]
        loc2 = prefecture_locations[pref2]
        result = float(int(earth_distance((loc1['LATITUDE'],loc1['LONGITUDE']), (loc2['LATITUDE'],loc2['LONGITUDE']))))
    except:
        result = -99999.0

    return result

logger.info('Parameters: '
            'NPP: {0:,}, '
            'n_positive: {1:,}, '
            'n_estimators: {2:,}, '
            'min_samples_leaf: {3}, '
            'max_features: {4}, '
            'n_jobs: {5}'.format(NPP,
                                 n_positive,
                                 n_estimators,
                                 min_samples_leaf,
                                 max_features,
                                 n_jobs
                             ))

logger.info('train_period_in_weeks: {0}'.format(train_period_in_weeks))

# prefecture_locations = read_file('prefecture_locations.csv', index='PREF_NAME')

user = read_file('user_list.csv', 'USER_ID_hash')
logger.info('Sample user: {0}'.format(list(itertools.islice(user.values(), 0, 1))[0]))

def coupon_computed_fields(coupon_list):
    # Add some computed fields in the coupon records
    for h,c in coupon_list.items():
    # Quantized various prices to make it easier to use Naive Bayers
        try:
            c['QUANTIZED_PRICE_RATE'] = int((c['PRICE_RATE'] // 10) * 10.0)
            c['QUANTIZED_DISCOUNT_PRICE'] = int(2 ** (int(math.log2(0.01 + c['DISCOUNT_PRICE']) * 2.0) / 2.0))
            c['QUANTIZED_CATALOG_PRICE'] = int(2 ** (int(math.log2(0.01 + c['CATALOG_PRICE']) * 2.0) / 2.0))
        except:
            print(c['PRICE_RATE'])
            print(c['DISCOUNT_PRICE'])
            print(c['CATALOG_PRICE'])
            raise
    
coupon = read_file('coupon_list_train.csv', 'COUPON_ID_hash')
coupon_computed_fields(coupon)


for c in list(itertools.islice(coupon.values(), 0, 3)):
    logger.info('Sample coupon: {0}'.format(c))

missing_coupon = dict([('CAPSULE_TEXT', None),
                       ('GENRE_NAME', None),
                       ('PRICE_RATE', -999),
                       ('CATALOG_PRICE', -999),
                       ('DISCOUNT_PRICE', -999),
                       ('DISPFROM', datetime.datetime.min),
                       ('DISPEND', datetime.datetime.max),
                       ('DISPPERIOD', (datetime.datetime.max-datetime.datetime.min).days),
                       ('VALIDFROM', datetime.datetime.min),
                       ('VALIDEND', datetime.datetime.max),
                       ('VALIDPERIOD', (datetime.datetime.max-datetime.datetime.min).days),
                       ('USABLE_DATE_MON', 0),
                       ('USABLE_DATE_TUE', 0),
                       ('USABLE_DATE_WED', 0),
                       ('USABLE_DATE_THU', 0),
                       ('USABLE_DATE_FRI', 0),
                       ('USABLE_DATE_SAT', 0),
                       ('USABLE_DATE_SUN', 0),
                       ('USABLE_DATE_HOLIDAY', 0),
                       ('USABLE_DATE_BEFORE_HOLIDAY', 0),
                       ('large_area_name', None),
                       ('ken_name', None),
                       ('small_area_name', None),
                       ('COUPON_ID_hash', '*MISSING*'),
                       ('QUANTIZED_PRICE_RATE', -999),
                       ('QUANTIZED_DISCOUNT_PRICE', -999),
                       ('QUANTIZED_CATALOG_PRICE', -999),
                   ])

purchase = read_file('coupon_detail_train.csv', 'PURCHASEID_hash')
train_area = read_file('coupon_area_train.csv')
test_area = read_file('coupon_area_test.csv')

# Dereference user and coupon references and remove records outside the date range
for k,v in purchase.items():
    if v['I_DATE'] >= train_start_date:
        v['USER'] = user[v['USER_ID_hash']]
        v['COUPON'] = coupon[v['COUPON_ID_hash']]
        del v['USER_ID_hash'], v['COUPON_ID_hash']
    else:
        del purchase[k]
        
logger.info('Retained {0:,} purchase records between {1} and {2}'.format(
    len(purchase),
    min(p['I_DATE'] for p in purchase.values()),
    max(p['I_DATE'] for p in purchase.values())))

# Dereference user and purchase references
visit = read_file('coupon_visit_train.csv')

missing_coupons = set()
found_coupons = set()

for v in visit:
    coupon_id_hash = v['VIEW_COUPON_ID_hash']
    if coupon_id_hash in coupon:
        v['COUPON']= coupon[coupon_id_hash]
        found_coupons.add(coupon_id_hash)
    else:
        v['COUPON']= missing_coupon
        missing_coupons.add(coupon_id_hash)

    user_id_hash = v['USER_ID_hash']
    if user_id_hash in user:
        v['USER']= user[user_id_hash]
    else:
        logger.info('Visit references user ID: {0} but no such user.'.format(user_id_has))
        
    del v['USER_ID_hash'], v['VIEW_COUPON_ID_hash']

if len(found_coupons) > 0:
    logger.info ("Visit file references {0} coupons found in the coupon list file.".format(len(found_coupons)))
if len(missing_coupons) > 0:
    logger.info ("Visit file references {0} coupons NOT FOUND in the coupon list file (replaced with dummy coupon).".format(len(missing_coupons)))
del missing_coupons

prefecture_set = set()
large_area_name_set = set()
ken_name_set = set()
small_area_name_set = set()
capsule_text_set = set()
genre_name_set = set()

# Scan through data to build indexes for frequently accessed data
# and to accumulate any other useful statistics:
logger.info('Scanning data...')

first_purchase_date = min((p['I_DATE'] for p in purchase.values()))
last_purchase_date = max((p['I_DATE'] for p in purchase.values()))

def range_violation(user, coupon, date):
    start_date = max(user['REG_DATE'], coupon['DISPFROM'])
    end_date = min(user['WITHDRAW_DATE'], coupon['DISPEND'])
    
    if date > end_date:
        return (1, date - end_date)
    
    if date < start_date:
        return (-1, start_date-date)
    
    return None

logger.info('First/last purchase dates: {0}/{1}'.format(first_purchase_date, last_purchase_date))

# Purchase
purchase_by_user = collections.OrderedDict()
purchase_by_user_coupon_week = collections.OrderedDict()
for p in purchase.values():
    purchase_by_user.setdefault(p['USER']['USER_ID_hash'], []).append(p)
    purchase_by_user_coupon_week[(p['USER']['USER_ID_hash'], p['COUPON']['COUPON_ID_hash'], week_index(p['I_DATE']))] = 1

    # Verify the assumption that purchases do not occur outside the ad display window or when the user is not registered.
    rv = range_violation(p['USER'], p['COUPON'], p['I_DATE'])
    if rv:
        logger.warning('*** Sale w/o display: '
                       '{0} days, {1} hours, {2} minutes, {3} seconds {4}'.format(rv[1].days,
                                                                                  rv[1].seconds//3600,
                                                                                  (rv[1].seconds % 3600) // 60,
                                                                                  rv[1].seconds % 60,
                                                                                  'after close' if rv[0]>0 else 'before open'))
        logger.warning('\tpurchase date: {0}'.format(p['I_DATE']))
        logger.warning('\tuser:  {0} reg date range: {1} to {2}'.format(p['USER']['USER_ID_hash'], p['USER']['REG_DATE'], p['USER']['WITHDRAW_DATE']))
        logger.warning('\tcoupon:  {0} disp date range: {1} to {2}'.format(p['COUPON']['COUPON_ID_hash'], p['COUPON']['DISPFROM'], p['COUPON']['DISPEND']))

# Sort purchase lists
for u,purchase_list in purchase_by_user.items():
    purchase_list.sort(key=operator.itemgetter('I_DATE'), reverse=False)
    purchase_by_user[u] = tuple(purchase_list)

# Visit
visit_by_user = collections.OrderedDict()
for v in visit:
    visit_by_user.setdefault(v['USER']['USER_ID_hash'], []).append(v)
del visit    

# Sort visit lists    
for u,visit_list in visit_by_user.items():
    visit_list.sort(key=operator.itemgetter('I_DATE'), reverse=False)
    visit_by_user[u] = tuple(visit_list)

# Area
small_area_by_coupon = collections.OrderedDict()
for a in itertools.chain(train_area, test_area):
    small_area_by_coupon.setdefault(a['COUPON_ID_hash'], set()).add(a['SMALL_AREA_NAME'])
    prefecture_set.add(a['PREF_NAME'])
    small_area_name_set.add(a['SMALL_AREA_NAME'])

# Coupon
for h,c in coupon.items():
    capsule_text_set.add(c['CAPSULE_TEXT'])
    genre_name_set.add(c['GENRE_NAME'])
    large_area_name_set.add(c['large_area_name'])
    ken_name_set.add(c['ken_name'])
    small_area_name_set.add(c['small_area_name'])

# User
user_history = collections.OrderedDict()

logger.info('Building user history list')

for h,u in user.items():
    visit_history = visit_by_user.get(h, ())
    purchase_history = purchase_by_user.get(h, ())

    user_history[h] = { 'user': u,
                        'visit': visit_history,
                        'purchase': purchase_history }
    
    prefecture_set.add(u['PREF_NAME'])

del purchase_by_user, visit_by_user 

def log_set(logger, s, name):
#    logger.info('{0} ({1}): {2}'.format(name, len(s), ', '.join(sorted(map(str, s)))))
    logger.info('{0}: {1} values'.format(name, len(s)))

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

class RandomFeatureSet:
    def names(self):
        return ('random_feature',)

    def map (self, user_history, coupon, date):
        return (random_state.randrange(2),)

class ExperimentalFeatureSet:
    def names(self):
        return (
            'week_index',
        )
    
    def map (self, user_history, coupon, date):
        return (
            week_index(date),
        )

def days_accessible(user, coupon, week_start_date):
    """Returns the number of days the coupon was accessible to the user during
    the week beginning on week_start_date"""

    start = max(week_start_date, user['REG_DATE'], coupon['DISPFROM'])
    end = min(week_start_date+datetime.timedelta(days=7), user['WITHDRAW_DATE'], coupon['DISPEND'])

    try:
        interval = (end-start).total_seconds()/86400.0

    except:
        print('end: ', end)
        print('start: ', start)
        print('end-start:', end-start)
        
    return interval if interval > 0 else 0

class AvailabilityFeatureSet:
    def names(self):
        return ('days_accessible_by_user',)

    def map (self, user_history, coupon, date):
        return (days_accessible(user_history['user'], coupon, date),)

class UserPurchaseHistoryFeatureSet:
    def names(self):
        return(
            'number_of_purchases',
            'number_of_genre_purchases',
            'number_of_capsule_purchases',

            'days_since_purchase',
            'days_since_2nd_purchase',
            
            'days_since_genre_purchase',
            'days_since_capsule_purchase',

            'days_since_small_area_purchase',
            'days_since_large_area_purchase',
            'days_since_ken_purchase',

            'recency_genre_purchase',
            'recency_capsule_purchase',

            'recency_small_area_purchase',
            'recency_large_area_purchase',
            'recency_ken_purchase',

            'percent_from_genre_purchase',
            'percent_from_capsule_purchase',

            'centered_discount_price',
            'centered_price_rate',

            'relative_to_previous_discount_price_purchase',
            'relative_to_previous_price_rate_purchase',

            'relative_to_max_discount_price',
            'relative_to_min_price_rate',

            'in_previous_purchase_area',

            'max_qty_genre',
            'max_qty_capsule',

            'max_qty_ken',
            'max_qty_large_area',
            'max_qty_small_area',
        )

    def map(self, user_history, coupon, date):
        purchase_count = genre_count = capsule_count = 0

        discount_price = price_rate = -9999.0

        sum_discount_price = 0
        sum_price_rate = 0

        min_price_rate = 100
        max_discount_price = 0

        second_last_purchase = last_purchase = datetime.datetime.min
        
        days_since_capsule = days_since_genre = 1 << 31
        days_since_small_area_name = days_since_large_area_name = days_since_ken_name = 1 << 31
        
        max_qty_genre = max_qty_capsule = 0
        max_qty_ken = max_qty_large_area = max_qty_small_area = 0
        previous_purchase_area = set()

        for p in user_history['purchase']:
            # Consider only the past
            if p['I_DATE'] < date:
                second_last_purchase = last_purchase
                last_purchase = p['I_DATE']
                days = (date - last_purchase).total_seconds()/86400.0

                discount_price = p['COUPON']['DISCOUNT_PRICE']
                price_rate = p['COUPON']['PRICE_RATE']

                min_price_rate = min(min_price_rate, price_rate)
                max_discount_price = max(max_discount_price, discount_price)

                purchase_count += 1
                sum_discount_price += discount_price
                sum_price_rate += price_rate

                item_count = p['ITEM_COUNT']

                if p['COUPON']['GENRE_NAME'] == coupon['GENRE_NAME']:
                    genre_count += 1
                    days_since_genre = min(days_since_genre, days)
                    max_qty_genre = max(max_qty_genre, item_count)

                if p['COUPON']['CAPSULE_TEXT'] == coupon['CAPSULE_TEXT']:
                    capsule_count += 1
                    days_since_capsule = min(days_since_capsule, days)
                    max_qty_capsule = max(max_qty_capsule, item_count)

                if p['COUPON']['small_area_name'] == coupon['small_area_name']:
                    days_since_small_area_name = min(days_since_small_area_name, days)
                    max_qty_small_area = max(max_qty_small_area, item_count)

                if p['COUPON']['large_area_name'] == coupon['large_area_name']:
                    days_since_large_area_name = min(days_since_large_area_name, days)
                    max_qty_large_area = max(max_qty_large_area, item_count)

                if p['COUPON']['ken_name'] == coupon['ken_name']:
                    days_since_ken_name = min(days_since_ken_name, days)
                    max_qty_ken = max(max_qty_ken, item_count)

                previous_purchase_area.add(p['SMALL_AREA_NAME'])

        days_since_purchase = (date - last_purchase).total_seconds()/86400.0

        result = (
            purchase_count,
            genre_count,
            capsule_count,

            days_since_purchase,
            (date - second_last_purchase).total_seconds()/86400.0,
            
            days_since_genre,
            days_since_capsule,
            
            days_since_small_area_name,
            days_since_large_area_name,
            days_since_ken_name,

            days_since_genre - days_since_purchase,
            days_since_capsule - days_since_purchase,

            days_since_small_area_name - days_since_purchase,
            days_since_large_area_name - days_since_purchase,
            days_since_ken_name - days_since_purchase,

            float(genre_count) / purchase_count if purchase_count > 0 else -999,
            float(capsule_count) / purchase_count if purchase_count > 0 else -999,

            coupon['DISCOUNT_PRICE'] - float(sum_discount_price)/ purchase_count if purchase_count > 0 else -999,
            coupon['PRICE_RATE'] - float(sum_price_rate)/ purchase_count if purchase_count > 0 else -999,

            coupon['DISCOUNT_PRICE'] - discount_price,
            coupon['PRICE_RATE'] - price_rate,

            coupon['DISCOUNT_PRICE'] - max_discount_price,
            coupon['PRICE_RATE'] - min_price_rate,

            len(small_area_by_coupon.get(coupon['COUPON_ID_hash'], set()).intersection(previous_purchase_area)),

            max_qty_genre,
            max_qty_capsule,

            max_qty_ken,
            max_qty_large_area,
            max_qty_small_area,
        )

        return result


class UserVisitHistoryFeatureSet:
    def names(self):
        return (
            'number_of_visits',
            'number_of_genre_visits',
            'number_of_capsule_visits',

            'days_since_visit',
            'days_since_genre_visit',
            'days_since_capsule_visit',

            'days_since_small_area_visit',
            'days_since_large_area_visit',
            'days_since_ken_visit',

            'recency_genre_visit',
            'recency_capsule_visit',

            'recency_small_area_visit',
            'recency_large_area_visit',
            'recency_ken_visit',
            
#            'days_since_coupon_visit',

            'percent_from_genre_visit',
            'percent_from_capsule_visit',

            'relative_to_previous_discount_price_visit',
            'relative_to_previous_price_rate_visit',

            'peak_hour',
            'peak_dow',
        )

    def map(self, user_history, coupon, date):
        genre_count = capsule_count = 0
        
        discount_price = price_rate = -9999.0

        days_since_visit = days_since_capsule = days_since_genre = days_since_coupon = 1 << 31
        days_since_small_area_name = days_since_large_area_name = days_since_ken_name = 1 << 31
        visit_count = 0

        peak_dow = peak_hour = -1
        peak_dow_count = peak_hour_count = 0
        
        hour_count = {}
        dow_count = {}

        previous_visit_area = set()
        
        for v in user_history['visit']:
            # Consider only the past
            if v['I_DATE'] < date:
                days = (date - v['I_DATE']).total_seconds()/86400.0
                days_since_visit = min(days_since_visit, days)

                discount_price = v['COUPON']['DISCOUNT_PRICE']
                price_rate = v['COUPON']['PRICE_RATE']
                
                visit_count += 1

                hour = v['I_DATE'].hour
                hour_count[hour] = hour_count.get(hour, 0) + 1

                dow = v['I_DATE'].weekday()
                dow_count[dow] = dow_count.get(dow, 0) + 1

                if v['COUPON']['GENRE_NAME'] == coupon['GENRE_NAME']:
                    genre_count += 1
                    days_since_genre = min(days_since_genre, days)

                if v['COUPON']['CAPSULE_TEXT'] == coupon['CAPSULE_TEXT']:
                    capsule_count += 1
                    days_since_capsule = min(days_since_capsule, days)

                if v['COUPON']['small_area_name'] == coupon['small_area_name']:
                    days_since_small_area_name = min(days_since_small_area_name, days)

                if v['COUPON']['large_area_name'] == coupon['large_area_name']:
                    days_since_large_area_name = min(days_since_large_area_name, days)

                if v['COUPON']['ken_name'] == coupon['ken_name']:
                    days_since_ken_name = min(days_since_ken_name, days)

                if v['COUPON']['COUPON_ID_hash'] == coupon['COUPON_ID_hash']:
                    days_since_coupon = min(days_since_coupon, days)

        if len(hour_count) > 0:
            peak_hour_count, peak_hour = sorted(((c,h) for (h,c) in hour_count.items()), reverse=True)[0]

        if len(dow_count) > 0:
            peak_dow_count, peak_dow = sorted(((c,d) for (d,c) in dow_count.items()), reverse=True)[0]

        result = (
            visit_count,
            genre_count,
            capsule_count,

            days_since_visit,
            days_since_genre,
            days_since_capsule,

            days_since_small_area_name,
            days_since_large_area_name,
            days_since_ken_name,

            days_since_genre - days_since_visit,
            days_since_capsule - days_since_visit,

            days_since_small_area_name - days_since_visit,
            days_since_large_area_name - days_since_visit,
            days_since_ken_name - days_since_visit,

#            days_since_coupon,

            float(genre_count) / visit_count if visit_count > 0 else 0,
            float(capsule_count) / visit_count if visit_count > 0 else 0,

            coupon['DISCOUNT_PRICE'] - discount_price,
            coupon['PRICE_RATE'] - price_rate,

            peak_hour,
            peak_dow,
        )
        return result

class SimpleUserFeatureSet:
    def names(self):
        return (
            'age',
            'gender',
            'prefecture',
            'days_as_member')

    def map (self, user_history, coupon, date):
        user = user_history['user']
        return (
            user['AGE'],
            gender_encoder.map(user['SEX_ID']),
            prefecture_encoder.map(user['PREF_NAME']),
            (date - user['REG_DATE']).total_seconds()/86400.0
        )


class SimpleCouponFeatureSet:
    def names(self):
        return ('capsule_text',
                'genre_name',
                'large_area_name',
                'ken_name',
                'small_area_name',
                'price_rate',
                'catalog_price',
                'discount_price',
                'price_reduction',
                'valid_period',
                'days_on_display',
                'display_days_left',
                'days_until_valid',
                'days_until_expiration',
        )

    def map (self, user_history, coupon, date):
        return (
            capsule_encoder.map(coupon['CAPSULE_TEXT']),
            genre_name_encoder.map(coupon['GENRE_NAME']),
            large_area_name_encoder.map(coupon['large_area_name']),
            prefecture_encoder.map(coupon['ken_name']),
            small_area_name_encoder.map(coupon['small_area_name']),
            coupon['PRICE_RATE'],
            coupon['CATALOG_PRICE'],
            coupon['DISCOUNT_PRICE'],
            coupon['CATALOG_PRICE']-coupon['DISCOUNT_PRICE'],
            coupon['VALIDPERIOD'],
            (date - coupon['DISPFROM']).total_seconds()/86400.0,
            (coupon['DISPEND'] - date).total_seconds()/86400.0,
            (coupon['VALIDFROM'] - date).total_seconds()/86400.0,
            (coupon['VALIDEND'] - date).total_seconds()/86400.0,
        )

class CouponUsableDateFeatureSet:
    def names(self):
        return (
            'usable_days_everything',
        )

    def map (self, user_history, coupon, date):
        return (
            (coupon['USABLE_DATE_MON']*1 + coupon['USABLE_DATE_TUE']*(1<<3) + coupon['USABLE_DATE_WED']*(1<<6) + coupon['USABLE_DATE_THU']*(1<<9) +
             coupon['USABLE_DATE_FRI']*(1<<12) + coupon['USABLE_DATE_SAT']*(1<<15) + coupon['USABLE_DATE_SUN']*(1<<18) + coupon['USABLE_DATE_HOLIDAY']*(1<<21) +
             coupon['USABLE_DATE_BEFORE_HOLIDAY']*(1<<24)),
        )

class JointFeatureSet:
    def names(self):
        return ('distance',)
    
    def map (self, user_history, coupon, date):
        return (prefecture_distance(user_history['user']['PREF_NAME'], coupon['ken_name']),)

# dump(user_history)
accumulators = (
    naive_bayes.MultinomialNBAccumulator('visit', 'small_area_name'),
    naive_bayes.MultinomialNBAccumulator('visit', 'large_area_name'),
    naive_bayes.MultinomialNBAccumulator('visit', 'ken_name'),
    naive_bayes.MultinomialNBAccumulator('visit', 'CAPSULE_TEXT'),
    naive_bayes.MultinomialNBAccumulator('visit', 'GENRE_NAME'),

    naive_bayes.MultinomialNBAccumulator('purchase', 'small_area_name'),
    naive_bayes.MultinomialNBAccumulator('purchase', 'large_area_name'),
    naive_bayes.MultinomialNBAccumulator('purchase', 'ken_name'),
    naive_bayes.MultinomialNBAccumulator('purchase', 'CAPSULE_TEXT'),
    naive_bayes.MultinomialNBAccumulator('purchase', 'GENRE_NAME'),
    
    naive_bayes.NBAccumulator('visit', 'small_area_name'),
    naive_bayes.NBAccumulator('visit', 'large_area_name'),
    naive_bayes.NBAccumulator('visit', 'ken_name'),
    naive_bayes.NBAccumulator('visit', 'CAPSULE_TEXT'),
    naive_bayes.NBAccumulator('visit', 'GENRE_NAME'),
    
    naive_bayes.NBAccumulator('purchase', 'small_area_name'),
    naive_bayes.NBAccumulator('purchase', 'large_area_name'),
    naive_bayes.NBAccumulator('purchase', 'ken_name'),
    naive_bayes.NBAccumulator('purchase', 'CAPSULE_TEXT'),
    naive_bayes.NBAccumulator('purchase', 'GENRE_NAME'),

    naive_bayes.MultinomialNBAccumulator('visit', 'QUANTIZED_PRICE_RATE'),
    naive_bayes.MultinomialNBAccumulator('visit', 'QUANTIZED_DISCOUNT_PRICE'),
    naive_bayes.MultinomialNBAccumulator('visit', 'QUANTIZED_CATALOG_PRICE'),
    
    naive_bayes.MultinomialNBAccumulator('purchase', 'QUANTIZED_PRICE_RATE'),
    naive_bayes.MultinomialNBAccumulator('purchase', 'QUANTIZED_DISCOUNT_PRICE'),
    naive_bayes.MultinomialNBAccumulator('purchase', 'QUANTIZED_CATALOG_PRICE'),
)

class NBFeatureSet:
    def names(self):
        return (
            'small_area_visit_history_mn',
            'large_area_visit_history_mn',
            'ken_visit_history_mn',
            'capsule_visit_history_mn',
            'genre_visit_history_mn',

            'small_area_purchase_history_mn',
            'large_area_purchase_history_mn',
            'ken_purchase_history_mn',
            'capsule_purchase_history_mn',
            'genre_purchase_history_mn',
            
            'small_area_visit_history_nb',
            'large_area_visit_history_nb',
            'ken_visit_history_nb',
            'capsule_visit_history_nb',
            'genre_visit_history_nb',
            
            'small_area_purchase_history_nb',
            'large_area_purchase_history_nb',
            'ken_purchase_history_nb',
            'capsule_text_purchase_history_nb',
            'genre_name_purchase_history_nb',

            'qtz_price_rate_visit_history_mn',
            'qtz_discount_price_visit_history_mn',
            'qtz_catalog_price_visit_history_mn',

            'qtz_price_rate_purchase_history_mn',
            'qtz_discount_price_purchase_history_mn',
            'qtz_catalog_price_purchase_history_mn',

        )
    
    def map(self, user_history, coupon, date):
        return (
            accumulators[0].score(coupon, user_history, date),
            accumulators[1].score(coupon, user_history, date),
            accumulators[2].score(coupon, user_history, date),
            accumulators[3].score(coupon, user_history, date),
            accumulators[4].score(coupon, user_history, date),
            
            accumulators[5].score(coupon, user_history, date),
            accumulators[6].score(coupon, user_history, date),
            accumulators[7].score(coupon, user_history, date),
            accumulators[8].score(coupon, user_history, date),
            accumulators[9].score(coupon, user_history, date),
            
            accumulators[10].score(coupon, user_history, date),
            accumulators[11].score(coupon, user_history, date),
            accumulators[12].score(coupon, user_history, date),
            accumulators[13].score(coupon, user_history, date),
            accumulators[14].score(coupon, user_history, date),
            
            accumulators[15].score(coupon, user_history, date),
            accumulators[16].score(coupon, user_history, date),
            accumulators[17].score(coupon, user_history, date),
            accumulators[18].score(coupon, user_history, date),
            accumulators[19].score(coupon, user_history, date),

            accumulators[20].score(coupon, user_history, date),
            accumulators[21].score(coupon, user_history, date),
            accumulators[22].score(coupon, user_history, date),

            accumulators[23].score(coupon, user_history, date),
            accumulators[24].score(coupon, user_history, date),
            accumulators[25].score(coupon, user_history, date),

        )
    
feature_extractors = (
    NBFeatureSet(),
#    ExperimentalFeatureSet(),
    AvailabilityFeatureSet(),
    UserPurchaseHistoryFeatureSet(),
    UserVisitHistoryFeatureSet(),
    SimpleUserFeatureSet(),
    SimpleCouponFeatureSet(),
    CouponUsableDateFeatureSet(),
#   JointFeatureSet(),   # This is just distance.
    RandomFeatureSet()
)


feature_names = reduce(operator.add, (fe.names() for fe in feature_extractors))

logger.info('{0} features'.format(len(feature_names)))

def features(user_history, coupon, date):
    start_of_week_date = start_of_week(date)
    x = reduce (operator.add,
                (fe.map(user_history,
                        coupon,
                         start_of_week_date) for fe in feature_extractors))
    if len(x) != len(feature_names):
        logger.error('len(features_names) is {0} but len of feature vector is {1}'.format(len(feature_names), len(x)))
        for fe in feature_extractors:
            f = fe.map(user_history, coupon, start_of_week_date)
            if len(f) != len(fe.names()):
                logger.error('Culprit may be len={0}, names={1}, len(names) = {2}'.format(len(f), fe.names(), len(fe.names())))
                sys.exit(1)
    return x


# Resample the probability space to get failed cases
# The space is assumed to be (user, coupon, date) tuples
# where date is in the overlap region where the user is registered and the coupon is displayed
user_history_list = tuple(user_history.values())
coupon_list = tuple(coupon.values())

purchase_sample = random_state.sample(tuple(purchase.values()), n_positive)

# Let garbage collector work
del purchase        

logger.info('Accumulating stats for Naive Bayes in random sample')

for p in purchase_sample:
    uh  = user_history[p['USER']['USER_ID_hash']]
    for a in accumulators:
        a.add(p, uh)

accumulators[18].dump(10)
accumulators[19].dump(10)
accumulators[20].dump(10)

accumulators[21].dump(10)
accumulators[22].dump(10)
accumulators[23].dump(10)
print('End of dump')
    
logger.info('Building features for random sample')

positive_features = []
positive_users = []
positive_coupons = []
for p in purchase_sample:
    uh  = user_history[p['USER']['USER_ID_hash']]

    f = features(uh, p['COUPON'], start_of_week(p['I_DATE']))
    positive_features.append(f)
    positive_users.append(p['USER']['USER_ID_hash'])
    positive_coupons.append(p['COUPON']['COUPON_ID_hash'])

logger.info ('Sampling outcome space to obtain some non-purchase outcomes')
nonpurchase_count = 0
saw_a_one = False

logger.info ('User space is {0} users'.format(len(user_history_list)))

negative_features = []
negative_users = []
negative_coupons = []
accessibility_misses = purchase_misses = 0
for p in purchase_sample:
    purchasing_user = p['USER']
    purchasing_user_hash = purchasing_user['USER_ID_hash']
    purchase_week_start = start_of_week(p['I_DATE'])
    purchase_week_index = week_index(purchase_week_start)

    for i in range(NPP):
        random_coupon = coupon_list[random_state.randrange(len(coupon_list))]
        while (days_accessible(purchasing_user, random_coupon, purchase_week_start) == 0 or
               (purchasing_user_hash, random_coupon['COUPON_ID_hash'], purchase_week_index) in purchase_by_user_coupon_week):
            random_coupon = coupon_list[random_state.randrange(len(coupon_list))]
            if days_accessible(purchasing_user, random_coupon, purchase_week_start) == 0:
                accessibility_misses += 1
            else:
                purchase_misses += 1
            
        f = features(user_history[purchasing_user_hash], random_coupon, purchase_week_start)
        negative_features.append(f)
        negative_users.append(purchasing_user_hash)
        negative_coupons.append(random_coupon['COUPON_ID_hash'])

logger.info('Misses due to accessibility: {0}; misses due to purchase {1}'.format(accessibility_misses, purchase_misses))
    

# Let garbage collector work
del purchase_by_user_coupon_week
            
positive_train_size = (2*n_positive) // 3
train_features = positive_features[:positive_train_size] + negative_features[:NPP*positive_train_size]
train_outcomes = positive_train_size * [1] + NPP*positive_train_size * [0]
train_users = positive_users[:positive_train_size] + negative_users[:NPP*positive_train_size]
train_coupons = positive_coupons[:positive_train_size] + negative_coupons[:NPP*positive_train_size]

positive_test_size = n_positive - positive_train_size
test_features = positive_features[positive_train_size:] + negative_features[NPP*positive_train_size:]
test_outcomes = positive_test_size * [1] + NPP*positive_test_size * [0]
test_users = positive_users[positive_train_size:] + negative_users[NPP*positive_train_size:]
test_coupons = positive_coupons[positive_train_size:] + negative_coupons[NPP*positive_train_size:]

logger.info('{0} training cases; {1} test cases'.format(len(train_features), len(test_features)))

for name, regressor in regressors:
    logger.info('Training {0}: {1}'.format(name, regressor))
    
    regressor.fit(train_features, train_outcomes)
    test_predictions = regressor.predict(test_features)
    if hasattr(regressor, 'feature_importances_'):
        logger.info('Importances:')
        for i,n in sorted(zip(regressor.feature_importances_, feature_names), reverse=True):
            logger.info('{0:>45}: {1:6.4f}'.format(n, i))
        
    logger.info('Performance:')
    logger.info('{0:>24}: {1:7.5f}/{2:7.5f}'.format('min/max test prediction', min(test_predictions), max(test_predictions)))
    logger.info('{0:>24}: {1:6.4f}'.format('Default classifier/regressor score', regressor.score(test_features, test_outcomes)))
    logger.info('{0:>24}: {1:6.4f}'.format('MSE', sklearn.metrics.mean_squared_error(test_outcomes, test_predictions)))
    logger.info('{0:>24}: {1:5.3f}'.format('auroc', sklearn.metrics.roc_auc_score(test_outcomes, test_predictions)))
    logger.info('{0:>24}: {1:5.3f}'.format('log loss', sklearn.metrics.log_loss(test_outcomes, test_predictions)))

logger.info('Writing test data')
with open('features.csv', "w") as feature_output:
    feature_writer = csv.writer(feature_output)
    feature_writer.writerow(tuple(('USER_ID_hash','COUPON_ID_hash') + feature_names + ('prediction', 'outcome')))
    
    for user, coupon, feature, outcome, prediction in zip(test_users, test_coupons, test_features, test_outcomes, test_predictions):
        feature_writer.writerow((user,coupon) + feature + (prediction, outcome))

logger.info('Scoring and writing output file.')

coupon_test = read_file('coupon_list_test.csv', 'COUPON_ID_hash')
coupon_computed_fields(coupon_test)

KEEP = 10
with open("submission.csv", "w") as outputfile, \
     open("probabilities.csv", "w") as probabilityfile:

    writer = csv.writer(outputfile)
    probability_writer = csv.writer(probabilityfile)
    
    writer.writerow(('USER_ID_hash', 'PURCHASED_COUPONS'))
    probability_writer.writerow(['USER_ID_hash'] + list(itertools.chain(*[('probability_{0}'.format(i), 'COUPON_ID_hash_{0}'.format(i)) for i in range(KEEP)])))

    for user_hash in sorted(user_history.keys()):
        a_user_history = user_history[user_hash]
        a_user = a_user_history['user']

        feature_hash_pairs = tuple(( (features(a_user_history, coupon, test_week_start_date),
                                      coupon['COUPON_ID_hash'])
                                     for coupon in coupon_test.values()
                                     if days_accessible(a_user, coupon, test_week_start_date) > 0))
        
        if len(feature_hash_pairs) > 0:
            f, h = zip(*feature_hash_pairs)
            probabilities = regressor.predict(f)
            p_and_h = sorted(zip(probabilities, h), reverse=True)
        else:
            p_and_h = ()
            
        winners = ' '.join((h for p,h in p_and_h[:10] if p > 0))
        writer.writerow((user_hash, winners))
        probability_writer.writerow([user_hash] + list(itertools.chain(*p_and_h[:10])))

logger.info('Finished.')
