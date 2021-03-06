import argparse
import codecs
import collections
import csv
import datetime
from functools import reduce
import itertools
import logging

# Logging
logger = logging.getLogger('')
FORMAT = '%(asctime)-15s %(name)s %(message)s'
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.DEBUG)

import operator
import naive_bayes
import naive_bayes_wrapper
import math
import numpy
import random
import sklearn
import sklearn.ensemble
import sklearn.ensemble.weight_boosting
import sys
import util

# TUNABLE PARAMETERS
# Training and test size selection
N = 202020
N = 306000
frac_positive = 0.54
negative_weight = 1.2
test_fraction = 0.01
beta = 3.0

# Estimator parameters
n_estimators1 = 500
n_estimators2 = 2000
min_samples_split = 10
min_samples_leaf = 5
max_features = 10
n_jobs=-1
oob_score=False

# seed=12345678
seed=998877665544

n_positive = int(N*frac_positive) # Number of postive training cases.
n_negative = N - n_positive

# Random number seeds
random_state = random.Random(seed)
classifier_random_state = numpy.random.RandomState(seed=seed)

# Important constants
train_start_date = datetime.datetime(year=2011, month=7, day=3, hour=0, minute=0)
test_start_date = datetime.datetime(year=2012, month=6, day=24)

def random_list_element(random_state, l):
    return l[random_state.randrange(len(l))]

class WrappedClassifier:
    """Wrap the brain-dead classifier API to make it look more like a regressor."""
    def __init__(self, classifier):
        self.classifier = classifier

    def __repr__(self):
        return ("WrappedClassifier({0})".format(self.classifier))

    def fit(self, x, y, sample_weight=None):
        self.classifier.fit(x, y, sample_weight=sample_weight)
        #        self.oob_score_ = self.classifier.oob_score_

        if hasattr(self.classifier, "oob_decision_function_"):
            self.oob_prediction_ = self.classifier.oob_decision_function_[:,1].reshape(len(y))

        if hasattr(self.classifier, "feature_importances_"):
            self.feature_importances_ = self.classifier.feature_importances_

    def predict(self, x):
        return self.classifier.predict_proba(x)[:,1]

    def set_params(self, **kwargs):
        self.classifier.set_params(**kwargs)

    def score(self, x, y):
        return self.classifier.score(x, y)

    

regressors = ( 
    ('RandomForestRegressor',
     sklearn.ensemble.RandomForestRegressor(
         random_state=classifier_random_state,
         n_estimators=n_estimators1,
         min_samples_leaf=min_samples_leaf,
         min_samples_split=min_samples_split,
         max_features=max_features,
         n_jobs=n_jobs,
         oob_score=oob_score
     )),

    ('RandomForestClassifier',
     WrappedClassifier(sklearn.ensemble.RandomForestClassifier(
         random_state=classifier_random_state,
         n_estimators=n_estimators1,
         min_samples_leaf=min_samples_leaf,
         max_features=max_features,
         n_jobs=n_jobs,
         criterion='entropy',
         oob_score=oob_score
     ))),

    ('GradientBootingRegressor',
     sklearn.ensemble.gradient_boosting.GradientBoostingRegressor(
         random_state=classifier_random_state,
         n_estimators=n_estimators1,
         min_samples_leaf=min_samples_leaf,
         max_features=max_features)),
    
    ('GradientBoostingClassifier',
     WrappedClassifier(sklearn.ensemble.gradient_boosting.GradientBoostingClassifier(
         random_state=classifier_random_state,
         n_estimators=n_estimators1,
         min_samples_leaf=min_samples_leaf,
         max_features=max_features))),

    ('AdaboostRegressor',
     sklearn.ensemble.weight_boosting.AdaBoostRegressor(
         n_estimators=n_estimators1,
     )),
    
    ('AdaboostClassifier',
     WrappedClassifier(sklearn.ensemble.weight_boosting.AdaBoostClassifier(
         n_estimators=n_estimators1,
     ))),

)

# For now, don't use boosting methods
regressors = regressors[0:1]

def week_index(date):
    return int((date - train_start_date).days / 7.0)

def week_from_index(index):
    return train_start_date + index*datetime.timedelta(days=7)

def start_of_week(date):
    return week_from_index(week_index(date))

def dump(d):
    for k, v in d.items():
        logger.debug('{0}:'.format(k))
        logger.debug('      {0}'.format(v))

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

def coupon_days_accessible(coupon, week_start_date):
    """Returns the number of days the coupon was accessible to anyone during week beginning on week_start_date"""

    start = max(week_start_date, coupon['DISPFROM'])
    end = min(week_start_date+datetime.timedelta(days=7), coupon['DISPEND'])

    try:
        interval = (end-start).total_seconds()/86400.0

    except:
        print('end: ', end)
        print('start: ', start)
        print('end-start:', end-start)
        interval = 0.0
        
    return interval if interval > 0 else 0

parser = argparse.ArgumentParser(description='Train, optionally test, and generate a coupon contest entry')
parser.add_argument('--validate', help='Hold out last week as a test/validation set', action='store_true')
parser.add_argument('--score', help='Generate a submission file with recommendations', action='store_true')
args = parser.parse_args()

if args.validate:
    train_end_date = test_start_date - datetime.timedelta(days=7)
else:
    train_end_date = test_start_date

train_period_in_weeks = (train_end_date - train_start_date).days // 7 

logger.info(
    'Parameters: '
    'N: {0:,}, '
    'frac_positive: {1:,}, '
    'n_positive: {2:,}, '
    'n_estimators1: {3:,}, '
    'test_fraction: {4:6.3f}, '
    'min_samples_leaf: {5}, '
    'max_features: {6}, '
    'n_jobs: {7}, '
    'validate: {8}, '
    'beta: {9}'
    .format(N,
            frac_positive,
            n_positive,
            n_estimators1,
            test_fraction,
            min_samples_leaf,
            max_features,
            n_jobs,
            args.validate,
            beta,
        )
)

logger.info('train_period_in_weeks: {0}'.format(train_period_in_weeks))

def user_computed_fields(user_list):
    # Add some computed fields in the user records
    for u in user_list.values():
        u['QUANTIZED_AGE'] = (u['AGE'] // 3) * 3

user = util.read_file('user_list.csv', 'USER_ID_hash')
user_computed_fields(user)

for u in list(itertools.islice(user.values(), 0, 2)):
    logger.debug('Sample user record: {0}'.format(u))

def add_coupon_computed_fields(coupon_list):
    # Add some computed fields in the coupon records
    for c in coupon_list.values():
        # Quantized various prices to make it easier to use Naive Bayes
        try:
            c['QUANTIZED_PRICE_RATE'] = int((c['PRICE_RATE'] // 10) * 10.0)
            c['QUANTIZED_DISCOUNT_PRICE'] = int(2 ** (int(math.log2(0.01 + c['DISCOUNT_PRICE']) * 2.0) / 2.0))
            c['QUANTIZED_CATALOG_PRICE'] = int(2 ** (int(math.log2(0.01 + c['CATALOG_PRICE']) * 2.0) / 2.0))
        except:
            print(c['PRICE_RATE'])
            print(c['DISCOUNT_PRICE'])
            print(c['CATALOG_PRICE'])
            raise

coupon = util.read_file('coupon_list_train.csv', 'COUPON_ID_hash')
add_coupon_computed_fields(coupon)

coupon_accessibility_by_week = {}
for c in coupon.values():
    for i in range(train_period_in_weeks):
        date = week_from_index(i)
        if coupon_days_accessible(c, date) > 0:
            coupon_accessibility_by_week.setdefault(i, set()).add(c['COUPON_ID_hash'])

for i in coupon_accessibility_by_week.keys():
    coupon_accessibility_by_week[i] = list(coupon_accessibility_by_week[i])

coupon_test = util.read_file('coupon_list_test.csv', 'COUPON_ID_hash')
add_coupon_computed_fields(coupon_test)

coupon.update(coupon_test)

for c in list(itertools.islice(coupon.values(), 0, 2)):
    logger.debug('Sample coupon record: {0}'.format(c))

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

purchase = util.read_file('coupon_detail_train.csv', 'PURCHASEID_hash')

train_area = util.read_file('coupon_area_train.csv')
test_area = util.read_file('coupon_area_test.csv')

validation_purchase = {}
validation_coupon = {}

# Dereference user and coupon references and remove records outside the date range
for k,v in purchase.items():
    v['HOUR'] = v['I_DATE'].hour
    v['DAY'] = v['I_DATE'].weekday()
    v['USER'] = user[v['USER_ID_hash']]
    v['COUPON'] = coupon[v['COUPON_ID_hash']]
    v['PURCHASE_WEEK_DATE'] = start_of_week(v['I_DATE'])
    del v['USER_ID_hash'], v['COUPON_ID_hash']

    if v['I_DATE'] < train_start_date or v['I_DATE'] >= train_end_date:
        del purchase[k]
        
    if args.validate and v['I_DATE'] >= train_end_date and v['I_DATE'] < test_start_date:
        validation_purchase.setdefault(v['USER']['USER_ID_hash'], set()).add(v['COUPON']['COUPON_ID_hash'])
        validation_coupon[v['COUPON']['COUPON_ID_hash']] = v['COUPON']

if args.validate:
    logger.info(
        'Validation week ({0}): {1} purchases; {2} purchasers; {3} distinct items; {4} max per purchaser'.format(
            train_end_date,
            sum([len(v) for v in validation_purchase.values()]),
            len(validation_purchase),
            len(validation_coupon),
            max([len(v) for v in validation_purchase.values()])
        )
    )
    for k,v in list(itertools.islice(validation_purchase.items(), 0, 2)):
        logger.debug('Sample validation week purchases: user: {0}, purchases: {1}'.format(k, v))
    

for p in list(itertools.islice(purchase.values(), 0, 2)):
    logger.debug('Sample purchase record: {0}'.format(p))
        
logger.info('Retained {0:,} purchase records between {1} and {2}'.format(
    len(purchase),
    min(p['I_DATE'] for p in purchase.values()),
    max(p['I_DATE'] for p in purchase.values())))

# Dereference user and purchase references
visit = util.read_file('coupon_visit_train.csv')

missing_coupons = set()
found_coupons = set()

for v in visit:
    v['HOUR'] = v['I_DATE'].hour
    v['DAY'] = v['I_DATE'].weekday()
    
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
purchase_by_user_week = collections.OrderedDict()
purchase_by_user_coupon_week = collections.OrderedDict()

for p in purchase.values():
    purchase_by_user.setdefault(p['USER']['USER_ID_hash'], []).append(p)
    purchase_by_user_week[(p['USER']['USER_ID_hash'], week_index(p['I_DATE']))] = 1
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


class AvailabilityFeatureSet:
    def names(self):
        return ('days_accessible_by_user',)

    def map (self, user_history, coupon, date):
        return (days_accessible(user_history['user'], coupon, date),)

class UserPurchaseHistoryFeatureSet:
    def names(self):
        return(
#            'number_of_purchases',
#            'number_of_genre_purchases',
#            'number_of_capsule_purchases',

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
#            'recency_large_area_purchase',
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

#            'max_qty_genre',
#            'max_qty_capsule',

#            'max_qty_ken',
#            'max_qty_large_area',
#            'max_qty_small_area',
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
                purchased_coupon = p['COUPON']

                if purchased_coupon['GENRE_NAME'] == coupon['GENRE_NAME']:
                    genre_count += 1
                    days_since_genre = min(days_since_genre, days)
                    max_qty_genre = max(max_qty_genre, item_count)

                if purchased_coupon['CAPSULE_TEXT'] == coupon['CAPSULE_TEXT']:
                    capsule_count += 1
                    days_since_capsule = min(days_since_capsule, days)
                    max_qty_capsule = max(max_qty_capsule, item_count)

                if purchased_coupon['small_area_name'] == coupon['small_area_name']:
                    days_since_small_area_name = min(days_since_small_area_name, days)
                    max_qty_small_area = max(max_qty_small_area, item_count)

                if purchased_coupon['large_area_name'] == coupon['large_area_name']:
                    days_since_large_area_name = min(days_since_large_area_name, days)
                    max_qty_large_area = max(max_qty_large_area, item_count)

                if purchased_coupon['ken_name'] == coupon['ken_name']:
                    days_since_ken_name = min(days_since_ken_name, days)
                    max_qty_ken = max(max_qty_ken, item_count)

                previous_purchase_area.add(p['SMALL_AREA_NAME'])

        days_since_purchase = (date - last_purchase).total_seconds()/86400.0

        result = (
#            purchase_count,
#            genre_count,
#            capsule_count,

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
#            days_since_large_area_name - days_since_purchase,
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

#            max_qty_genre,
#            max_qty_capsule,

#            max_qty_ken,
#            max_qty_large_area,
#            max_qty_small_area,
        )

        return result


class UserVisitHistoryFeatureSet:
    def names(self):
        return (
            'number_of_visits',
#            'number_of_genre_visits',
#            'number_of_capsule_visits',

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
#            'peak_dow',
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

                visited_coupon = v['COUPON']

                if visited_coupon['GENRE_NAME'] == coupon['GENRE_NAME']:
                    genre_count += 1
                    days_since_genre = min(days_since_genre, days)

                if visited_coupon['CAPSULE_TEXT'] == coupon['CAPSULE_TEXT']:
                    capsule_count += 1
                    days_since_capsule = min(days_since_capsule, days)

                if visited_coupon['small_area_name'] == coupon['small_area_name']:
                    days_since_small_area_name = min(days_since_small_area_name, days)

                if visited_coupon['large_area_name'] == coupon['large_area_name']:
                    days_since_large_area_name = min(days_since_large_area_name, days)

                if visited_coupon['ken_name'] == coupon['ken_name']:
                    days_since_ken_name = min(days_since_ken_name, days)

                if visited_coupon['COUPON_ID_hash'] == coupon['COUPON_ID_hash']:
                    days_since_coupon = min(days_since_coupon, days)

        if len(hour_count) > 0:
            peak_hour_count, peak_hour = sorted(((c,h) for (h,c) in hour_count.items()), reverse=True)[0]

        if len(dow_count) > 0:
            peak_dow_count, peak_dow = sorted(((c,d) for (d,c) in dow_count.items()), reverse=True)[0]

        result = (
            visit_count,
#            genre_count,
#            capsule_count,

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
#            peak_dow,
        )
        return result

class SimpleUserFeatureSet:
    def names(self):
        return (
            'age',
#            'gender',
            'prefecture',
            'days_as_member')

    def map (self, user_history, coupon, date):
        user = user_history['user']
        return (
            user['AGE'],
#            gender_encoder.map(user['SEX_ID']),
            prefecture_encoder.map(user['PREF_NAME']),
            (date - user['REG_DATE']).total_seconds()/86400.0
        )


class SimpleCouponFeatureSet:
    def names(self):
        return ('capsule_text',
                'genre_name',
#                'large_area_name',
                'ken_name',
                'small_area_name',
                'price_rate',
                'catalog_price',
                'discount_price',
#                'price_reduction',
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
#            large_area_name_encoder.map(coupon['large_area_name']),
            prefecture_encoder.map(coupon['ken_name']),
            small_area_name_encoder.map(coupon['small_area_name']),
            coupon['PRICE_RATE'],
            coupon['CATALOG_PRICE'],
            coupon['DISCOUNT_PRICE'],
#            coupon['CATALOG_PRICE']-coupon['DISCOUNT_PRICE'],
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
    naive_bayes_wrapper.CacheableWrapper(naive_bayes.MNStrategy(), user_history, coupon, 'visit', 'small_area_name', 'small_area_name'),
    naive_bayes_wrapper.CacheableWrapper(naive_bayes.MNStrategy(), user_history, coupon, 'visit', 'large_area_name', 'large_area_name'),
    naive_bayes_wrapper.CacheableWrapper(naive_bayes.MNStrategy(), user_history, coupon, 'visit', 'ken_name', 'ken_name'),
    naive_bayes_wrapper.CacheableWrapper(naive_bayes.MNStrategy(), user_history, coupon, 'visit', 'CAPSULE_TEXT', 'CAPSULE_TEXT'),
    naive_bayes_wrapper.CacheableWrapper(naive_bayes.MNStrategy(), user_history, coupon, 'visit', 'GENRE_NAME', 'GENRE_NAME'),

    naive_bayes_wrapper.CacheableWrapper(naive_bayes.MNStrategy(), user_history, coupon, 'purchase', 'small_area_name', 'small_area_name'),
    naive_bayes_wrapper.CacheableWrapper(naive_bayes.MNStrategy(), user_history, coupon, 'purchase', 'large_area_name', 'large_area_name'),
    naive_bayes_wrapper.CacheableWrapper(naive_bayes.MNStrategy(), user_history, coupon, 'purchase', 'ken_name', 'ken_name'),
    naive_bayes_wrapper.CacheableWrapper(naive_bayes.MNStrategy(), user_history, coupon, 'purchase', 'CAPSULE_TEXT', 'CAPSULE_TEXT'),
    naive_bayes_wrapper.CacheableWrapper(naive_bayes.MNStrategy(), user_history, coupon, 'purchase', 'GENRE_NAME', 'GENRE_NAME'),
    
    naive_bayes_wrapper.CacheableWrapper(naive_bayes.BNStrategy(), user_history, coupon, 'visit', 'small_area_name', 'small_area_name'),
    naive_bayes_wrapper.CacheableWrapper(naive_bayes.BNStrategy(), user_history, coupon, 'visit', 'large_area_name', 'large_area_name'),
    naive_bayes_wrapper.CacheableWrapper(naive_bayes.BNStrategy(), user_history, coupon, 'visit', 'ken_name', 'ken_name'),
    naive_bayes_wrapper.CacheableWrapper(naive_bayes.BNStrategy(), user_history, coupon, 'visit', 'CAPSULE_TEXT','CAPSULE_TEXT'),
    naive_bayes_wrapper.CacheableWrapper(naive_bayes.BNStrategy(), user_history, coupon, 'visit', 'GENRE_NAME', 'GENRE_NAME'),
    
    naive_bayes_wrapper.CacheableWrapper(naive_bayes.BNStrategy(), user_history, coupon, 'purchase', 'small_area_name','small_area_name'),
    naive_bayes_wrapper.CacheableWrapper(naive_bayes.BNStrategy(), user_history, coupon, 'purchase', 'large_area_name', 'large_area_name'),
    naive_bayes_wrapper.CacheableWrapper(naive_bayes.BNStrategy(), user_history, coupon, 'purchase', 'ken_name', 'ken_name'),
    naive_bayes_wrapper.CacheableWrapper(naive_bayes.BNStrategy(), user_history, coupon, 'purchase', 'CAPSULE_TEXT', 'CAPSULE_TEXT'),
    naive_bayes_wrapper.CacheableWrapper(naive_bayes.BNStrategy(), user_history, coupon, 'purchase', 'GENRE_NAME', 'GENRE_NAME'),

    naive_bayes_wrapper.CacheableWrapper(naive_bayes.MNStrategy(), user_history, coupon, 'visit', 'QUANTIZED_PRICE_RATE', 'QUANTIZED_PRICE_RATE'),
    naive_bayes_wrapper.CacheableWrapper(naive_bayes.MNStrategy(), user_history, coupon, 'visit', 'QUANTIZED_DISCOUNT_PRICE', 'QUANTIZED_DISCOUNT_PRICE'),
    naive_bayes_wrapper.CacheableWrapper(naive_bayes.MNStrategy(), user_history, coupon, 'visit', 'QUANTIZED_CATALOG_PRICE', 'QUANTIZED_CATALOG_PRICE'),
    
    naive_bayes_wrapper.CacheableWrapper(naive_bayes.MNStrategy(), user_history, coupon, 'purchase', 'QUANTIZED_PRICE_RATE', 'QUANTIZED_PRICE_RATE'),
    naive_bayes_wrapper.CacheableWrapper(naive_bayes.MNStrategy(), user_history, coupon, 'purchase', 'QUANTIZED_DISCOUNT_PRICE', 'QUANTIZED_DISCOUNT_PRICE'),
    naive_bayes_wrapper.CacheableWrapper(naive_bayes.MNStrategy(), user_history, coupon, 'purchase', 'QUANTIZED_CATALOG_PRICE', 'QUANTIZED_CATALOG_PRICE'),

    naive_bayes_wrapper.SimpleNBWrapper(user_history, coupon, 'QUANTIZED_AGE', 'small_area_name'),
    naive_bayes_wrapper.SimpleNBWrapper(user_history, coupon, 'QUANTIZED_AGE', 'large_area_name'),
    naive_bayes_wrapper.SimpleNBWrapper(user_history, coupon, 'QUANTIZED_AGE', 'ken_name'),
    naive_bayes_wrapper.SimpleNBWrapper(user_history, coupon, 'QUANTIZED_AGE', 'CAPSULE_TEXT'),
    naive_bayes_wrapper.SimpleNBWrapper(user_history, coupon, 'QUANTIZED_AGE', 'GENRE_NAME'),

    naive_bayes_wrapper.SimpleNBWrapper(user_history, coupon, 'SEX_ID', 'small_area_name'),
    naive_bayes_wrapper.SimpleNBWrapper(user_history, coupon, 'SEX_ID', 'large_area_name'),
    naive_bayes_wrapper.SimpleNBWrapper(user_history, coupon, 'SEX_ID', 'ken_name'),
    naive_bayes_wrapper.SimpleNBWrapper(user_history, coupon, 'SEX_ID', 'CAPSULE_TEXT'),
    naive_bayes_wrapper.SimpleNBWrapper(user_history, coupon, 'SEX_ID', 'GENRE_NAME'),

    naive_bayes_wrapper.CacheableWrapper(naive_bayes.MNStrategy(), user_history, coupon, 'visit', 'CAPSULE_TEXT', 'DAY', from_coupon=False),
    naive_bayes_wrapper.CacheableWrapper(naive_bayes.MNStrategy(), user_history, coupon, 'visit', 'GENRE_NAME', 'DAY', from_coupon=False),
    naive_bayes_wrapper.CacheableWrapper(naive_bayes.MNStrategy(), user_history, coupon, 'visit', 'CAPSULE_TEXT', 'HOUR', from_coupon=False),
    naive_bayes_wrapper.CacheableWrapper(naive_bayes.MNStrategy(), user_history, coupon, 'visit', 'GENRE_NAME', 'HOUR', from_coupon=False),
    
    naive_bayes_wrapper.CacheableWrapper(naive_bayes.MNStrategy(), user_history, coupon, 'purchase', 'small_area_name', 'SMALL_AREA_NAME', from_coupon=False),
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
            
#            'small_area_visit_history_nb',
#            'large_area_visit_history_nb',
#            'ken_visit_history_nb',
#            'capsule_visit_history_nb',
#            'genre_visit_history_nb',
            
#            'small_area_purchase_history_nb',
#            'large_area_purchase_history_nb',
#            'ken_purchase_history_nb',
#            'capsule_purchase_history_nb',
#            'genre_purchase_history_nb',

            'qtz_price_rate_visit_history_mn',
            'qtz_discount_price_visit_history_mn',
            'qtz_catalog_price_visit_history_mn',

            'qtz_price_rate_purchase_history_mn',
            'qtz_discount_price_purchase_history_mn',
            'qtz_catalog_price_purchase_history_mn',

            'age_small_area',
            'age_large_area',
            'age_ken',
            'age_capsule_text',
            'age_genre_name',

            'gender_small_area',
#            'gender_large_area',
            'gender_ken',
            'gender_capsule_text',
            'gender_genre_name',

            'day_vs_capsule',
            'day_vs_genre',
            'hour_vs_capsule',
            'hour_vs_genre',

            'residence_small_area_vs_small_area',
        )
    
    def map(self, user_history, coupon, date):
        user_hash = user_history['user']['USER_ID_hash']
        coupon_hash = coupon['COUPON_ID_hash']
        return (
            accumulators[0].score(coupon_hash, user_hash, date),
            accumulators[1].score(coupon_hash, user_hash, date),
            accumulators[2].score(coupon_hash, user_hash, date),
            accumulators[3].score(coupon_hash, user_hash, date),
            accumulators[4].score(coupon_hash, user_hash, date),
            
            accumulators[5].score(coupon_hash, user_hash, date),
            accumulators[6].score(coupon_hash, user_hash, date),
            accumulators[7].score(coupon_hash, user_hash, date),
            accumulators[8].score(coupon_hash, user_hash, date),
            accumulators[9].score(coupon_hash, user_hash, date),
            
#            accumulators[10].score(coupon_hash, user_hash, date),
#            accumulators[11].score(coupon_hash, user_hash, date),
#            accumulators[12].score(coupon_hash, user_hash, date),
#            accumulators[13].score(coupon_hash, user_hash, date),
#            accumulators[14].score(coupon_hash, user_hash, date),
            
#            accumulators[15].score(coupon_hash, user_hash, date),
#            accumulators[16].score(coupon_hash, user_hash, date),
#            accumulators[17].score(coupon_hash, user_hash, date),
#            accumulators[18].score(coupon_hash, user_hash, date),
#            accumulators[19].score(coupon_hash, user_hash, date),

            accumulators[20].score(coupon_hash, user_hash, date),
            accumulators[21].score(coupon_hash, user_hash, date),
            accumulators[22].score(coupon_hash, user_hash, date),

            accumulators[23].score(coupon_hash, user_hash, date),
            accumulators[24].score(coupon_hash, user_hash, date),
            accumulators[25].score(coupon_hash, user_hash, date),

            accumulators[26].score(coupon_hash, user_hash, date),
            accumulators[27].score(coupon_hash, user_hash, date),
            accumulators[28].score(coupon_hash, user_hash, date),
            accumulators[29].score(coupon_hash, user_hash, date),
            accumulators[30].score(coupon_hash, user_hash, date),

            accumulators[31].score(coupon_hash, user_hash, date),
#            accumulators[32].score(coupon_hash, user_hash, date),
            accumulators[33].score(coupon_hash, user_hash, date),
            accumulators[34].score(coupon_hash, user_hash, date),
            accumulators[35].score(coupon_hash, user_hash, date),
            
            accumulators[36].score(coupon_hash, user_hash, date),
            accumulators[37].score(coupon_hash, user_hash, date),
            accumulators[38].score(coupon_hash, user_hash, date),
            accumulators[39].score(coupon_hash, user_hash, date),

            accumulators[40].score(coupon_hash, user_hash, date),
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
#    RandomFeatureSet()
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

purchase_sample = random_state.sample(list(purchase.values()), n_positive)

logger.info('Accumulating stats for Naive Bayes in random sample')

for p in purchase_sample:
    user_hash = p['USER']['USER_ID_hash']
    for a in accumulators:
        a.add(p, user_hash)

accumulators[18].dump(10)
accumulators[19].dump(10)
accumulators[20].dump(10)

accumulators[21].dump(10)
accumulators[22].dump(10)
accumulators[23].dump(10)

accumulators[26].dump(10)
accumulators[27].dump(10)
accumulators[28].dump(10)
accumulators[29].dump(10)
accumulators[30].dump(10)

accumulators[31].dump(10)
accumulators[32].dump(10)
accumulators[33].dump(10)
accumulators[34].dump(10)
accumulators[35].dump(10)

accumulators[36].dump(10)
accumulators[37].dump(10)
accumulators[38].dump(10)
accumulators[39].dump(10)

logger.info('Building features for random sample')
positive_features = []
positive_users = []
positive_coupons = []
for p in purchase_sample:
    purchasing_user_hash = p['USER']['USER_ID_hash']
    purchased_coupon = p['COUPON']
    
    uh  = user_history[purchasing_user_hash]

    f = features(uh, purchased_coupon, start_of_week(p['I_DATE']))
    positive_features.append(f)
    positive_users.append(purchasing_user_hash)
    positive_coupons.append(purchased_coupon['COUPON_ID_hash'])

del purchase_sample
    
logger.info ('User space is {0} users'.format(len(user_history_list)))
logger.info ('Sampling outcome space to obtain some non-purchase outcomes')

negative_features = []
negative_users = []
negative_coupons = []
accessibility_misses = purchase_misses = 0

purchase_list = list(purchase_by_user_week.keys())
del purchase
# Sample negative outcome space with replacement --- we want the possibility of multiple
# negative outcomes for the same coupon, user, and week.
for i in range(n_negative):
    purchasing_user_hash,purchase_week_index = purchase_list[random_state.randrange(len(purchase_list))]
    purchasing_user = user[purchasing_user_hash]
    purchase_week_start = week_from_index(purchase_week_index)
    
    random_coupon_hash = random_list_element(random_state, coupon_accessibility_by_week[purchase_week_index])
    random_coupon = coupon[random_coupon_hash]
    
    while (days_accessible(purchasing_user, random_coupon, purchase_week_start) == 0 or
           (purchasing_user_hash, random_coupon_hash, purchase_week_index) in purchase_by_user_coupon_week):
        if days_accessible(purchasing_user, random_coupon, purchase_week_start) == 0:
            accessibility_misses += 1
        else:
            purchase_misses += 1

        random_coupon_hash = random_list_element(random_state, coupon_accessibility_by_week[purchase_week_index])
        random_coupon = coupon[random_coupon_hash]

    f = features(user_history[purchasing_user_hash], random_coupon, purchase_week_start)
    negative_features.append(f)
    negative_users.append(purchasing_user_hash)
    negative_coupons.append(random_coupon['COUPON_ID_hash'])


logger.info('From {0} negative cases: misses due to accessibility: {1}; misses due to purchase {2}'.format(n_negative, accessibility_misses, purchase_misses))
# negative_weight = float(n_negative+1)/float(purchase_misses) * n_positive/n_negative
logger.info('Using weight of {0:7.2f} on negative samples.'.format(negative_weight))
    

# Let garbage collector work
# del purchase_by_user_coupon_week
            
positive_train_size = int((1.0-test_fraction) * n_positive)
negative_train_size = int((1.0-test_fraction) * n_negative)

train_features = positive_features[:positive_train_size] + negative_features[:negative_train_size]
train_outcomes = positive_train_size * [1] + negative_train_size * [0]
train_weights = positive_train_size * [1] + negative_train_size * [negative_weight]

positive_test_size = n_positive - positive_train_size
negative_test_size = n_negative - negative_train_size

test_features = positive_features[positive_train_size:] + negative_features[negative_train_size:]
test_users = positive_users[positive_train_size:] + negative_users[negative_train_size:]
test_coupons = positive_coupons[positive_train_size:] + negative_coupons[negative_train_size:]
test_outcomes = positive_test_size * [1] + negative_test_size * [0]

logger.info('{0} training cases; {1} test cases'.format(len(train_features), len(test_features)))

for name, regressor in regressors:
    logger.info('Training {0}: {1}'.format(name, regressor))
    regressor.fit(train_features, train_outcomes, sample_weight=train_weights)
    
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
    
    for u, c, f, o, p in zip(test_users, test_coupons, test_features, test_outcomes, test_predictions):
        feature_writer.writerow((u,c) + f + (p, o))


# ***** BOOSTING EXPERIMENT ****
negative_features = []
negative_users = []
negative_coupons = []
negative_weights = []
accessibility_misses = purchase_misses = 0
# Sample negative outcome space with replacement --- we want the possibility of multiple
# negative outcomes for mthe same coupon, user, and week.
accepted = 0

beta_frac,beta_whole = math.modf(beta)
for i in range(n_negative):
    max_score = -1
    # Handle non-integer values of beta by interpolating between the integer values.
    for j in range(int(beta_whole) + (beta_frac > random_state.random())):
        purchasing_user_hash,purchase_week_index = purchase_list[random_state.randrange(len(purchase_list))]
        purchasing_user = user[purchasing_user_hash]
        purchase_week_start = week_from_index(purchase_week_index)
    
        random_coupon_hash = random_list_element(random_state, coupon_accessibility_by_week[purchase_week_index])
        random_coupon = coupon[random_coupon_hash]

        while (days_accessible(purchasing_user, random_coupon, purchase_week_start) == 0 or
               (purchasing_user_hash, random_coupon_hash, purchase_week_index) in purchase_by_user_coupon_week):
            if days_accessible(purchasing_user, random_coupon, purchase_week_start) == 0:
                accessibility_misses += 1
            else:
                purchase_misses += 1
                
            random_coupon_hash = random_list_element(random_state, coupon_accessibility_by_week[purchase_week_index])
            random_coupon = coupon[random_coupon_hash]

        f = features(user_history[purchasing_user_hash], random_coupon, purchase_week_start)
        score = (regressors[0][1].predict([f]))[0]
        if score > max_score:
            max_score = score
            f_max = f
            user_hash_max = purchasing_user_hash
            coupon_hash_max = random_coupon_hash

    negative_features.append(f_max)
    negative_users.append(user_hash_max)
    negative_coupons.append(coupon_hash_max)
    
    if i % 100 == 0:
        logger.info('{0}; last probability was {1}'.format(i, max_score))

logger.info('From {0} negative cases: misses due to accessibility: {1}; misses due to purchase {2}'.format(n_negative, accessibility_misses, purchase_misses))
# negative_weight = float(n_negative+1)/float(purchase_misses) * n_positive/n_negative
    
train_features = positive_features[:positive_train_size] + negative_features[:negative_train_size]
train_outcomes = positive_train_size * [1] + negative_train_size * [0]
train_weights = positive_train_size * [1] + negative_weights[:negative_train_size]
    
test_features = positive_features[positive_train_size:] + negative_features[negative_train_size:]
test_users = positive_users[positive_train_size:] + negative_users[negative_train_size:]
test_coupons = positive_coupons[positive_train_size:] + negative_coupons[negative_train_size:]
test_outcomes = positive_test_size * [1] + negative_test_size * [0]

for name, regressor in regressors:
    regressor.set_params(n_estimators=n_estimators2)
    
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

        
def score_users(user_list, week_start_date, coupon_list, submission_file_name='submission.csv', probability_file_name='probabilities.csv'):
    KEEP = 10
    with open(submission_file_name, "w") as submission_file, \
         open(probability_file_name, "w") as probability_file:
        
        writer = csv.writer(submission_file)
        probability_writer = csv.writer(probability_file)
        
        writer.writerow(('USER_ID_hash', 'PURCHASED_COUPONS'))
        probability_writer.writerow(['USER_ID_hash'] + list(itertools.chain(*[('probability_{0}'.format(i), 'COUPON_ID_hash_{0}'.format(i)) for i in range(KEEP)])))
        
        for user_hash in user_list:
            a_user_history = user_history[user_hash]
            a_user = a_user_history['user']
            
            feature_hash_pairs = tuple(( (features(a_user_history, coupon, week_start_date),
                                          coupon['COUPON_ID_hash'])
                                         for coupon in coupon_list
                                         if days_accessible(a_user, coupon, week_start_date) > 0))
            
            if len(feature_hash_pairs) > 0:
                f, h = zip(*feature_hash_pairs)
                probabilities = regressor.predict(f)
                p_and_h = sorted(zip(probabilities, h), reverse=True)
            else:
                p_and_h = ()
                
            winners = ' '.join((h for p,h in p_and_h[:10] if p > 0))
            writer.writerow((user_hash, winners))
            probability_writer.writerow([user_hash] + list(itertools.chain(*p_and_h[:10])))

logger.info('Scoring and writing output files.')

if args.validate:
    logger.info('Writing validation week purchases...')
    with open('validation_purchases.csv', 'w') as validation_purchase_file:
        writer = csv.writer(validation_purchase_file)
        writer.writerow(('USER_ID_hash', 'PURCHASES'))
        for user in sorted(validation_purchase.keys()):
            writer.writerow([user, ' '.join(sorted(validation_purchase[user]))])
    logger.info('Finished writing validation_purchases')

    logger.info('Scoring validation week users/purchases...')
    score_users(sorted(validation_purchase.keys()),
                train_end_date,
                validation_coupon.values(),
                submission_file_name='validation.csv',
                probability_file_name='validation_probabilities.csv')

if args.score:
    logger.info('Scoring test week users/purchases...')
    score_users(sorted(user_history.keys()), test_start_date, coupon_test.values())

logger.info('Finished.')
