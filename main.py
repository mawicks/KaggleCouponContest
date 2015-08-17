import codecs
import collections
import csv
import datetime
from functools import reduce
import itertools
import logging
import operator
import math
import numpy
import random
import sklearn
import sklearn.ensemble
import sklearn.ensemble.weight_boosting
import sys

# Tunable parameters
N = 180000
n_estimators = 4000
min_samples_leaf = 1 + int(0.00025*N)
min_samples_leaf = 5
max_features = 9
n_jobs=-1
oob_score=False

# Random number seeds
random_state = random.Random(123456)
random_state.seed(12345)

classifier_random_state = numpy.random.RandomState(seed=987654)
classifier_random_state.seed(12345)


# Important constants
sample_start_date = datetime.datetime(year=2011, month=7, day=3, hour=0, minute=0)

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

    ('RandomForestClassifer',
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
regressors = regressors[1:2]

def week_index(date):
    return int((date - sample_start_date).days / 7)

def start_of_week(date):
    return sample_start_date + week_index(date)*datetime.timedelta(days=7)

logger = logging.getLogger(__name__)
FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.DEBUG)

logger.info('Parameters: '
            'N: {0}, '
            'n_estimators: {1}, '
            'min_samples_leaf: {2}, '
            'max_features: {3}, '
            'n_jobs: {4}'.format(N,
                                 n_estimators,
                                 min_samples_leaf,
                                 max_features,
                                 n_jobs
                             ))

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
            result = collections.OrderedDict(
                (o.__getattribute__(index), o)
                for o in ( Type._make((m(i) for m,i in zip(item_mapper, line))) for line in reader )
            )
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
        result = float(int(earth_distance((loc1.LATITUDE,loc1.LONGITUDE), (loc2.LATITUDE,loc2.LONGITUDE))))
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

purchase = collections.OrderedDict (
    (k, v._replace(USER_ID_hash=user[v.USER_ID_hash],
                   COUPON_ID_hash=coupon[v.COUPON_ID_hash]))
    for k,v in read_file('coupon_detail_train.csv', 'Purchase', 'PURCHASEID_hash').items()
)

visit = tuple( v._replace(USER_ID_hash=user[v.USER_ID_hash],
                           VIEW_COUPON_ID_hash=coupon.get(v.VIEW_COUPON_ID_hash, missing_coupon),
                           PURCHASEID_hash=purchase.get(v.PURCHASEID_hash, None))
                for v in read_file('coupon_visit_train.csv', 'Visit') )

train_area = read_file('coupon_area_train.csv', 'Area')
test_area = read_file('coupon_area_test.csv', 'Area')


prefecture_set = set()
large_area_name_set = set()
ken_name_set = set()
small_area_name_set = set()
capsule_text_set = set()
genre_name_set = set()

# Scan through data to build indexes for frequently accessed data
# and to accumulate any other useful statistics:
logger.info('Scanning data...')

first_purchase_date = min((p.I_DATE for p in purchase.values()))
last_purchase_date = max((p.I_DATE for p in purchase.values()))

def range_violation(user, coupon, date):
    start_date = max(user.REG_DATE, coupon.DISPFROM)
    end_date = min(user.WITHDRAW_DATE, coupon.DISPEND)
    
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
    purchase_by_user.setdefault(p.USER_ID_hash.USER_ID_hash, []).append(p)
    purchase_by_user_coupon_week[(p.USER_ID_hash.USER_ID_hash, p.COUPON_ID_hash.COUPON_ID_hash, week_index(p.I_DATE))] = 1

    # Verify the assumption that purchases do not occur outside the ad display window or when the user is not registered.
    rv = range_violation(user[p.USER_ID_hash.USER_ID_hash], p.COUPON_ID_hash, p.I_DATE)
    if rv:
        logger.warning('*** Sale w/o display: '
                       '{0} days, {1} hours, {2} minutes, {3} seconds {4}'.format(rv[1].days,
                                                                                  rv[1].seconds//3600,
                                                                                  (rv[1].seconds % 3600) // 60,
                                                                                  rv[1].seconds % 60,
                                                                                  'after close' if rv[0]>0 else 'before open'))
        logger.warning('\tpurchase date: {0}'.format(p.I_DATE))
        logger.warning('\tuser:  {0} reg date range: {1} to {2}'.format(p.USER_ID_hash.USER_ID_hash, p.USER_ID_hash.REG_DATE, p.USER_ID_hash.WITHDRAW_DATE))
        logger.warning('\tcoupon:  {0} disp date range: {1} to {2}'.format(p.COUPON_ID_hash.COUPON_ID_hash, p.COUPON_ID_hash.DISPFROM, p.COUPON_ID_hash.DISPEND))

# Visit
visit_by_user = collections.OrderedDict()
for v in visit:
    visit_by_user.setdefault(v.USER_ID_hash.USER_ID_hash, []).append(v)

# Area
small_area_by_coupon = collections.OrderedDict()
for a in itertools.chain(train_area, test_area):
    small_area_by_coupon.setdefault(a.COUPON_ID_hash, set()).add(a.SMALL_AREA_NAME)
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
user_history = collections.OrderedDict()
UserHistory = collections.namedtuple('UserHistory', ['user', 'visit', 'purchase'])
for h,u in user.items():
    user_history[h] = UserHistory(user=u, visit=visit_by_user.get(h, []), purchase=purchase_by_user.get(h, []))
    prefecture_set.add(u.PREF_NAME)

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
        return ('week_index',)
    
    def map (self, user_history, coupon, date):
        return (week_index(date),)

def days_accessible(user, coupon, week_start_date):
    """Returns the number of days the coupon was accessible to the user during
    the week beginning on week_start_date"""

    start = max(week_start_date, user.REG_DATE, coupon.DISPFROM)
    end = min(week_start_date+datetime.timedelta(days=7), user.WITHDRAW_DATE, coupon.DISPEND)
    
    interval = (end-start).total_seconds()/86400.0
    return interval if interval > 0 else 0

class AvailabilityFeatureSet:
    def names(self):
        return ('days_accessible_by_user',)

    def map (self, user_history, coupon, date):
        return (days_accessible(user_history.user, coupon, date),)

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

            'recency_small_area_purchase',
            'recency_large_area_purchase',
            'recency_ken_purchase',

            'percent_from_genre_purchase',
            'percent_from_capsule_purchase',

            'centered_discount_price',
            'centered_price_rate',

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

        for p in user_history.purchase:
            # Consider only the past
            if p.I_DATE < date:
                second_last_purchase = last_purchase
                last_purchase = p.I_DATE
                days = (date - last_purchase).days

                min_price_rate = min(min_price_rate, p.COUPON_ID_hash.PRICE_RATE)
                max_discount_price = max(max_discount_price, p.COUPON_ID_hash.PRICE_RATE)

                purchase_count += 1
                sum_discount_price += p.COUPON_ID_hash.DISCOUNT_PRICE
                sum_price_rate += p.COUPON_ID_hash.PRICE_RATE

                if p.COUPON_ID_hash.GENRE_NAME == coupon.GENRE_NAME:
                    genre_count += 1
                    days_since_genre = min(days_since_genre, days)
                    max_qty_genre = max(max_qty_genre, p.ITEM_COUNT)

                if p.COUPON_ID_hash.CAPSULE_TEXT == coupon.CAPSULE_TEXT:
                    capsule_count += 1
                    days_since_capsule = min(days_since_capsule, days)
                    max_qty_capsule = max(max_qty_capsule, p.ITEM_COUNT)

                if p.COUPON_ID_hash.small_area_name == coupon.small_area_name:
                    days_since_small_area_name = min(days_since_small_area_name, days)
                    max_qty_small_area = max(max_qty_small_area, p.ITEM_COUNT)

                if p.COUPON_ID_hash.large_area_name == coupon.large_area_name:
                    days_since_large_area_name = min(days_since_large_area_name, days)
                    max_qty_large_area = max(max_qty_large_area, p.ITEM_COUNT)

                if p.COUPON_ID_hash.ken_name == coupon.ken_name:
                    days_since_ken_name = min(days_since_ken_name, days)
                    max_qty_ken = max(max_qty_ken, p.ITEM_COUNT)

                previous_purchase_area.add(p.SMALL_AREA_NAME)

        days_since_purchase = (date - last_purchase).days

        result = (
            purchase_count,
            genre_count,
            capsule_count,

            days_since_purchase,
            (date - second_last_purchase).days,
            
            days_since_genre,
            days_since_capsule,
            
            days_since_small_area_name,
            days_since_large_area_name,
            days_since_ken_name,

            days_since_purchase - days_since_small_area_name,
            days_since_purchase - days_since_large_area_name,
            days_since_purchase - days_since_ken_name,

            float(genre_count) / purchase_count if purchase_count > 0 else -999,
            float(capsule_count) / purchase_count if purchase_count > 0 else -999,

            coupon.DISCOUNT_PRICE - float(sum_discount_price)/ purchase_count if purchase_count > 0 else -999,
            coupon.PRICE_RATE - float(sum_price_rate)/ purchase_count if purchase_count > 0 else -999,

            coupon.DISCOUNT_PRICE - max_discount_price,
            coupon.PRICE_RATE - min_price_rate,

            len(small_area_by_coupon.get(coupon.COUPON_ID_hash, set()).intersection(previous_purchase_area)),

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

            'recency_small_area_visit',
            'recency_large_area_visit',
            'recency_ken_visit',
            
#            'days_since_coupon_visit',

            'percent_from_genre_visit',
            'percent_from_capsule_visit',

            'peak_hour',
            'peak_dow',
        )

    def map(self, user_history, coupon, date):
        genre_count = 0
        capsule_count = 0
        days_since_visit = days_since_capsule = days_since_genre = days_since_coupon = 1 << 31
        days_since_small_area_name = days_since_large_area_name = days_since_ken_name = 1 << 31
        visit_count = 0

        peak_dow = peak_hour = -1
        peak_dow_count = peak_hour_count = 0
        
        hour_count = {}
        dow_count = {}
        
        for v in user_history.visit:
            # Consider only the past
            if v.I_DATE < date:
                days = (date - v.I_DATE).days
                days_since_visit = min(days_since_visit, days)
                visit_count += 1

                hour = v.I_DATE.hour
                hour_count[hour] = hour_count.get(hour, 0) + 1

                dow = v.I_DATE.weekday()
                dow_count[dow] = dow_count.get(dow, 0) + 1

                if v.VIEW_COUPON_ID_hash.GENRE_NAME == coupon.GENRE_NAME:
                    genre_count += 1
                    days_since_genre = min(days_since_genre, days)

                if v.VIEW_COUPON_ID_hash.CAPSULE_TEXT == coupon.CAPSULE_TEXT:
                    capsule_count += 1
                    days_since_capsule = min(days_since_capsule, days)

                if v.VIEW_COUPON_ID_hash.small_area_name == coupon.small_area_name:
                    days_since_small_area_name = min(days_since_small_area_name, days)

                if v.VIEW_COUPON_ID_hash.large_area_name == coupon.large_area_name:
                    days_since_large_area_name = min(days_since_large_area_name, days)

                if v.VIEW_COUPON_ID_hash.ken_name == coupon.ken_name:
                    days_since_ken_name = min(days_since_ken_name, days)

                if v.VIEW_COUPON_ID_hash.COUPON_ID_hash == coupon.COUPON_ID_hash:
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

            days_since_visit - days_since_small_area_name,
            days_since_visit - days_since_large_area_name,
            days_since_visit - days_since_ken_name,

#            days_since_coupon,

            float(genre_count) / visit_count if visit_count > 0 else 0,
            float(capsule_count) / visit_count if visit_count > 0 else 0,

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
        user = user_history.user
        return (user.AGE,
                gender_encoder.map(user.SEX_ID),
                prefecture_encoder.map(user.PREF_NAME),
                (date - user.REG_DATE).days)


class SimpleCouponFeatureSet:
    def names(self):
        return ('capsule_text', 'genre_name',
                'large_area_name', 'ken_name', 'small_area_name',
                'price_rate', 'catalog_price', 'discount_price',
                'price_reduction',
                'valid_period',
                'days_on_display',
                'display_days_left',
                'days_until_valid',
                'days_until_expiration',
        )

    def map (self, user_history, coupon, date):
        return (
            capsule_encoder.map(coupon.CAPSULE_TEXT),
            genre_name_encoder.map(coupon.GENRE_NAME),
            large_area_name_encoder.map(coupon.large_area_name),
            prefecture_encoder.map(coupon.ken_name),
            small_area_name_encoder.map(coupon.small_area_name),
            coupon.PRICE_RATE,
            coupon.CATALOG_PRICE,
            coupon.DISCOUNT_PRICE,
            coupon.CATALOG_PRICE-coupon.DISCOUNT_PRICE,
            coupon.VALIDPERIOD,
            (date - coupon.DISPFROM).days,
            (coupon.DISPEND - date).days,
            (coupon.VALIDFROM - date).days,
            (coupon.VALIDEND - date).days,
        )

class CouponUsableDateFeatureSet:
    def names(self):
        return (
#            'usable_date_mon',
#            'usable_date_tue',
#            'usable_date_wed',
#            'usable_date_thu',
#            'usable_date_fri',
#            'usable_date_sat',
#            'usable_date_sun',
#            'usable_date_holiday',
#           'usable_date_before_holiday',
            'usable_date_weekend',
            'usable_date_sun_and_holiday',
        )

    def map (self, user_history, coupon, date):
        return (
#            coupon.USABLE_DATE_MON,
#            coupon.USABLE_DATE_TUE,
#            coupon.USABLE_DATE_WED,
#            coupon.USABLE_DATE_THU,
#            coupon.USABLE_DATE_FRI,
#            coupon.USABLE_DATE_SAT,
#            coupon.USABLE_DATE_SUN,
#            coupon.USABLE_DATE_HOLIDAY,
#            coupon.USABLE_DATE_BEFORE_HOLIDAY,
            coupon.USABLE_DATE_SAT+coupon.USABLE_DATE_SUN,
            ( coupon.USABLE_DATE_SUN + coupon.USABLE_DATE_HOLIDAY + coupon.USABLE_DATE_BEFORE_HOLIDAY ),
        )

class JointFeatureSet:
    def names(self):
        return ('distance',)
    
    def map (self, user_history, coupon, date):
        return (prefecture_distance(user_history.user.PREF_NAME, coupon.ken_name),)

# dump(user_history)

feature_extractors = (
    ExperimentalFeatureSet(),
    AvailabilityFeatureSet(),
    UserPurchaseHistoryFeatureSet(),
    UserVisitHistoryFeatureSet(),
    SimpleUserFeatureSet(),
    SimpleCouponFeatureSet(),
    CouponUsableDateFeatureSet(),
    JointFeatureSet(),
    RandomFeatureSet()
)

feature_names = reduce(operator.add, (fe.names() for fe in feature_extractors))

def features(user_history, coupon, date):
    start_of_week_date = start_of_week(date)
    x = reduce (operator.add,
                (fe.map(user_history,
                        coupon,
                         start_of_week_date) for fe in feature_extractors))
    if len(x) != len(feature_names):
        logger.error('len(features_names) is {0} but len of feature vector is {1}'.format(len(feature_names), len(x)))

    return x

# Resample the probability space to get failed cases
# The space is assumed to be (user, coupon, date) tuples
# where date is in the overlap region where the user is registered and the coupon is displayed
user_history_list = tuple(user_history.values())
coupon_list = tuple(coupon.values())

random_state.seed(123456)
purchase_sample = random_state.sample(tuple(purchase.values()), N//2)

sample_features = []
for p in purchase_sample:
    f = features(user_history[p.USER_ID_hash.USER_ID_hash], p.COUPON_ID_hash, start_of_week(p.I_DATE))
    sample_features.append(f)

nonpurchase_sample = []

logger.info ('Sampling outcome space to obtain some non-purchase outcomes')
nonpurchase_count = 0
saw_a_one = False

while nonpurchase_count < N//2:
    random_user_history = user_history_list[random_state.randrange(len(user_history_list))]
    random_coupon = coupon_list[random_state.randrange(len(coupon_list))]
    random_user = random_user_history.user

    random_week_index = random.randrange(51) # Only 51 weeks in training data; 52nd week is test set.
    random_week_start = sample_start_date + datetime.timedelta(days=7*random_week_index)

    if days_accessible(random_user, random_coupon, random_week_start) > 0:
        result = purchase_by_user_coupon_week.get((random_user.USER_ID_hash, random_coupon.COUPON_ID_hash, random_week_index), 0)
        if result == 0:
            f = features(random_user_history, random_coupon, random_week_start)
            sample_features.append(f)
            nonpurchase_count += 1

sample_outcomes = (N//2) * [1.0] + nonpurchase_count * [0.0]

features_and_outcomes = list(zip(sample_features,sample_outcomes))
random_state.shuffle(features_and_outcomes)

train_size = (2*N) // 3
train_features,train_outcomes = zip(*features_and_outcomes[0:train_size])
test_features,test_outcomes = zip(*features_and_outcomes[train_size:])
del features_and_outcomes

logger.info('{0} features'.format(len(feature_names)))
for name, regressor in regressors:
    logger.info('Training {0}: {1}'.format(name, regressor))

    regressor.fit(train_features, train_outcomes)
    test_prediction = regressor.predict(test_features)

    if hasattr(regressor, 'feature_importances_'):
        logger.info('Importances:')
        for i,n in sorted(zip(regressor.feature_importances_, feature_names), reverse=True):
            logger.info('{0:>30}: {1:6.4f}'.format(n, i))
        
    logger.info('Performance:')
    logger.info('{0:>24}: {1:7.5f}/{2:7.5f}'.format('min/max test prediction', min(test_prediction), max(test_prediction)))
    logger.info('{0:>24}: {1:6.4f}'.format('Default classifier/regressor score', regressor.score(test_features, test_outcomes)))
    logger.info('{0:>24}: {1:6.4f}'.format('MSE', sklearn.metrics.mean_squared_error(test_outcomes, test_prediction)))
    logger.info('{0:>24}: {1:5.3f}'.format('auroc', sklearn.metrics.roc_auc_score(test_outcomes, test_prediction)))
    logger.info('{0:>24}: {1:5.3f}'.format('log loss', sklearn.metrics.log_loss(test_outcomes, test_prediction)))


logger.info('Scoring and writing output file.')
coupon_test = read_file('coupon_list_test.csv', 'Coupon', 'COUPON_ID_hash')

with open("submission.csv", "w") as outputfile:
    writer = csv.writer(outputfile)
    writer.writerow(('USER_ID_hash', 'PURCHASED_COUPONS'))

    test_week_start_date = datetime.datetime(year=2012, month=6, day=24)

    for user_hash in sorted(user_history.keys()):
        a_user_history = user_history[user_hash]
        a_user = a_user_history.user

        feature_hash_pairs = tuple(( (features(a_user_history, coupon, test_week_start_date),
                                      coupon.COUPON_ID_hash)
                                     for coupon in coupon_test.values()
                                     if days_accessible(a_user, coupon, test_week_start_date) > 0))
        
        if len(feature_hash_pairs) > 0:
            f, h = zip(*feature_hash_pairs)
            probabilities = regressor.predict(f)
            p_and_h = sorted(zip(probabilities, h), reverse=True)
        else:
            p_and_h = ()
            
        winners = ' '.join((h for p,h in p_and_h[0:10] if p > 0))
        writer.writerow((user_hash, winners))

logger.info('Finished.')
