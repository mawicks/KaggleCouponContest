import codecs
import collections
import csv
import datetime
from functools import reduce
import logging
import operator
import math
import numpy
import random
import sklearn
import sklearn.ensemble
import sys

# Parameters
random_state = random.Random(123456)
random_state.seed(12345)
classifier_random_state = numpy.random.RandomState(seed=987654)
classifier_random_state.seed(12345)

N = 20000
n_estimators = 200

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
                        
    return timestamp.date()

def date_mapper_min(d):
    if d == 'NA':
        timestamp = datetime.datetime.min
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

area = read_file('coupon_area_train.csv', 'Area')


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

def is_displayable(user, coupon, date):
    start_date = max(user.REG_DATE, coupon.DISPFROM)
    end_date = min(user.WITHDRAW_DATE, coupon.DISPEND)
    return 1 if date >= start_date and date <= end_date else 0

logger.info('First/last purchase dates: {0}/{1}'.format(first_purchase_date, last_purchase_date))

# Purchase
purchase_by_user = collections.OrderedDict()
purchase_by_user_coupon_date = collections.OrderedDict()
for p in purchase.values():
    purchase_by_user.setdefault(p.USER_ID_hash.USER_ID_hash, []).append(p)
    purchase_by_user_coupon_date[(p.USER_ID_hash.USER_ID_hash, p.COUPON_ID_hash.COUPON_ID_hash, p.I_DATE)] = 1

    # Verify the assumption that purchases do not occur outside the ad display window or when the user is not registered.
    if not is_displayable(user[p.USER_ID_hash.USER_ID_hash], p.COUPON_ID_hash, p.I_DATE):
        logger.warn('Sale w/o display?  \n\tuser:{0}, \n\tcoupon:{1}, \n\tdate:{2}'.format(p.USER_ID_hash, p.COUPON_ID_hash, p.I_DATE))

# Visit
visit_by_user = collections.OrderedDict()
for v in visit:
    visit_by_user.setdefault(v.USER_ID_hash.USER_ID_hash, []).append(v)

# Area
area_by_coupon = collections.OrderedDict()
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
user_history = collections.OrderedDict()
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

class RandomFeatureSet:
    def names(self):
        return ('random_feature',)

    def map (self, user_history, coupon, date):
        return (random_state.randrange(2),)

class SimpleDateFeatureSet:
    def names(self):
        return ('day_of_week',)

    def map(self, user_history, coupon, date):
        return (date.isoweekday(),)

class UserHistoryFeatureSet:
    def names(self):
        return('number_of_genre_purchases',
               'number_of_capsule_purchases',
               'days_since_genre',
               'days_since_capsule',
               'percent_from_genre',
               'percent_from_capsule'
        )

    def map(self, user_history, coupon, date):
        genre_count = 0
        capsule_count = 0
        days_since_capsule = days_since_genre = 1 << 31
        purchase_count = 0

        for p in user_history.purchase:
            days = (date - p.I_DATE).days
            if p.I_DATE < date:
                purchase_count += 1
            
            if (p.COUPON_ID_hash.GENRE_NAME == coupon.GENRE_NAME and
                p.I_DATE < date):
                genre_count += 1
                days_since_genre = min(days_since_genre, days)
                
            if (p.COUPON_ID_hash.CAPSULE_TEXT == coupon.CAPSULE_TEXT and
                p.I_DATE < date):
                capsule_count += 1
                days_since_capsule = min(days_since_capsule, days)
                
        result = (float(genre_count),
                  float(capsule_count),
                  float(days_since_genre),
                  float(days_since_capsule),
                  float(genre_count) / purchase_count if purchase_count > 0 else 0,
                  float(capsule_count) / purchase_count if purchase_count > 0 else 0)

        return result

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
                'price_rate', 'catalog_price', 'discount_price',
                'price_reduction',
                'valid_period',
                'days_on_display', 'display_days_left',
                'days_until_valid', 'days_until_expiration')

    def map (self, user_history, coupon, date):
        return (capsule_encoder.map(coupon.CAPSULE_TEXT),
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
                (coupon.VALIDEND - date).days)

class JointFeatureSet:
    def names(self):
        return ('distance',)
    
    def map (self, user_history, coupon, date):
        return (prefecture_distance(user_history.user.PREF_NAME, coupon.ken_name),)
    
# dump(user_history)

feature_extractors = (
    UserHistoryFeatureSet(),
    SimpleUserFeatureSet(),
    SimpleCouponFeatureSet(),
    JointFeatureSet(),
    SimpleDateFeatureSet(),
    RandomFeatureSet()
)

def features(user_history, coupon, date):
    return reduce (operator.add, (fe.map(user_history,
                                         coupon,
                                         date) for fe in feature_extractors))

feature_names = reduce(operator.add, (fe.names() for fe in feature_extractors))

# Resample the probability space to get failed cases
# The space is assumed to be (user, coupon, date) tuples
# where date is in the overlap region where the user is registered and the coupon is displayed
user_history_list = tuple(user_history.values())
coupon_list = tuple(coupon.values())

random_state.seed(123456)
purchase_sample = random_state.sample(tuple(purchase.values()), N//2)

training_features = []
for p in purchase_sample:
    f = features(user_history[p.USER_ID_hash.USER_ID_hash], p.COUPON_ID_hash, p.I_DATE)
    training_features.append(f)

nonpurchase_sample = []

logger.info ('Sampling outcome space to obtain some non-purchase outcomes')
nonpurchase_count = 0
nonpurchase_count = 0
while nonpurchase_count < N//2:
    random_user_history = user_history_list[random_state.randrange(len(user_history_list))]
    random_coupon = coupon_list[random_state.randrange(len(coupon_list))]
    random_user = random_user_history.user
    
    start_date = max(first_purchase_date, random_user.REG_DATE, random_coupon.DISPFROM)
    end_date = min(last_purchase_date, random_user.WITHDRAW_DATE, random_coupon.DISPEND)
    
    if start_date <= end_date:
        random_date = start_date + datetime.timedelta(days=random_state.randrange((end_date-start_date).days+1))
        result = purchase_by_user_coupon_date.get((random_user.USER_ID_hash, random_coupon.COUPON_ID_hash, random_date), 0)
        if result == 0:
            f = features(random_user_history, random_coupon, random_date)
            training_features.append(f)
            nonpurchase_count += 1
            
training_outcomes = (N//2) * [1.0] + nonpurchase_count * [0.0]

regressor = sklearn.ensemble.RandomForestRegressor(random_state=classifier_random_state,
                                                   n_estimators=n_estimators,
                                                   min_samples_leaf=75,
                                                   oob_score=True)

logger.info('Training...')

regressor.fit(training_features, training_outcomes)
training_score = regressor.score(training_features, training_outcomes)

logger.info('Importances:')
for i,n in sorted(zip(regressor.feature_importances_, feature_names), reverse=True):
    logger.info('{0:>30}: {1:6.4f}'.format(n, i))
    
logger.info('Performance:')
logger.info('{0:>24}: {1:6.4f}/{2:6.4f}'.format('training/oob r-squared', training_score, regressor.oob_score_))
logger.info('{0:>24}: {1:7.5f}/{2:7.5f}'.format('min/max oob prediction', min(regressor.oob_prediction_), max(regressor.oob_prediction_)))
logger.info('{0:>24}: {1:5.3f}'.format('auroc', sklearn.metrics.roc_auc_score(training_outcomes, regressor.oob_prediction_)))

coupon_test = read_file('coupon_list_test.csv', 'Coupon', 'COUPON_ID_hash')

bias_correct = numpy.vectorize(lambda x: x / (x + 1e6*(1-x)))

logger.info('Scoring and writing output file.')
with open("submission.csv", "w") as outputfile:
    writer = csv.writer(outputfile)
    writer.writerow(('USER_ID_hash', 'PURCHASED_COUPONS'))
                    
    first_test_date = datetime.date(year=2012, month=6, day=24)
    for user_hash in sorted(user_history.keys()):
        a_user_history = user_history[user_hash]

        probabilities  = numpy.zeros(len(coupon_test))

        for days in range(7):
            a_date = first_test_date + datetime.timedelta(days=days)
            d, f, h = zip(*tuple(( (is_displayable(a_user_history.user, coupon, a_date),
                                    features(a_user_history, coupon, a_date),
                                    coupon.COUPON_ID_hash)
                                   for coupon in coupon_test.values() ) ))
            
            # It's okay to add probabilities here because they are small after bias correction.
            # As the final probability, we want 1 - \prod (1-p_i), but this is approximately \sum p_i.
            # Also, ignore returned probabilities when the ad isn't displayed by setting them to zero.
            # Negative examples were sampled over a set where displayed==1 and
            # there were no examples of purchases where displayed==0 in the training data.
            # The predictor will return incorrect results in this region, where there was no training data.
            # The actual probability is assumed to be zero.
            
            probabilities += numpy.array(d) * bias_correct(regressor.predict(f))

        x = sorted(zip(probabilities, h), reverse=True)
        winners = ' '.join(tuple( (y[1] for y in x[0:10]) ))
        writer.writerow((user_hash, winners))

logger.info('Finished.')
