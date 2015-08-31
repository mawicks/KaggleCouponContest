import collections
import csv
import datetime
import logging

logger = logging.getLogger(__name__)

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
