import argparse
import average_precision
import codecs
import logging
import sys

# Logging
logger = logging.getLogger('')
# logging.basicConfig(format='%(asctime)-15s %(name)s %(message)s')
logging.basicConfig(format='%(message)s')

logger.setLevel(logging.DEBUG)

import util

parser = argparse.ArgumentParser(description='Score a set of predictions against a set of purchases.')
parser.add_argument('purchases', help='File containing purchases.')
parser.add_argument('predictions', help='File containing predictions.')
args = parser.parse_args()

purchases = util.read_file(args.purchases, index='USER_ID_hash')
predictions = util.read_file(args.predictions, index='USER_ID_hash')

if len(purchases) != len(predictions):
    logger.error('Mismatch in number of records')
    sys.exit(1)

for k,v in purchases.items():
    v['PURCHASES'] = v['PURCHASES'].split()

for k,v in predictions.items():
    v['PURCHASED_COUPONS'] = v['PURCHASED_COUPONS'].split()

(purchases,predictions) = zip(*[(purchases[k]['PURCHASES'],predictions[k]['PURCHASED_COUPONS']) for k in sorted(purchases.keys())])

logger.info ('mapk(10): {0:8.5f}'.format(average_precision.mapk(purchases, predictions)))




    
