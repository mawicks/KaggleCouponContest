import datetime
import functools
import logging
import naive_bayes
import operator

logger = logging.getLogger(__name__)

class CacheableWrapper:
    """Naive Bayes Wrapper"""
    def __init__ (self, strategy, user_history_index, coupon_index, purchase_or_visit, class_name, attribute_name, from_coupon=True):
        """strategy is MNStrategy or BNStrategy
           purchase_or_visit ('purchase' or 'visit') identifies which type of history
           to accumulate stats on"""

        self.strategy = strategy

        self.user_history_index = user_history_index
        self.coupon_index = coupon_index
        
        self.purchase_or_visit = purchase_or_visit
        self.class_name = class_name
        self.attribute_name = attribute_name

        self.history_getter = operator.itemgetter(purchase_or_visit)
        self.class_getter = operator.itemgetter(class_name)

        self.from_coupon = from_coupon

        if from_coupon:
            self.attribute_getter = lambda x: x['COUPON'][attribute_name]
        else:
            self.attribute_getter = operator.itemgetter(attribute_name)

        self.estimator = naive_bayes.Estimator(strategy)
        
    def __repr__ (self):
        return "CacheableWrapper({0},{1},{2},{3},{4},{5},from_coupon={6})".format(self.strategy,
                                                                                  self.user_history_index,
                                                                                  self.coupon_index,
                                                                                  self.purchase_or_visit,
                                                                                  self.class_name,
                                                                                  self.attribute_name,
                                                                                  self.from_coupon
        )
    @functools.lru_cache(maxsize=256)
    def accumulate_history(self, user_hash, date):
        history = {}
        for history_item in self.history_getter(self.user_history_index[user_hash]):
            if history_item['I_DATE'] < date:
                av = self.attribute_getter(history_item)
                history[av] = history.get(av, 0) + 1
        return history

    def add (self, purchase, user_hash):
        class_value = self.class_getter(purchase['COUPON'])
        history_set = self.accumulate_history(user_hash, purchase['I_DATE'])
        self.estimator.add(class_value, history_set)
                
    def dump (self, limit=None):
        logger.debug('{0}: {1}/{2}'.format(self.purchase_or_visit, self.class_name, self.attribute_name))

        self.estimator.dump(limit)

    @functools.lru_cache(maxsize=256)
    def __score(self, candidate_class_value, user_hash, date):
        history_set = self.accumulate_history(user_hash, date)
        return self.estimator.score(candidate_class_value, history_set)
                                    
    def score (self, coupon_hash, user_hash, date):
        candidate_class_value = self.class_getter(self.coupon_index[coupon_hash])
        return self.__score(candidate_class_value, user_hash, date)
    
class SimpleNBWrapper:
    """Naive Bayes Wrapper"""
    def __init__ (self, user_history_index, coupon_index, user_attribute_name, coupon_class_name):
        self.user_history_index = user_history_index
        self.coupon_index = coupon_index

        self.user_attribute_name = user_attribute_name
        self.coupon_class_name = coupon_class_name

        self.user_attribute_getter = operator.itemgetter(user_attribute_name)
        self.coupon_class_getter = operator.itemgetter(coupon_class_name)

        self.estimator = naive_bayes.Estimator(naive_bayes.BNStrategy())
        
    def __repr__ (self):
        return "SimpleNBWrapper({0},{1},{2},{3})".format(self.user_history_index,
                                                         self.coupon_index,
                                                         self.user_attribute_name,
                                                         self.coupon_class_name)
    def add (self, purchase, user_hash):
        class_value = self.coupon_class_getter(purchase['COUPON'])
        user_value = self.user_attribute_getter(self.user_history_index[user_hash]['user'])
        self.estimator.add(class_value, {user_value: 1})
                
    def dump (self, limit=None):
        logger.debug('{0}/{1}'.format(self.coupon_class_name, self.user_attribute_name))

        self.estimator.dump(limit)

    @functools.lru_cache(maxsize=256)
    def __score(self, candidate_class_value, user_hash):
        user_value = self.user_attribute_getter(self.user_history_index[user_hash]['user'])
        return self.estimator.score(candidate_class_value, {user_value: 1})
                                    
    def score (self, coupon_hash, user_hash, date):
        candidate_class_value = self.coupon_class_getter(self.coupon_index[coupon_hash])
        return self.__score(candidate_class_value, user_hash)
    
if __name__ == "__main__":

    import collections

    def Coupon(city):
        return { 'city': city }

    def Visit(coupon, date):
        return { 'COUPON': coupon, 'I_DATE': date }

    def Purchase(coupon, date):
        return { 'COUPON': coupon, 'I_DATE': date }

    def History(visit):
        return { 'visit': visit }
    
    coupon_arlington = Coupon('Arlington')
    coupon_alexandria = Coupon('Alexandria')
    coupon_dc = Coupon('DC')
    coupon_orlando = Coupon('Orlando')

    all_coupons = [coupon_arlington, coupon_alexandria, coupon_dc, coupon_orlando]

    view_date = datetime.datetime(year=2012, month=1, day=1)
    purchase_date = datetime.datetime(year=2014, month=1, day=1)
    test_date = datetime.datetime(year=2014, month=2, day=1)
    
    visit_arlington = Visit(coupon_arlington, view_date)
    visit_dc = Visit(coupon_dc, view_date)
    visit_alexandria = Visit(coupon_alexandria, view_date)
    visit_orlando = Visit(coupon_orlando, view_date)

    purchases = [
        (Purchase(coupon_arlington, purchase_date), History(tuple(5 * [visit_arlington]))),
        (Purchase(coupon_orlando, purchase_date), History(tuple(5 * [visit_orlando]))),
        (Purchase(coupon_alexandria, purchase_date), History((visit_arlington, visit_alexandria, visit_alexandria, visit_alexandria, visit_dc))),
        (Purchase(coupon_alexandria, purchase_date),History((visit_arlington, visit_alexandria, visit_arlington, visit_alexandria))), 
        (Purchase(coupon_dc, purchase_date),History((visit_arlington, visit_arlington, visit_dc, visit_dc, visit_dc))), 
        (Purchase(coupon_arlington, purchase_date),History((visit_dc,))),
    ]

    accumulator = CacheableWrapper(naive_bayes.BNStrategy(), 'visit', 'city')
    mn_accumulator = CacheableWrapper(naive_bayes.MNStrategy(), 'visit', 'city')
    
    for coupon_purchased, history in purchases:
        accumulator.add(coupon_purchased, history)
        mn_accumulator.add(coupon_purchased, history)

    accumulator.dump()
    mn_accumulator.dump()
                                
    test_histories = [
        History((visit_arlington, visit_dc)),
        History(tuple(3 * [visit_orlando])),
        History((visit_orlando, visit_dc, visit_orlando)),
    ]

    for test_history in test_histories:
        print ('\n**** NEW HISTORY ******')
        for coupon in all_coupons:
            print('scoring:\n\t{0} against\n\t{1}'.format(coupon, test_history))
            print('\tBernoulli result:{0}'.format(accumulator.score(coupon, test_history, test_date)))
            print('\tMultinomial result:{0}\n\n'.format(mn_accumulator.score(coupon, test_history, test_date)))

