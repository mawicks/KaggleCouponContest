import datetime
import functools
import naive_bayes
import operator


class CacheableWrapper:
    """Naive Bayes Wrapper"""
    def __init__ (self, strategy, user_history_index, coupon_index, purchase_or_visit, field_name):
        """strategy is MNStrategy or BNStrategy
           purchase_or_visit ('purchase' or 'visit') identifies which type of history
           to accumulate stats on"""

        self.strategy = strategy

        self.user_history_index = user_history_index
        self.coupon_index = coupon_index
        
        self.purchase_or_visit = purchase_or_visit
        self.field_name = field_name

        self.history_getter = operator.itemgetter(purchase_or_visit)
        self.field_getter = operator.itemgetter(field_name)

        self.estimator = naive_bayes.Estimator(strategy)
        
    def __repr__ (self):
        return "CacheableWrapper({0},{1},{2},{3},{4})".format(self.strategy,
                                                              self.user_history_index,
                                                              self.coupon_index,
                                                              self.purchase_or_visit,
                                                              self.field_name)
    @functools.lru_cache(maxsize=256)
    def accumulate_history(self, user_hash, date):
        history = {}
        for history_item in self.history_getter(self.user_history_index[user_hash]):
            if history_item['I_DATE'] < date:
                fv = self.field_getter(history_item['COUPON'])
                history[fv] = history.get(fv, 0) + 1
        return history

    def add (self, purchase, user_hash):
        field_value = self.field_getter(purchase['COUPON'])
        history_set = self.accumulate_history(user_hash, purchase['I_DATE'])
        self.estimator.add(field_value, history_set)
                
    def dump (self, limit=None):
        print ('{0}: {1}'.format(self.purchase_or_visit, self.field_name))

        self.estimator.dump(limit)

    @functools.lru_cache(maxsize=256)
    def __score(self, candidate_field_value, user_hash, date):
        history_set = self.accumulate_history(user_hash, date)
        return self.estimator.score(candidate_field_value, history_set)
                                    
    def score (self, coupon_hash, user_hash, date):
        candidate_field_value = self.field_getter(self.coupon_index[coupon_hash])
        return self.__score(candidate_field_value, user_hash, date)
    
class Wrapper:
    """Naive Bayes Wrapper"""
    def __init__ (self, strategy, purchase_or_visit, field_name):
        """strategy is MNStrategy or BNStrategy
           purchase_or_visit ('purchase' or 'visit') identifies which type of history
           to accumulate stats on"""

        self.strategy = strategy
        self.field_name = field_name
        self.purchase_or_visit = purchase_or_visit

        self.history_getter = operator.itemgetter(purchase_or_visit)
        self.field_getter = operator.itemgetter(field_name)

        self.estimator = naive_bayes.Estimator(strategy)
        
    def __repr__ (self):
        return "Wrapper({0},{1},{2})".format(self.strategy, self.purchase_or_visit, self.field_name)

    def accumulate_history(self, history_list, date):
        history = {}
        for history_item in history_list:
            if history_item['I_DATE'] < date:
                fv = self.field_getter(history_item['COUPON'])
                history[fv] = history.get(fv, 0) + 1
        return history
            
    def add (self, purchase, user_history):
        field_value = self.field_getter(purchase['COUPON'])

        history_list = self.history_getter(user_history)
        history_set = self.accumulate_history(history_list, purchase['I_DATE'])

        self.estimator.add(field_value, history_set)
                
    def dump (self, limit=None):
        print ('{0}: {1}'.format(self.purchase_or_visit, self.field_name))

        self.estimator.dump(limit)

    def score (self, coupon, user_history, date):
        candidate_field_value = self.field_getter(coupon)

        history_list = self.history_getter(user_history)
        history_set = self.accumulate_history(history_list, date)

        return self.estimator.score(candidate_field_value, history_set)
        
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

    accumulator = Wrapper(naive_bayes.BNStrategy(), 'visit', 'city')
    mn_accumulator = Wrapper(naive_bayes.MNStrategy(), 'visit', 'city')
    
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

