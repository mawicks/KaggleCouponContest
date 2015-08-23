import datetime
import functools
import operator
import math


class NBAccumulator:
    """Naive Base accumulator"""
    def __init__ (self, purchase_or_visit, field_name):
        """purchase_or_visit ('purchase' or 'visit') identifies which type of history
        to accumulate stats on"""
        self.field_name = field_name
        self.purchase_or_visit = purchase_or_visit

        self.known_field_values = set()
        
        self.history_getter = operator.attrgetter(purchase_or_visit)
        self.field_getter = operator.attrgetter(field_name)
        self.coupon_getter = operator.attrgetter('VIEW_COUPON_ID_hash' if purchase_or_visit == 'visit' else 'COUPON_ID_hash')
        
        self.item_count_by_field_value = {}
        self.item_count = 0
        
        self.count_by_field_value_and_earlier_field_value = {}

        self.column_count = {}
        self.row_count = {}
        self.total_count = 0

    def __repr__ (self):
        return "NBAccumulator({0},{1})".format(self.purchase_or_visit, self.field_name)

#    @functools.lru_cache(maxsize=1024)
    def __history_set(self, history_list, date):
        history_set = set()
        for history_item in history_list:
            if history_item.I_DATE < date:
                history_set.add(self.field_getter(self.coupon_getter(history_item)))
        return history_set

    def add (self, purchase, user_history):
        date = purchase.I_DATE
        field_value = self.field_getter(purchase.COUPON_ID_hash)

        self.known_field_values.add(field_value)
        self.item_count_by_field_value[field_value] = self.item_count_by_field_value.get(field_value, 0) + 1
        self.item_count += 1

        history_list = self.history_getter(user_history)
        history_set = self.__history_set(history_list, date)
                
        for historic_field_value in history_set:
            self.known_field_values.add(historic_field_value)
            
            t = (field_value, historic_field_value)
            self.count_by_field_value_and_earlier_field_value[t] = self.count_by_field_value_and_earlier_field_value.get(t, 0) + 1
            
            self.column_count[historic_field_value] = self.column_count.get(historic_field_value, 0) + 1
            self.row_count[field_value] = self.row_count.get(field_value, 0) + 1
            self.total_count += 1
            
    def dump (self, limit=None):
        print ('{0}: {1}'.format(self.purchase_or_visit, self.field_name))
        
        if limit:
            print ('Limited to {0} category values'.format(limit))
            
        field_values = list(self.known_field_values)[0:limit]
        print('value\n\t\tall {0}'.format(' '.join(map('{0:>10}'.format, field_values))))
        
        for fv in field_values:
            print ('{0:>10}:\n\t{1:>10} {2}'.format(
                fv,
                self.row_count.get(fv, 0),
                ' '.join(map('{0:>10}'.format, [self.count_by_field_value_and_earlier_field_value.get((fv,pv), 0) for pv in field_values]))
            ))

        print ('column sums:\n\t{0:>10} {1}'.format(
            self.total_count,
            ' '.join(map('{0:>10}'.format, (self.column_count[column] for column in field_values[0:limit])))
        ))

        print ('item counts:\n\t{0:>10} {1}'.format(
            self.item_count,
            ' '.join(map('{0:>10}'.format, (self.item_count_by_field_value.get(fv,0) for fv in field_values[0:limit])))
        ))
        
    def score (self, coupon, user_history, date):
        candidate_field_value = self.field_getter(coupon)

        history_list = self.history_getter(user_history)
        history_set = self.__history_set(history_list, date)

        N = len(self.known_field_values)

        candidate_item_count = self.item_count_by_field_value.get(candidate_field_value, 0)
        not_candidate_item_count = self.item_count - candidate_item_count

#        print('*candidate_item_count', candidate_item_count)
#        print('*not_candidate_item_count', not_candidate_item_count)

        p_class = float(1.0 + candidate_item_count) / (N + self.item_count)
        p_not_class = 1.0 - p_class

#        print('\tp_class={0}/{1} ({2}), p_not_class={3}'.format(
#            candidate_item_count, self.item_count, p_class, p_not_class)
#        )
        
        
        log_likelihood = math.log(p_class / p_not_class)
        
        candidate_row_count = self.row_count.get(candidate_field_value, 0)
        
        for fv in self.known_field_values:
            t = (candidate_field_value, fv)
            field_value_count = self.count_by_field_value_and_earlier_field_value.get(t, 0)

            p_x_candidate = float(1.0 + field_value_count) / (N + candidate_row_count)
            p_x_not_candidate = float(1.0 + self.column_count.get(fv, 0) - field_value_count) / (N + self.total_count - candidate_row_count)

#            print('\t\tfv={0}, p_x_candidate={1}/{2} ({3}), p_x_not_candidate = {4}/{5} ({6})'.format(
#                fv, field_value_count, candidate_row_count, p_x_candidate, self.column_count.get(fv,0) - field_value_count, self.total_count-candidate_row_count, p_x_not_candidate)
#              )
            
            if fv in history_set:
                log_likelihood += math.log (p_x_candidate / p_x_not_candidate)
            else:
                log_likelihood += math.log ((1 - p_x_candidate) / (1 - p_x_not_candidate))

        return log_likelihood
    
class MultinomialNBAccumulator:
    """Naive Base accumulator"""
    def __init__ (self, purchase_or_visit, field_name):
        """purchase_or_visit ('purchase' or 'visit') identifies which type of history
        to accumulate stats on"""
        self.field_name = field_name
        self.purchase_or_visit = purchase_or_visit

        self.known_field_values = set()

        self.history_getter = operator.attrgetter(purchase_or_visit)
        self.field_getter = operator.attrgetter(field_name)
        self.coupon_getter = operator.attrgetter('VIEW_COUPON_ID_hash' if purchase_or_visit == 'visit' else 'COUPON_ID_hash')
        
        self.item_count_by_field_value = {}
        self.item_count = 0
        
        self.count_by_field_value_and_earlier_field_value = {}

        self.column_count = {}
        self.row_count = {}
        self.total_count = 0

    def __repr__ (self):
        return "NBAccumulator({0},{1})".format(self.purchase_or_visit, self.field_name)

#    @functools.lru_cache(maxsize=1024)
    def __history_set(self, history_list, date):
        history_set = {}
        for history_item in history_list:
            if history_item.I_DATE < date:
                fv = self.field_getter(self.coupon_getter(history_item))
                history_set[fv] = history_set.get(fv, 0) + 1
        return history_set

    def add (self, purchase, user_history):
        date = purchase.I_DATE
        field_value = self.field_getter(purchase.COUPON_ID_hash)

        self.known_field_values.add(field_value)
        self.item_count_by_field_value[field_value] = self.item_count_by_field_value.get(field_value, 0) + 1
        self.item_count += 1

        history_list = self.history_getter(user_history)
        history_set = self.__history_set(history_list, date)
                
        for historic_field_value,count in history_set.items():
            self.known_field_values.add(historic_field_value)
            
            t = (field_value, historic_field_value)
            self.count_by_field_value_and_earlier_field_value[t] = self.count_by_field_value_and_earlier_field_value.get(t, 0) + count

            self.column_count[historic_field_value] = self.column_count.get(historic_field_value, 0) + count
            self.row_count[field_value] = self.row_count.get(field_value, 0) + count
            self.total_count += count

    def dump (self, limit=None):
        print ('{0}: {1}'.format(self.purchase_or_visit, self.field_name))
        
        if limit:
            print ('Limited to {0} category values'.format(limit))
            
        field_values = list(self.known_field_values)[0:limit]
        print('value\n\t\tall {0}'.format(' '.join(map('{0:>10}'.format, map(str, field_values)))))
        
        for fv in field_values:
            print ('{0:>10}:\n\t{1:>10} {2}'.format(
                str(fv),
                self.row_count.get(fv, 0),
                ' '.join(map('{0:>10}'.format, [self.count_by_field_value_and_earlier_field_value.get((fv,pv), 0) for pv in field_values]))
            ))

        print ('column sums:\n\t{0:>10} {1}'.format(
            self.total_count,
            ' '.join(map('{0:>10}'.format, map(str, (self.column_count[column] for column in field_values[0:limit]))))
        ))

        print ('item counts:\n\t{0:>10} {1}'.format(
            self.item_count,
            ' '.join(map('{0:>10}'.format, (self.item_count_by_field_value.get(fv,0) for fv in field_values[0:limit])))
        ))
        
    def score (self, coupon, user_history, date):
        candidate_field_value = self.field_getter(coupon)

        history_list = self.history_getter(user_history)
        history_set = self.__history_set(history_list, date)

        N = len(self.known_field_values)

        candidate_item_count = self.item_count_by_field_value.get(candidate_field_value, 0)
        not_candidate_item_count = self.item_count - candidate_item_count

#        print('candidate_item_count', candidate_item_count)
#        print('not_candidate_item_count', not_candidate_item_count)
        
        p_class = float(1.0 + candidate_item_count) / (N + self.item_count)
        p_not_class = 1.0 - p_class

#        print('\tp_class={0}/{1} ({2}), p_not_class={3}'.format(
#            candidate_item_count, self.item_count, p_class, p_not_class)
#        )

        log_likelihood = math.log(p_class / p_not_class)
        
        candidate_row_count = self.row_count.get(candidate_field_value, 0)
        for fv,count in history_set.items():
            t = (candidate_field_value, fv)
            field_value_count = self.count_by_field_value_and_earlier_field_value.get(t, 0)
            
            p_x_candidate = float(1.0 + field_value_count) / (N + candidate_row_count )
            p_x_not_candidate = float(1.0 + self.column_count.get(fv, 0) - field_value_count) / (N + self.total_count - candidate_row_count)

#            print('\t\tfv={0}, p_x_candidate={1}/{2} ({3}), p_x_not_candidate = {4}/{5} ({6})'.format(
#                fv, field_value_count, candidate_row_count, p_x_candidate, self.column_count.get(fv,0) - field_value_count, self.total_count-candidate_row_count, p_x_not_candidate)
#            )

            log_likelihood += count * math.log (p_x_candidate / p_x_not_candidate)
            
        return log_likelihood

if __name__ == "__main__":

    import collections
    Coupon = collections.namedtuple('Coupon', ['city'])
    Visit = collections.namedtuple('Visit', ['VIEW_COUPON_ID_hash', 'I_DATE'])
    Purchase = collections.namedtuple('Purchase', ['COUPON_ID_hash', 'I_DATE'])
    History = collections.namedtuple('History', ['visit'])

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

    accumulator = NBAccumulator('visit', 'city')
    mn_accumulator = MultinomialNBAccumulator('visit', 'city')
    
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

