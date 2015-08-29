import math

class MNStrategy:
    def __init___(self):
        pass

    def occurences(self, count):
        return count

    def log_likelihood(self, count, p_x_candidate, p_x_not_candidate):
        return count * math.log (p_x_candidate / p_x_not_candidate)

class BNStrategy:
    def __init___(self):
        pass

    def occurences(self, count):
        return 1 if count > 0 else 0

    def log_likelihood(self, count, p_x_candidate, p_x_not_candidate):
        if count > 0:
            return math.log (p_x_candidate / p_x_not_candidate)
        else:
            return math.log ((1.0-p_x_candidate) / (1.0 - p_x_not_candidate))

class Estimator:
    """Naive Base accumulator"""
    def __init__ (self, strategy):
        """strategy is MNStrategy or BNStrategy
           purchase_or_visit ('purchase' or 'visit') identifies which type of history
           to accumulate stats on"""

        self.strategy = strategy
        self.known_field_values = set()

        self.item_count_by_field_value = {}
        self.item_count = 0

        self.count_by_field_value_and_earlier_field_value = {}

        self.column_count = {}
        self.row_count = {}
        self.total_count = 0

    def __repr__ (self):
        return "Accumulator({0}".format(self.strategy)

    def add (self, field_value, history_set):
        self.item_count_by_field_value[field_value] = self.item_count_by_field_value.get(field_value, 0) + 1
        self.item_count += 1

        for historic_field_value,count in history_set.items():
            self.known_field_values.add(historic_field_value)
            count = self.strategy.occurences(count)

            t = (field_value, historic_field_value)
            self.count_by_field_value_and_earlier_field_value[t] = self.count_by_field_value_and_earlier_field_value.get(t, 0) + count

            self.column_count[historic_field_value] = self.column_count.get(historic_field_value, 0) + count
            self.row_count[field_value] = self.row_count.get(field_value, 0) + count
            self.total_count += count

    def dump (self, limit=None):
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
            ' '.join(map('{0:>10}'.format, (self.column_count.get(column, 0) for column in field_values[0:limit])))
        ))

        print ('item counts:\n\t{0:>10} {1}'.format(
            self.item_count,
            ' '.join(map('{0:>10}'.format, (self.item_count_by_field_value.get(fv,0) for fv in field_values[0:limit])))
        ))

    def score (self, candidate_field_value, history_set):
        N = len(self.known_field_values)

        candidate_item_count = self.item_count_by_field_value.get(candidate_field_value, 0)
        not_candidate_item_count = self.item_count - candidate_item_count

        p_class = float(1.0 + candidate_item_count) / (N + self.item_count)
        p_not_class = 1.0 - p_class

        log_likelihood = math.log(p_class / p_not_class)

        candidate_row_count = self.row_count.get(candidate_field_value, 0)

        for fv in self.known_field_values:
            t = (candidate_field_value, fv)

            field_value_count = self.count_by_field_value_and_earlier_field_value.get(t, 0)

            p_x_candidate = float(1.0 + field_value_count) / (N + candidate_row_count)
            p_x_not_candidate = float(1.0 + self.column_count.get(fv, 0) - field_value_count) / (N + self.total_count - candidate_row_count)

            count = history_set.get(fv, 0)
            log_likelihood += self.strategy.log_likelihood(count, p_x_candidate, p_x_not_candidate)

        return log_likelihood
