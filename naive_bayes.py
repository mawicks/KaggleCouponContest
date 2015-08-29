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
        self.known_attribute_values = set()

        self.class_counts = {}
        self.count_by_class_and_attribute = {}
        self.class_count = 0

        self.column_count = {}
        self.row_count = {}
        self.total_count = 0

    def __repr__ (self):
        return "Accumulator({0}".format(self.strategy)

    def add (self, class_value, history_set):
        self.class_counts[class_value] = self.class_counts.get(class_value, 0) + 1
        self.class_count += 1

        for historic_attribute_value,count in history_set.items():
            self.known_attribute_values.add(historic_attribute_value)
            count = self.strategy.occurences(count)

            t = (class_value, historic_attribute_value)
            self.count_by_class_and_attribute[t] = self.count_by_class_and_attribute.get(t, 0) + count

            self.column_count[historic_attribute_value] = self.column_count.get(historic_attribute_value, 0) + count
            self.row_count[class_value] = self.row_count.get(class_value, 0) + count
            self.total_count += count

    def dump (self, limit=None):
        if limit:
            print ('Limited to {0} category values'.format(limit))

        field_values = list(self.known_attribute_values)[0:limit]
        print('value\n\t\tall {0}'.format(' '.join(map('{0:>10}'.format, field_values))))

        for fv in field_values:
            print ('{0:>10}:\n\t{1:>10} {2}'.format(
                fv,
                self.row_count.get(fv, 0),
                ' '.join(map('{0:>10}'.format, [self.count_by_class_and_attribute.get((fv,pv), 0) for pv in field_values]))
            ))

        print ('column sums:\n\t{0:>10} {1}'.format(
            self.total_count,
            ' '.join(map('{0:>10}'.format, (self.column_count.get(column, 0) for column in field_values[0:limit])))
        ))

        print ('item counts:\n\t{0:>10} {1}'.format(
            self.class_count,
            ' '.join(map('{0:>10}'.format, (self.class_counts.get(fv,0) for fv in field_values[0:limit])))
        ))

    def score (self, candidate_class_value, history_set):
        N = len(self.known_attribute_values)

        candidate_class_count = self.class_counts.get(candidate_class_value, 0)
        not_candidate_class_count = self.class_count - candidate_class_count

        p_class = float(1.0 + candidate_class_count) / (N + self.class_count)
        p_not_class = 1.0 - p_class

        log_likelihood = math.log(p_class / p_not_class)

        candidate_row_count = self.row_count.get(candidate_class_value, 0)

        for fv in self.known_attribute_values:
            t = (candidate_class_value, fv)

            field_value_count = self.count_by_class_and_attribute.get(t, 0)

            p_x_candidate = float(1.0 + field_value_count) / (N + candidate_row_count)
            p_x_not_candidate = float(1.0 + self.column_count.get(fv, 0) - field_value_count) / (N + self.total_count - candidate_row_count)

            count = history_set.get(fv, 0)
            log_likelihood += self.strategy.log_likelihood(count, p_x_candidate, p_x_not_candidate)

        return log_likelihood
