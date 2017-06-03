import logging
import random

logger = logging.getLogger(__name__)

class Shaper:
    def __init__(self, random_state, size = 10, desired_density=lambda x: x, recompute_interval=100):
        self.random_state = random_state
        self.size = self.count = size
        self.bins = size * [1.0]
        self.ratios = size * [1.0]
        self.desired_density = desired_density
        self.max_count = 1
        self.interval = recompute_interval
        
    def add(self, p):
        qp = int(p * self.size)
        if qp == self.size:
            qp = self.size -1
        self.bins[qp] += 1
        self.count += 1
        if self.count % self.interval == 0:
            self._estimate()

    def dump(self):
        print('actual dist: {0}'.format([float(c)/self.count for c in self.bins]))
        print('selection frequency: {0}'.format(self.ratios))

    def _estimate(self):
        max_ratio = 0.0
        for i in range(self.size):
            midpoint = (i+0.5)/float(self.size)
            self.ratios[i] = self.desired_density(midpoint) * self.count / self.bins[i]
            if self.ratios[i] > max_ratio:
                max_ratio = self.ratios[i]

        for i in range(self.size):
            self.ratios[i] /= max_ratio
                
    def accept(self, p):
        self.add(p)
        """Should I accept p?"""
        qp = int(p * self.size)
        if qp == self.size:
            qp = self.size -1
        return self.ratios[qp] > self.random_state.random()
        
class QuotaShaper:
    def __init__(self, size = 10, desired_density=lambda x: x):
        self.size = size
        self.bins = size * [0]

        self.count = 0
        self.desired_density = desired_density

        self.density_cache = [ desired_density((0.5+i)/self.size) for i in range(self.size)]

        area = sum(self.density_cache)

        for i in range(self.size):
            self.density_cache[i] /= area
        
    def dump(self):
        print('actual dist: {0}'.format([float(c)/self.count for c in self.bins]))
        
    def accept(self, p):
        """Should I accept p?"""
        qp = int(p * self.size)
        if qp == self.size:
            qp = self.size -1

        if self.bins[qp] <= self.density_cache[qp] * self.count:
            self.bins[qp] += 1
            self.count += 1
            return True
        else:
            return False
    
if __name__ == "__main__":

    shaper = Shaper(random.Random())
#    shaper = QuotaShaper()

    def my_random():
        return min(random.random(), random.random(), random.random())

    size = 10
    bins = size * [0]
    count = 0
    while count < 10000:
        p = my_random()
        if shaper.accept(p):
            bins[int(size*p)] += 1
            count += 1
            
    print([c/float(count) for c in bins])
