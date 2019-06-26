from individual import Individual
from utils import *

class Population:

    def __init__(self, num_pops):
        self.num_pops = num_pops
        self.pops = []
        for i in range(num_pops):
            indi = Individual()
            indi.initialize()
            self.pops.append(indi)

    def get_individual_at(self, i):
        return self.pops[i]

    def get_pop_size(self):
        return len(self.pops)

    def set_populations(self, new_pops):
        self.pops = new_pops





    def __str__(self):
        _str = []
        for i in range(self.get_pop_size()):
            _str.append(str(self.get_individual_at(i)))
        return '\n'.join(_str)