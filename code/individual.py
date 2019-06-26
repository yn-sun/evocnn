from layers import ConvLayer, PoolLayer, FullLayer
import numpy as np
import random
from utils import *

class Individual:

    def __init__(self, x_prob=0.9, x_eta=0.05, m_prob=0.2, m_eta=0.05):
        self.indi = []
        self.x_prob = x_prob
        self.x_eta = x_eta
        self.m_prob = m_prob
        self.m_eta = m_eta
        self.mean = 0
        self.std = 0
        self.complxity = 0

        #####################
        self.featur_map_size_range = [3, 50]
        self.filter_size_range = [2, 20]
        self.pool_kernel_size_range = [1, 2]
        self.hidden_neurons_range = [1000, 2000]
        self.mean_range = [-1,1]
        self.std_range = [0,1]

    def clear_state_info(self):
        self.complxity = 0
        self.mean = 0
        self.std = 0

    '''
    initialize a simle CNN network including one convolutional layer, one pooling layer, and one full connection layer
    '''
    def initialize(self):
        self.indi = self.init_one_individual()

    def init_one_individual(self):
        init_num_conv = np.random.randint(1, 3)
        init_num_pool = np.random.randint(1, 3)
        init_num_full = np.random.randint(1, 3)
        _list = []
        for _ in range(init_num_conv):
            _list.append(self.add_a_random_conv_layer())
        for _ in range(init_num_pool):
            _list.append(self.add_a_random_pool_layer())
        for _ in range(init_num_full-1):
            _list.append(self.add_a_random_full_layer())
        _list.append(self.add_a_common_full_layer())
        return _list

    def get_layer_at(self,i):
        return self.indi[i]

    def get_layer_size(self):
        return len(self.indi)
    def init_mean(self):
        return np.random.random()*(self.mean_range[1] - self.mean_range[0]) + self.mean_range[0]
    def init_std(self):
        return np.random.random()*(self.std_range[1] - self.std_range[0]) + self.std_range[0]

    def init_feature_size(self):
        return np.random.randint(self.filter_size_range[0], self.filter_size_range[1])
    def init_feature_map_size(self):
        return np.random.randint(self.featur_map_size_range[0], self.featur_map_size_range[1])
    def init_kernel_size(self):
        kernel_size_num = len(self.pool_kernel_size_range)
        n = np.random.randint(kernel_size_num)
        return np.power(2, self.pool_kernel_size_range[n])
    def init_hidden_neuron_size(self):
        return np.random.randint(self.hidden_neurons_range[0], self.hidden_neurons_range[1])

    def mutation(self):
        if flip(self.m_prob):
            #for the units
            unit_list = []
            for i in range(self.get_layer_size()-1):
                cur_unit = self.get_layer_at(i)
                if flip(0.5):
                    #mutation
                    p_op = self.mutation_ope(rand())
                    max_length = 6
                    current_length = (len(unit_list) + self.get_layer_size() - i-1)
                    if p_op == 0: #add a new

                        if current_length < max_length: # when length exceeds this length, only mutation no add new unit
                            unit_list.append(self.generate_a_new_layer(cur_unit.type, self.get_layer_size()))
                            unit_list.append(cur_unit)
                        else:
                            updated_unit = self.mutation_a_unit(cur_unit, self.m_eta)
                            unit_list.append(updated_unit)
                    if p_op == 1: #modify the lement
                        updated_unit = self.mutation_a_unit(cur_unit, self.m_eta)
                        unit_list.append(updated_unit)

                else:
                    unit_list.append(cur_unit)
            #avoid all units have been removed, add a full layer
            if len(unit_list) == 0:
                unit_list.append(self.add_a_random_conv_layer())
                unit_list.append(self.add_a_random_pool_layer())
            unit_list.append(self.get_layer_at(-1))
            # judge the first unit and the second unit
            if unit_list[0].type != 1:
                unit_list.insert(0, self.add_a_random_conv_layer())
            self.indi = unit_list


    def mutation_a_unit(self, unit, eta):
        if unit.type ==1:
            #mutate a conv layer
            return self.mutate_conv_unit(unit, eta)
        elif unit.type ==2:
            #mutate a pool layer
            return self.mutate_pool_unit(unit, eta)
        else:
            #mutate a full layer
            return self.mutate_full_layer(unit, eta)

    def mutate_conv_unit(self, unit, eta):
        # feature map size, feature map number, mean std
        fms = unit.filter_width
        fmn = unit.feature_map_size
        mean = unit.weight_matrix_mean
        std = unit.weight_matrix_std

        new_fms = int(self.pm(self.filter_size_range[0], self.filter_size_range[-1], fms, eta))
        new_fmn = int(self.pm(self.featur_map_size_range[0], self.featur_map_size_range[1], fmn, eta))
        new_mean = self.pm(self.mean_range[0], self.mean_range[1], mean, eta)
        new_std = self.pm(self.std_range[0], self.std_range[1], std, eta)
        conv_layer = ConvLayer(filter_size=[new_fms, new_fms], feature_map_size=new_fmn, weight_matrix=[new_mean, new_std])
        return conv_layer

    def mutate_pool_unit(self, unit, eta):
        #kernel size, pool_type
        ksize = np.log2(unit.kernel_width)
        pool_type = unit.kernel_type

        new_ksize = self.pm(self.pool_kernel_size_range[0], self.pool_kernel_size_range[-1], ksize, eta)
        new_ksize = int(np.power(2, new_ksize))
        new_pool_type = self.pm(0, 1, pool_type, eta)
        pool_layer = PoolLayer(kernel_size=[new_ksize,new_ksize], pool_type=new_pool_type)
        return pool_layer

    def mutate_full_layer(self, unit, eta):
        #num of hidden neurons, mean ,std
        n_hidden = unit.hidden_neuron_num
        mean = unit.weight_matrix_mean
        std = unit.weight_matrix_std

        new_n_hidden = int(self.pm(self.hidden_neurons_range [0], self.hidden_neurons_range[-1], n_hidden, eta))
        new_mean = self.pm(self.mean_range[0], self.mean_range[1], mean, eta)
        new_std = self.pm(self.std_range[0], self.std_range[1], std, eta)
        full_layer = FullLayer(hidden_neuron_num=new_n_hidden, weight_matrix=[new_mean, new_std])
        return full_layer

#0 add, 1 modify  2delete

    def mutation_ope(self, r):
        if r < 0.33:
            return 1
        elif r >0.66:
            return 2
        else:
            return 0

    def add_a_common_full_layer(self):
        mean = self.init_mean()
        std = self.init_std()
        full_layer = FullLayer(hidden_neuron_num=2, weight_matrix=[mean, std])
        return full_layer
    def add_a_random_full_layer(self):
        mean = self.init_mean()
        std = self.init_std()
        hidden_neuron_num = self.init_hidden_neuron_size()
        full_layer = FullLayer(hidden_neuron_num=hidden_neuron_num, weight_matrix=[mean, std])
        return full_layer
    def add_a_random_conv_layer(self):
        s1 = self.init_feature_size()
        filter_size=s1,s1
        feature_map_size =  self.init_feature_map_size()
        mean = self.init_mean()
        std = self.init_std()
        conv_layer = ConvLayer(filter_size=filter_size, feature_map_size=feature_map_size, weight_matrix=[mean, std])
        return conv_layer
    def add_a_random_pool_layer(self):
        s1 = self.init_kernel_size()
        kernel_size=s1, s1
        pool_type=np.random.random(size=1)
        pool_layer = PoolLayer(kernel_size=kernel_size, pool_type=pool_type[0])
        return pool_layer


    def generate_a_new_layer(self, current_unit_type, unit_length):
        if current_unit_type == 3:
            #judge if current length = 1, add conv or pool
            if unit_length == 1:
                if random.random() < 0.5:
                    return self.add_a_random_conv_layer()
                else:
                    return self.add_a_random_pool_layer()
            else:
                return self.add_a_random_full_layer()
        else:
            r = random.random()
            if r <0.5:
                return self.add_a_random_conv_layer()
            else:
                return self.add_a_random_pool_layer()



    def pm(self, xl, xu, x, eta):
        delta_1 = (x - xl) / (xu - xl)
        delta_2 = (xu - x) / (xu - xl)
        rand = np.random.random()
        mut_pow = 1.0 / (eta + 1.)
        if rand < 0.5:
            xy = 1.0 - delta_1
            val = 2.0 * rand + (1.0 - 2.0 * rand) * xy**(eta + 1)
            delta_q = val**mut_pow - 1.0
        else:
            xy = 1.0 - delta_2
            val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * xy**(eta + 1)
            delta_q = 1.0 - val**mut_pow
        x = x + delta_q * (xu - xl)
        x = min(max(x, xl), xu)
        return x






    def __str__(self):


        str_ = []
        str_.append('Length:{}, Num:{}'.format(self.get_layer_size(), self.complxity))
        str_.append('Mean:{:.2f}'.format(self.mean))
        str_.append('Std:{:.2f}'.format(self.std))

        for i in range(self.get_layer_size()):
            unit = self.get_layer_at(i)
            if unit.type == 1:
                str_.append("conv[{},{},{},{:.2f},{:.2f}]".format(unit.filter_width, unit.filter_height, unit.feature_map_size, unit.weight_matrix_mean, unit.weight_matrix_std))
            elif unit.type ==2:
                str_.append("pool[{},{},{:.2f}]".format(unit.kernel_width, unit.kernel_height, unit.kernel_type))
            elif unit.type ==3:
                str_.append("full[{},{},{}]".format(unit.hidden_neuron_num, unit.weight_matrix_mean, unit.weight_matrix_std))
            else:
                raise Exception("Incorrect unit flag")
        return ', '.join(str_)


if __name__ =='__main__':
    ind = Individual()
    print(ind.randint(1,10))


