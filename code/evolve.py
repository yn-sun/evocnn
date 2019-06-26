from population import Population
from evaluate import Evaluate
import numpy
import tensorflow.examples.tutorials.mnist as input_data
import tensorflow as tf
import collections
from utils import *
import copy


class Evolve_CNN:
    def __init__(self, m_prob, m_eta, x_prob, x_eta, population_size, train_data, train_label, validate_data, validate_label, number_of_channel, epochs, batch_size, train_data_length, validate_data_length, eta):
        self.m_prob = m_prob
        self.m_eta = m_eta
        self.x_prob = x_prob
        self.x_eta = x_eta
        self.population_size = population_size
        self.train_data = train_data
        self.train_label = train_label
        self.validate_data = validate_data
        self.validate_label = validate_label
        self.epochs = epochs
        self.eta = eta
        self.number_of_channel = number_of_channel
        self.batch_size = batch_size
        self.train_data_length = train_data_length
        self.validate_data_length = validate_data_length

    def initialize_popualtion(self):
        print("initializing population with number {}...".format(self.population_size))
        self.pops = Population(self.population_size)
        # all the initialized population should be saved
        save_populations(gen_no=-1, pops=self.pops)
    def evaluate_fitness(self, gen_no):
        print("evaluate fintesss")
        evaluate = Evaluate(self.pops, self.train_data, self.train_label, self.validate_data, self.validate_label, self.number_of_channel, self.epochs, self.batch_size, self.train_data_length, self.validate_data_length)
        evaluate.parse_population(gen_no)
#         # all theinitialized population should be saved
        save_populations(gen_no=gen_no, pops=self.pops)
        print(self.pops)


    def recombinate(self, gen_no):
        print("mutation and crossover...")
        offspring_list = []
        for _ in range(int(self.pops.get_pop_size()/2)):
            p1 = self.tournament_selection()
            p2 = self.tournament_selection()
            # crossover
            offset1, offset2 = self.crossover(p1, p2)
            # mutation
            offset1.mutation()
            offset2.mutation()
            offspring_list.append(offset1)
            offspring_list.append(offset2)
        offspring_pops = Population(0)
        offspring_pops.set_populations(offspring_list)
        save_offspring(gen_no, offspring_pops)
        #evaluate these individuals
        evaluate = Evaluate(self.pops, self.train_data, self.train_label, self.validate_data, self.validate_label, self.number_of_channel, self.epochs, self.batch_size, self.train_data_length, self.validate_data_length)
        evaluate.parse_population(gen_no)
#         #save
        self.pops.pops.extend(offspring_pops.pops)
        save_populations(gen_no=gen_no, pops=self.pops)

    def environmental_selection(self, gen_no):
        assert(self.pops.get_pop_size() == 2*self.population_size)
        elitsam = 0.2
        e_count = int(np.floor(self.population_size*elitsam/2)*2)
        indi_list = self.pops.pops
        indi_list.sort(key=lambda x:x.mean, reverse=True)
        elistm_list = indi_list[0:e_count]

        left_list = indi_list[e_count:]
        np.random.shuffle(left_list)
        np.random.shuffle(left_list)

        for _ in range(self.population_size-e_count):
            i1 = randint(0, len(left_list))
            i2 = randint(0, len(left_list))
            winner = self.selection(left_list[i1], left_list[i2])
            elistm_list.append(winner)

        self.pops.set_populations(elistm_list)
        save_populations(gen_no=gen_no, pops=self.pops)
        np.random.shuffle(self.pops.pops)


    def crossover(self, p1, p2):
        p1 = copy.deepcopy(p1)
        p2 = copy.deepcopy(p2)
        p1.clear_state_info()
        p2.clear_state_info()
        #for different unit, we define two list, one to save their index and the other one save unit
        p1_conv_index_list = []
        p1_conv_layer_list = []
        p1_pool_index_list = []
        p1_pool_layer_list = []
        p1_full_index_list = []
        p1_full_layer_list = []

        p2_conv_index_list = []
        p2_conv_layer_list = []
        p2_pool_index_list = []
        p2_pool_layer_list = []
        p2_full_index_list = []
        p2_full_layer_list = []

        for i in range(p1.get_layer_size()):
            unit = p1.get_layer_at(i)
            if unit.type == 1:
                p1_conv_index_list.append(i)
                p1_conv_layer_list.append(unit)
            elif unit.type == 2:
                p1_pool_index_list.append(i)
                p1_pool_layer_list.append(unit)
            else:
                p1_full_index_list.append(i)
                p1_full_layer_list.append(unit)


        for i in range(p2.get_layer_size()):
            unit = p2.get_layer_at(i)
            if unit.type == 1:
                p2_conv_index_list.append(i)
                p2_conv_layer_list.append(unit)
            elif unit.type == 2:
                p2_pool_index_list.append(i)
                p2_pool_layer_list.append(unit)
            else:
                p2_full_index_list.append(i)
                p2_full_layer_list.append(unit)

        #begin crossover on conn layer
        l = min(len(p1_conv_layer_list), len(p2_conv_layer_list))
        for i in range(l):
            unit_p1 = p1_conv_layer_list[i]
            unit_p2 = p2_conv_layer_list[i]
            if flip(self.x_prob):
                #filter size
                this_range = p1.filter_size_range
                w1 = unit_p1.filter_width
                w2 = unit_p2.filter_width
                n_w1, n_w2 = self.sbx(w1, w2, this_range[0], this_range[-1], self.x_eta)
                unit_p1.filter_width = int(n_w1)
                unit_p1.filter_height = int(n_w1)
                unit_p2.filter_width = int(n_w2)
                unit_p2.filter_height = int(n_w2)
                #feature map size
                this_range = p1.featur_map_size_range
                s1 = unit_p1.feature_map_size
                s2 = unit_p2.feature_map_size
                n_s1, n_s2 = self.sbx(s1, s2, this_range[0], this_range[-1], self.x_eta)
                unit_p1.feature_map_size = int(n_s1)
                unit_p2.feature_map_size = int(n_s2)
                #mean
                this_range = p1.mean_range
                m1 = unit_p1.weight_matrix_mean
                m2 = unit_p2.weight_matrix_mean
                n_m1, n_m2 = self.sbx(m1, m2, this_range[0], this_range[-1], self.x_eta)
                unit_p1.weight_matrix_mean = n_m1
                unit_p2.weight_matrix_mean = n_m2
                #std
                this_range = p1.std_range
                std1 = unit_p1.weight_matrix_std
                std2 = unit_p2.weight_matrix_std
                n_std1, n_std2 = self.sbx(std1, std2, this_range[0], this_range[-1], self.x_eta)
                unit_p1.weight_matrix_std = n_std1
                unit_p2.weight_matrix_std = n_std2

            p1_conv_layer_list[i] = unit_p1
            p2_conv_layer_list[i] = unit_p2

        l = min(len(p1_pool_layer_list), len(p2_pool_layer_list))
        for i in range(l):
            unit_p1 = p1_pool_layer_list[i]
            unit_p2 = p2_pool_layer_list[i]
            if flip(self.x_prob):
                # kernel size
                this_range = p1.pool_kernel_size_range
                k1 = np.log2(unit_p1.kernel_width)
                k2 = np.log2(unit_p2.kernel_width)
                n_k1, n_k2 = self.sbx(k1, k2, this_range[0], this_range[-1], self.x_eta)
                n_k1 = int(np.power(2, n_k1))
                n_k2 = int(np.power(2, n_k2))
                unit_p1.kernel_width = n_k1
                unit_p1.kernel_height = n_k1
                unit_p2.kernel_width = n_k2
                unit_p2.kernel_height = n_k2
                #pool type
                t1 = unit_p1.kernel_type
                t2 = unit_p2.kernel_type
                n_t1, n_t2 = self.sbx(t1, t2, 0, 1, self.x_eta)
                unit_p1.kernel_type = n_t1
                unit_p2.kernel_type = n_t2

            p1_pool_layer_list[i] = unit_p1
            p2_pool_layer_list[i] = unit_p2

        l = min(len(p1_full_layer_list), len(p2_full_layer_list))
        for i in range(l-1):
            unit_p1 = p1_full_layer_list[i]
            unit_p2 = p2_full_layer_list[i]
            if flip(self.x_prob):
                this_range = p1.hidden_neurons_range
                n1 = unit_p1.hidden_neuron_num
                n2 = unit_p2.hidden_neuron_num
                n_n1, n_n2 = self.sbx(n1, n2, this_range[0], this_range[1], self.x_eta)
                unit_p1.hidden_neuron_num = int(n_n1)
                unit_p2.hidden_neuron_num = int(n_n2)
                # std amnd mean
                this_range = p1.mean_range
                m1 = unit_p1.weight_matrix_mean
                m2 = unit_p2.weight_matrix_mean
                n_m1, n_m2 = self.sbx(m1, m2, this_range[0], this_range[-1], self.x_eta)
                unit_p1.weight_matrix_mean = n_m1
                unit_p2.weight_matrix_mean = n_m2

                this_range = p1.std_range
                std1 = unit_p1.weight_matrix_std
                std2 = unit_p2.weight_matrix_std
                n_std1, n_std2 = self.sbx(std1, std2, this_range[0], this_range[-1], self.x_eta)
                unit_p1.weight_matrix_std = n_std1
                unit_p2.weight_matrix_std = n_std2

            p1_full_layer_list[i] = unit_p1
            p2_full_layer_list[i] = unit_p2

        # for the last full layer, only mean and std
        unit_p1 = p1_full_layer_list[-1]
        unit_p2 = p2_full_layer_list[-1]
        if flip(self.x_prob):
        # std amnd mean
            this_range = p1.mean_range
            m1 = unit_p1.weight_matrix_mean
            m2 = unit_p2.weight_matrix_mean
            n_m1, n_m2 = self.sbx(m1, m2, this_range[0], this_range[-1], self.x_eta)
            unit_p1.weight_matrix_mean = n_m1
            unit_p2.weight_matrix_mean = n_m2

            this_range = p1.std_range
            std1 = unit_p1.weight_matrix_std
            std2 = unit_p2.weight_matrix_std
            n_std1, n_std2 = self.sbx(std1, std2, this_range[0], this_range[-1], self.x_eta)
            unit_p1.weight_matrix_std = n_std1
            unit_p2.weight_matrix_std = n_std2
        p1_full_layer_list[-1] = unit_p1
        p2_full_layer_list[-1] = unit_p2

        p1_units = p1.indi
        # assign these crossovered values to the p1 and p2
        for i in range(len(p1_conv_index_list)):
            p1_units[p1_conv_index_list[i]] = p1_conv_layer_list[i]
        for i in range(len(p1_pool_index_list)):
            p1_units[p1_pool_index_list[i]] = p1_pool_layer_list[i]
        for i in range(len(p1_full_index_list)):
            p1_units[p1_full_index_list[i]] = p1_full_layer_list[i]
        p1.indi = p1_units

        p2_units = p2.indi
        for i in range(len(p2_conv_index_list)):
            p2_units[p2_conv_index_list[i]] = p2_conv_layer_list[i]
        for i in range(len(p2_pool_index_list)):
            p2_units[p2_pool_index_list[i]] = p2_pool_layer_list[i]
        for i in range(len(p2_full_index_list)):
            p2_units[p2_full_index_list[i]] = p2_full_layer_list[i]
        p2.indi = p2_units

        return p1, p2



    def sbx_test(self, v1, v2, xl, xu, eta):
        return 0.1, 0.5

    def sbx(self, v1, v2, xl, xu, eta):
        if flip(0.5):
            if abs(v1-v2)>1e-14:
                x1 = min(v1, v2)
                x2 = max(v1, v2)
                r = rand()
                beta = 1.0 + (2.0 * (x1 - xl) / (x2 - x1))
                alpha = 2.0 - beta**-(eta + 1)
                if r <= 1.0 / alpha:
                    beta_q = (r * alpha)**(1.0 / (eta + 1))
                else:
                    beta_q = (1.0 / (2.0 - r * alpha))**(1.0 / (eta + 1))
                c1 = 0.5 * (x1 + x2 - beta_q * (x2 - x1))
                beta = 1.0 + (2.0 * (xu - x2) / (x2 - x1))
                alpha = 2.0 - beta**-(eta + 1)
                if r <= 1.0 / alpha:
                    beta_q = (r * alpha)**(1.0 / (eta + 1))
                else:
                    beta_q = (1.0 / (2.0 - r * alpha))**(1.0 / (eta + 1))
                c2 = 0.5 * (x1 + x2 + beta_q * (x2 - x1))
                c1 = min(max(c1, xl), xu)
                c2 = min(max(c2, xl), xu)
                if flip(0.5):
                    return c2, c1
                else:
                    return c1, c2
            else:
                return v1, v2
        else:
            return v1, v2


    def tournament_selection(self):
        ind1_id = randint(0, self.pops.get_pop_size())
        ind2_id = randint(0, self.pops.get_pop_size())
        ind1 = self.pops.get_individual_at(ind1_id)
        ind2 = self.pops.get_individual_at(ind2_id)
        winner = self.selection(ind1, ind2)
        return winner

    def selection(self, ind1, ind2):
        mean_threshold = 0.05
        complexity_threhold = 100
        if ind1.mean > ind2.mean:
            if ind1.mean - ind2.mean > mean_threshold:
                return ind1
            else:
                if ind2.complxity < (ind1.complxity-complexity_threhold):
                    return ind2
                else:
                    return ind1
        else:
            if ind2.mean - ind1.mean > mean_threshold:
                return ind2
            else:
                if ind1.complxity < (ind2.complxity-complexity_threhold):
                    return ind1
                else:
                    return ind2



if __name__ == '__main__':
    ev = Evolve_CNN(m_prob=0.2, m_eta=0.05, x_prob=0.9, x_eta=0.05, population_size=10, train_data=None, validate_data=None, epochs=1, eta=1)
    ev.initialize_popualtion()
    print(ev.pops)
    print('='*100)
    ind1 = ev.tournament_selection()
    ind2 = ev.tournament_selection()
    print('p1->', ind1)
    print('p2->', ind2)
    print('='*100)
    new_p1, new_p2 = ev.crossover(ind1, ind2)
    print('np1->', new_p1)
    print('np2->', new_p2)
    print('='*100)
    new_p1.mutation()
    new_p2.mutation()
    print('nnp1->', new_p1)
    print('nnp2->', new_p2)



