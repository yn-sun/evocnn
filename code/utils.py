import numpy as np
import os
import pickle
from time import gmtime, strftime
from population import *
from individual import *

def get_data_path():
    return os.getcwd() + '/pops.dat'

def save_populations(gen_no, pops):
    data = {'gen_no':gen_no, 'pops':pops, 'create_time':strftime("%Y-%m-%d %H:%M:%S", gmtime())}
    path = get_data_path()
    with open(path, 'wb') as file_handler:
        pickle.dump(data, file_handler)

def load_population():
    path = get_data_path()
    with open(path, 'rb') as file_handler:
        data = pickle.load(file_handler)
    return data['gen_no'], data['pops'],data['create_time']

def save_offspring(gen_no, pops):
    data = {'gen_no':gen_no, 'pops':pops, 'create_time':strftime("%Y-%m-%d %H:%M:%S", gmtime())}
    path = os.getcwd() + '/offsprings_data/gen_{}.dat'.format(gen_no)
    with open(path, 'wb') as file_handler:
        pickle.dump(data, file_handler)

def load_save_log_data():
    file_name = '/am/lido/home/yanan/eclipse-workspace/Ver3/pops.dat'
    with open(file_name, 'br') as file_h:
        data = pickle.load(file_h)
        print(data)
        pops = data['pops'].pops
        for i in range(len(pops)):
            print(pops[i])

def save_append_individual(indi, file_path):
    with open(file_path, 'a') as myfile:
        myfile.write(indi)
        myfile.write("\n")
def randint(low, high):
    return np.random.random_integers(low, high-1)

def rand():
    return np.random.random()

def flip(f):
    if rand() <= f:
        return True
    else:
        return False

if __name__ =='__main__':
    load_save_log_data()





