from evolve import Evolve_CNN
from utils import *
import tensorflow as tf

def begin_evolve(m_prob, m_eta, x_prob, x_eta, pop_size, train_data, train_label, validate_data, validation_label, number_of_channel, epochs, batch_size, train_data_length, validate_data_length, total_generation_number, eta):
    cnn = Evolve_CNN(m_prob, m_eta, x_prob, x_eta, pop_size, train_data, train_label, validate_data, validation_label, number_of_channel, epochs, batch_size, train_data_length, validate_data_length, eta)
    cnn.initialize_popualtion()
    cnn.evaluate_fitness(0)
    for cur_gen_no in range(total_generation_number):
        print('The {}/{} generation'.format(cur_gen_no+1, total_generation_number))
        cnn.recombinate(cur_gen_no+1)
        cnn.environmental_selection(cur_gen_no+1)

def restart_evolve(m_prob, m_eta, x_prob, x_eta, pop_size, train_data, train_label, validate_data, validation_label, number_of_channel, epochs, batch_size, train_data_length, validate_data_length, total_gene_number, eta):
    gen_no, pops, _= load_population()
    cnn = Evolve_CNN(m_prob, m_eta, x_prob, x_eta, pop_size, train_data, train_label, validate_data, validation_label, number_of_channel, epochs, batch_size, train_data_length, validate_data_length, eta)
    cnn.pops = pops
    if gen_no < 0: # go to evaluate
        print('first to evaluate...')
        cnn.evaluate_fitness(1)
    else:
        for cure_gen_no in range(gen_no+1, total_gene_number+1):
            print('Continue to evolve from the {}/{} generation...'.format(cure_gen_no, total_gene_number))
            cnn.recombinate(cure_gen_no)
            cnn.environmental_selection(cure_gen_no)



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    tf.logging.set_verbosity(tf.logging.ERROR)
    if not tf.gfile.Exists('./save_data'):
        tf.gfile.MkDir('./save_data')

    #train_data, validation_data, test_data = get_mnist_data()
    batch_size = 100
    tf.reset_default_graph()
    number_of_channel = 1
    train_data_length = 10000
    validate_data_length = 2000
    total_generation_number = 50# total generation number
    pop_size = 50
    epochs = 10
    eta = 1/20
    #CUDA1
    #begin_evolve(0.9, 0.05, 0.2, 0.05, pop_size, None, None, None, None, number_of_channel, epochs, batch_size, train_data_length, validate_data_length, total_generation_number, eta)
    restart_evolve(0.9, 0.05, 0.2, 0.05, pop_size, None, None, None, None, number_of_channel, epochs, batch_size, train_data_length, validate_data_length, total_generation_number, eta)




