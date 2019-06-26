'''
All three kinds of layers can be initialized with their default parameters
'''

class ConvLayer:
    def __init__(self, filter_size=[2,2], feature_map_size=8, weight_matrix=[0.0,1.0]):
        self.filter_width = filter_size[0]
        self.filter_height = filter_size[1]
        self.feature_map_size = feature_map_size
        self.weight_matrix_mean = weight_matrix[0]
        self.weight_matrix_std = weight_matrix[1]
        self.type = 1
    def __str__(self):
        return "Conv Layer: filter:[{0},{1}], feature map number:{2}, weight:[{3},{4}]".format(self.filter_width, self.filter_height, self.feature_map_size, self.weight_matrix_mean, self.weight_matrix_std)

class PoolLayer:
    def __init__(self, kernel_size=[2,2], pool_type=0.1):
        self.kernel_width = kernel_size[0]
        self.kernel_height = kernel_size[1]
        self.kernel_type = pool_type # values below 0.5 means max other wise mean
        self.type = 2

    def __str__(self):
        return "Pool Layer: kernel:[{0},{1}], type:{2}".format(self.kernel_width, self.kernel_height, "max" if self.kernel_type<0.5 else "mean")

class FullLayer:
    def __init__(self, hidden_neuron_num=10, weight_matrix=[0.0,1.0]):
        self.hidden_neuron_num = hidden_neuron_num
        self.weight_matrix_mean = weight_matrix[0]
        self.weight_matrix_std = weight_matrix[1]
        self.type = 3

    def __str__(self):
        return "Full Layer: hidden neurons:{}, weight:[{},{}]".format(self.hidden_neuron_num, self.weight_matrix_mean, self.weight_matrix_std)