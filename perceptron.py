import numpy as np

class NeuralNetwork:
    def __init__(self, input_nodes=3, hidden_nodes=3, output_nodes=3, rate=0.3):
        """
        :param int input_nodes: количество узлов во входном слое
        :param int hidden_nodes: количество узлов в скрытом слое
        :param int output_nodes: количество узлов в выходном слое
        :param float rate: коэфициент обучения
        :param str load_from: загрузить данные из предварительно обученной модели
        """
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.rate = rate
        self.w_i_h = None  # весовые коэффициенты между входным и скрытым слоем
        self.w_h_o = None  # весовые коэффициенты между скрытым и выходным слоем
        self.__init_weights()

    def train(self, input_list, target_list):
        """Тренировка нейронной сети - уточнение весовых коэффициентов
        :param iterable input_list: входные данные
        :param iterable target_list: целевые значения
        """
        # Преобразуем входные данные в двумерный массив [1, 2, 3, 4] -> array([[1], [2], [3], [4]])
        inputs = np.array(input_list, ndmin=2).T
        targets = np.array(target_list, ndmin=2).T

        # Расчитаем входящие сигналы для скрытого слоя
        h_inputs = np.dot(self.w_i_h, inputs)

        # Расчитаем исходящие сигналы для скрытого слоя
        h_outputs = self.__activation_function(h_inputs)

        # Расчитаем входящие сигналы для выходного слоя
        o_inputs = np.dot(self.w_h_o, h_outputs)

        # Расчитаем исходящие сигналы для выходного слоя
        o_outputs = self.__activation_function(o_inputs)

        # Выходная ошибка сети = целевое значение - фактическое значение
        o_errors = targets - o_outputs

        # Ошибки скрытого слоя - это ошибки выходного слоя сети,
        # распределенные пропорционально весовым коэфициентам связей
        # и рекомбинированные на скрытых узлах
        h_errors = np.dot(self.w_h_o.T, o_errors)

        # Обновим весовые по следующей формуле:
        # alpha * e * sigmoid(x) * (1 - sigmoid(x)) * o, где
        # alpha - коэфициент обучения,
        # e - выходная ошибка,
        # sigmoid(x) * (1 - sigmoid(x)) - производная от функции активации (сигмойды в нашем случае),
        # o - выходной сигнал предыдущего слоя.

        # Обновим весовые коэфициенты между скрытым и выходным слоем сети
        self.w_h_o += self.rate * np.dot((o_errors * o_outputs * (1 - o_outputs)), h_outputs.T)

        # Обновим весовые коэфициенты между входным и скрытым слоем сети
        self.w_i_h += self.rate * np.dot((h_errors * h_outputs * (1 - h_outputs)), inputs.T)
        return np.linalg.norm(o_errors)

    def predict(self, input_list):
        """Опрос нейронной сети - получение значений сигналов выходных узлов
        :param iterable input_list: входные данные
        :return numpy.array: выходные данные
        """
        # Преобразуем входные данные в двумерный массив [1, 2, 3, 4] -> array([[1], [2], [3], [4]])
        inputs = np.array(input_list, ndmin=2).T

        # Расчитаем входящие сигналы для скрытого слоя
        h_inputs = np.dot(self.w_i_h, inputs)

        # Расчитаем исходящие сигналы для скрытого слоя
        h_outputs = self.__activation_function(h_inputs)

        # Расчитаем входящие сигналы для выходного слоя
        o_inputs = np.dot(self.w_h_o, h_outputs)

        # Расчитаем исходящие сигналы для выходного слоя
        o_outputs = self.__activation_function(o_inputs)

        return o_outputs

    def back_query(self, output_list):
        """Осуществляет обратный запрос к сети
        :param iterable output_list: обратные исходящие сигналы сети
        :return numpy.array: выходные данные
        """
        # Преобразуем входные данные в двумерный массив [1, 2, 3, 4] -> array([[1], [2], [3], [4]])
        o_outputs = np.array(output_list, ndmin=2).T

        # Преобразуем выходящие сигналы во входящие сигналы для выходного слоя
        o_inputs = self.__back_activation_function(o_outputs)

        # Расчитаем исходящие сигналы для скрытого слоя
        h_outputs = np.dot(self.w_h_o.T, o_inputs)

        # Нормализуем сигналы от 0.01 до 0.99, т.к. сигмойда не может давать знаения за пределами этих чисел
        h_outputs -= np.min(h_outputs)
        h_outputs /= np.max(h_outputs)
        h_outputs *= 0.98
        h_outputs += 0.01

        # Расчитаем входящие сигналы для скрытого слоя
        hidden_inputs = self.__back_activation_function(h_outputs)

        # Расчитаем исходящие сигналы для входного слоя
        inputs = np.dot(self.w_i_h.T, hidden_inputs)

        # Нормализуем сигналы от 0.01 до 0.99, т.к. сигмойда не может давать знаения за пределами этих чисел
        inputs -= np.min(inputs)
        inputs /= np.max(inputs)
        inputs *= 0.98
        inputs += 0.01

        return inputs

    def __init_weights(self):
        """Инициализация случайных весов используя "улучшенный" вариант инициализации весовых коэфициентов. 
           Весовые коэфициенты выбираются из нормального распределения центром в нуле и со стандартным отклонением, 
           величина которого обратно пропорциональна квадратному корню из количества входящих связей на узел.
        """
        self.w_i_h = np.random.normal(0.0, pow(self.hidden_nodes, -0.5), (self.hidden_nodes, self.input_nodes))
        self.w_h_o = np.random.normal(0.0, pow(self.output_nodes, -0.5), (self.output_nodes, self.hidden_nodes))

    @staticmethod
    def __activation_function(x):
        """Функция активации нейронной сети
        :param iterable x: матрица входящих сигналов сети
        :return numpy.array: матрица сглаженных комбинированных сигналов
        """
        return 1.0 / (1.0 + np.exp(-x))  # в качастве функции активации будет выступать сигмойда

    @staticmethod
    def __back_activation_function(y):
        """Функция обратной активации нейронной сети
           В нашем случае, для сигмойды обратной функцией является логит - y = 1/(1 + e**-x) <=> x = ln(y/(1-y)) 
        :param iterable y: матрица обратных исходящих сигналов сети
        :return numpy.array: матрица обратно сглаженных комбинированных сигналов
        """
        return np.log(y / (1.0 - y))