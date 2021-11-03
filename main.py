import random
import input_numbers
import expected_output
import numpy as np
from perceptron import NeuralNetwork

def add_noise(array):
    for i in range(len(array)):
        array[i]+=random.random() + 0.2

    return array

def show_prediction_results(nn, input_value, expected_value):
    with_noise = np.array(add_noise(input_value))
    with_noise =  np.where(with_noise > 1, 1, 0)

    result = nn.predict(with_noise)
    
    max_index = np.argmax(result)
    print(f"Percentage: {result[max_index] * 100} | Expected value: {expected_value} | Predicted letter: {expected_output.EXPECTED_DESCRIPTION[max_index]}")


def main():
    nn = NeuralNetwork(input_nodes=36, hidden_nodes=25, output_nodes=5)

    for i in range(10000):
        nn.train(input_numbers.L, expected_output.L)
        nn.train(input_numbers.U, expected_output.U)
        nn.train(input_numbers.T, expected_output.T)
        nn.train(input_numbers.O, expected_output.O)
        nn.train(input_numbers.K, expected_output.K)


    show_prediction_results(nn, input_numbers.L, 'L')
    show_prediction_results(nn, input_numbers.U, 'U')
    show_prediction_results(nn, input_numbers.T, 'T')
    show_prediction_results(nn, input_numbers.O, 'O')
    show_prediction_results(nn, input_numbers.K, 'K')


if __name__ == "__main__":
    main()
