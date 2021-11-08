import random
import input_numbers
import expected_output
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from perceptron import NeuralNetwork


def add_noise(array):
    for i in range(len(array)):
        array[i] += random.random() + 0.2

    return array


def show_prediction_results(nn, input_value, expected_value):
    with_noise = np.array(add_noise(np.array(input_value, dtype=float).copy()))
    with_noise = np.where(with_noise > 1, 1, 0)

    result = nn.predict(with_noise)
    
    max_index = np.argmax(result)
    print(f"Percentage: {result[max_index] * 100} | Expected value: {expected_value} | Predicted letter: {expected_output.EXPECTED_DESCRIPTION[max_index]}")

    dataset_keys = list(input_numbers.dataset)
    scores_content = []
    for j in range(len(dataset_keys)):
        scores_content.append(tuple((
            np.reshape(input_numbers.dataset[dataset_keys[j]], newshape=input_numbers.dataset_entry_shape),
            result[j].item()
        )))

    return tuple((
            tuple((
                np.reshape(with_noise, newshape=input_numbers.dataset_entry_shape),
                expected_output.EXPECTED_DESCRIPTION[max_index]
            )),
            scores_content
        ))


def main():
    params = [0.07]
    _labels = [
        f"Target training error deviation [0.07]: "
    ]
    _p_types = [float]

    for i in range(len(params)):
        try:
            temp = _p_types[i](input(_labels[i]))
            params[i] = temp if temp > 0 else params[i]
        except ValueError:
            continue

    nn = NeuralNetwork(input_nodes=36, hidden_nodes=25, output_nodes=5)

    for i in range(int(10e4)):
        current_index = random.choice(list(input_numbers.dataset))
        current_err = nn.train(add_noise(np.array(input_numbers.dataset[current_index]).copy()),
                               expected_output.target_vectors[current_index])
        if current_err < params[0]:
            break
    # for i in range(10000):
    #     nn.train(input_numbers.L, expected_output.L)
    #     nn.train(input_numbers.U, expected_output.U)
    #     nn.train(input_numbers.T, expected_output.T)
    #     nn.train(input_numbers.O, expected_output.O)
    #     nn.train(input_numbers.K, expected_output.K)

    # show_prediction_results(nn, input_numbers.L, 'L')
    # show_prediction_results(nn, input_numbers.U, 'U')
    # show_prediction_results(nn, input_numbers.T, 'T')
    # show_prediction_results(nn, input_numbers.O, 'O')
    # show_prediction_results(nn, input_numbers.K, 'K')

    plot_content = []
    for key in list(input_numbers.dataset):
        plot_content.append(show_prediction_results(nn, input_numbers.dataset[key], expected_value=key))

    figures = []
    for item in plot_content:
        fig = plt.figure()
        gs = GridSpec(2, len(list(input_numbers.dataset)), figure=fig)
        ax_main = fig.add_subplot(gs[0, :])
        ax_main.imshow(item[0][0])
        ax_main.set_title(str(item[0][1]))
        ax_main.set_axis_off()
        for i in range(len(list(input_numbers.dataset))):
            ax_target = fig.add_subplot(gs[1, i])
            ax_target.imshow(item[1][i][0])
            ax_target.set_title(f"{item[1][i][1]:.3f}")
            ax_target.set_axis_off()
        figures.append(fig)
    plt.show()
    pass


if __name__ == "__main__":
    main()
