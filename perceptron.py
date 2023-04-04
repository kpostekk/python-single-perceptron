import pandas as pd
import numpy as np


def load_dataframes():
    # Load the dataframes
    raw_df_train = pd.read_csv('data/train_set.csv', header=None)
    raw_df_test = pd.read_csv('data/test_set.csv', header=None)

    train_data = pd.DataFrame(columns=['classname', 'vector'])
    train_data['vector'] = [col for col in raw_df_train.iloc[:, :-1].values]
    train_data['classname'] = [col for col in raw_df_train.iloc[:, -1].values]

    test_data = pd.DataFrame(columns=['classname', 'vector'])
    test_data['vector'] = [col for col in raw_df_test.iloc[:, :-1].values]
    test_data['classname'] = [col for col in raw_df_test.iloc[:, -1].values]

    vector_length = len(test_data['vector'][0])

    return test_data, train_data, vector_length


class Perceptron:
    def __init__(self, weights, threshold, learning_rate):
        self.weights = np.array(weights)
        self.threshold = threshold
        self.learning_rate = learning_rate

    def predict(self, vector):
        prediction_sum = sum(
            [self.weights[i] * vector[i] for i in range(len(self.weights))]
        )
        return 'Iris-versicolor' if prediction_sum >= self.threshold else 'Iris-virginica'

    def train(self, vector, expected):
        prediction = self.predict(vector)

        if prediction == expected:
            return

        w = np.append(self.weights, self.threshold)
        x = np.append(vector, -1)
        d = 1 if expected == 'Iris-versicolor' else 0
        y = 1 if prediction == 'Iris-versicolor' else 0
        w_prime = w + self.learning_rate * (d - y) * x
        self.weights = w_prime[:-1]
        self.threshold = w_prime[-1]

    def __str__(self):
        return f'Perceptron({self.weights=} {self.threshold=} {self.learning_rate=})'


def main():
    test, train, vl = load_dataframes()
    # print(test)
    # print(train)

    # Initialize the perceptron
    perceptron = Perceptron(
        weights=[np.random.random() for _ in range(vl)],
        threshold=0,
        learning_rate=0.1
    )

    gen_iterations = 0
    accuracy = 0

    while accuracy < 1.0 and gen_iterations < 1000:
        test_result = pd.DataFrame(columns=['expected', 'predicted'])

        for index, row in test.iterrows():
            predicted = perceptron.predict(row['vector'])
            test_result = pd.concat([
                test_result,
                pd.DataFrame({'expected': [row['classname']], 'predicted': [predicted]})
            ])

        accuracy = len(test_result[test_result['expected'] == test_result['predicted']]) / len(test_result)
        print(f'{accuracy=} {gen_iterations=}')

        # Learn from the training data
        for index, row in train.iterrows():
            perceptron.train(row['vector'], row['classname'])
            # print(perceptron)

        gen_iterations += 1


if __name__ == '__main__':
    main()
