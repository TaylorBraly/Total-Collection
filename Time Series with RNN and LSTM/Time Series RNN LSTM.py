import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
# plotting
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
# normalizing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

seq_len = 60


class RNN:
    def __init__(self, X, Y, header=True, hidden_layer_size=1):
        np.random.seed(1)

        self.hidden_layer_size = hidden_layer_size
        self.X = X
        self.Y = Y
        input_layer_size = 1
        output_layer_size = 1  # len(self.Y)

        # self.Wxh = 2 * np.random.random((hidden_layer_size, input_layer_size)) - 1 #w01 input gate
        # self.Whh = 2 * np.random.random((hidden_layer_size, hidden_layer_size)) - 1 #w12 forget gate
        # self.Why = 2 * np.random.random((output_layer_size, hidden_layer_size)) - 1 #w23 output gate
        # self.bh = np.zeros((hidden_layer_size, 1))
        # self.by = np.zeros((output_layer_size, 1))

        ########
        # Formula outputs for LSTM
        self.input_activation = np.empty(shape=(len(self.X[0]), hidden_layer_size))
        self.input_gate = np.empty(shape=(len(self.X[0]), hidden_layer_size))
        self.forget_gate = np.empty(shape=(len(self.X[0]), hidden_layer_size))
        self.output_gate = np.empty(shape=(len(self.X[0]), hidden_layer_size))
        self.internal_state = np.empty(shape=(len(self.X[0]), hidden_layer_size))
        self.H = np.empty(shape=(len(self.X[0]), hidden_layer_size))  # self.output
        # Weight matrices for LSTM (analogous to Wxh)
        self.W_activation = 2 * np.random.random((len(self.X[0][0]), hidden_layer_size)) - 1
        self.W_input = 2 * np.random.random((len(self.X[0][0]), hidden_layer_size)) - 1
        self.W_forget = 2 * np.random.random((len(self.X[0][0]), hidden_layer_size)) - 1
        self.W_output = 2 * np.random.random((len(self.X[0][0]), hidden_layer_size)) - 1
        # Weight matrices for LSTM (analogous to Whh)
        self.U_activation = 2 * np.random.random((hidden_layer_size, hidden_layer_size)) - 1
        self.U_input = 2 * np.random.random((hidden_layer_size, hidden_layer_size)) - 1
        self.U_forget = 2 * np.random.random((hidden_layer_size, hidden_layer_size)) - 1
        self.U_output = 2 * np.random.random((hidden_layer_size, hidden_layer_size)) - 1
        # Bias vectors for LSTM
        self.b_activation = 2 * np.random.random((hidden_layer_size, 1)) - 1
        self.b_input = 2 * np.random.random((hidden_layer_size, 1)) - 1
        self.b_forget = 2 * np.random.random((hidden_layer_size, 1)) - 1
        self.b_output = 2 * np.random.random((hidden_layer_size, 1)) - 1
        # Weight matrix and bias vector from h to z (aka predicted y)
        self.Why = 2 * np.random.random((output_layer_size, hidden_layer_size)) - 1
        self.by = np.zeros((output_layer_size, 1))
        # self.Whz = 2 * np.random.random((hidden_layer_size, 1)) - 1
        # self.bz  = 2 * np.random.random((1, 1)) - 1
        ########

    def train(self, data, max_iterations=10, learning_rate=0.2):
        print('W_activation', self.W_activation)
        print('W_input', self.W_input)
        print('W_forget', self.W_forget)
        print('W_output', self.W_output)

        print('U_activation', self.U_activation)
        print('U_input', self.U_input)
        print('U_forget', self.U_forget)
        print('U_output', self.U_output)

        print('b_activation', self.b_activation)
        print('b_input', self.b_input)
        print('b_forget', self.b_forget)
        print('b_output', self.b_output)

        self.data = data[seq_len:]

        ########
        # inputs = np.empty((0,seq_len), float)
        # inputs = np.append(inputs, self.X[0])
        # inputs = np.reshape(inputs, (-seq_len, 1))
        ########
        out = np.empty((0, 1), float)
        j = 0
        check = True

        # print('train out:')
        for i in range(0, len(self.data)):
            inputs = np.empty((0, seq_len), float)
            inputs = np.append(inputs, self.X[i])
            inputs = np.reshape(inputs, (-seq_len, 1))
            # print(inputs)
            y = self.forward_pass(inputs)
            # print('y_train', y)
            # print('arctanh(y)',np.arctanh(y))
            dy = np.empty((0, 1), float)
            # y = np.arctanh(y)
            delta_y = (self.Y[i] - y)
            # print('self.Y[i]', self.Y[i])
            y = np.reshape(y, (-1, 1))
            dy = np.append(dy, np.array(delta_y), axis=0)
            # error += .5*delta_y**2
            j += 1
            # print(delta_y)
            # print(j, error)
            # if(np.absolute(delta_y) < .001):
            #     check = False
            # if(check == True):
            # if(i<300):
            # if(np.absolute(delta_y) > .01):
            self.backward_pass(y, dy, learning_rate)
            # print(j)
            # y = self.forward_pass(inputs)
            out = np.append(out, np.array(y), axis=0)

            ########
            # p = np.concatenate([inputs[1:seq_len], y], axis=0)
            # inputs = np.empty((0,seq_len), float)
            # inputs = np.append(inputs, p)
            # inputs = np.reshape(inputs, (-seq_len, 1))
            ########

            # error = 0.5 * np.power((out.flatten() - self.Y), 2)
            # print("The total error is " + str(np.sum(error)/len(out)))
        print()
        print('W_activation', self.W_activation)
        print('W_input', self.W_input)
        print('W_forget', self.W_forget)
        print('W_output', self.W_output)

        print('U_activation', self.U_activation)
        print('U_input', self.U_input)
        print('U_forget', self.U_forget)
        print('U_output', self.U_output)

        print('b_activation', self.b_activation)
        print('b_input', self.b_input)
        print('b_forget', self.b_forget)
        print('b_output', self.b_output)

        return out

    def tr(self, data, max_iterations=10, learning_rate=0.2):
        self.data = data[seq_len:]

        ########
        # inputs = np.empty((0,seq_len), float)
        # inputs = np.append(inputs, self.X[0])
        # inputs = np.reshape(inputs, (-seq_len, 1))
        ########
        out = np.empty((0, 1), float)

        # print('train out:')
        for i in range(0, len(self.data)):
            inputs = np.empty((0, seq_len), float)
            inputs = np.append(inputs, self.X[i])
            inputs = np.reshape(inputs, (-seq_len, 1))
            # print(inputs)
            y = self.forward_pass(inputs)
            # print('y_train', y)
            # print('arctanh(y)',np.arctanh(y))
            dy = np.empty((0, 1), float)
            # y = np.arctanh(y)
            delta_y = (self.Y[i] - y)
            # print('self.Y[i]', self.Y[i])
            y = np.reshape(y, (-1, 1))
            # dy = np.append(dy, np.array(delta_y), axis=0)
            # self.backward_pass(y, dy, learning_rate)
            # y = self.forward_pass(inputs)
            out = np.append(out, np.array(y), axis=0)

            ########
            # p = np.concatenate([inputs[1:seq_len], y], axis=0)
            # inputs = np.empty((0,seq_len), float)
            # inputs = np.append(inputs, p)
            # inputs = np.reshape(inputs, (-seq_len, 1))
            ########

            # error = 0.5 * np.power((out.flatten() - self.Y), 2)
            # print("The total error is " + str(np.sum(error)/len(out)))

        return out

    def test(self, data, X, Y, max_iterations=1000, learning_rate=0.8):
        # print('self.X[T]', self.X[983])
        # print('self.last_inputs', self.last_inputs)
        self.data = data[seq_len:]
        self.X = X
        # print('len self.X', len(self.X))
        self.Y = Y

        # inputs = np.empty((0,seq_len), float)
        # inputs = np.append(inputs, self.X[0])
        # inputs = np.reshape(inputs, (-seq_len, 1))
        out = np.empty((0, 1), float)
        error = 0

        for i in range(0, len(self.data)):
            inputs = np.empty((0, seq_len), float)
            inputs = np.append(inputs, self.X[i])
            inputs = np.reshape(inputs, (-seq_len, 1))

            y = self.forward_pass(inputs)
            # if(i < 7):
            # print('inputs', inputs)
            # print('Y[i]', Y[i])
            # print('y', y)
            delta_y = (self.Y[i] - y)
            # error = np.sqrt(0.5 * delta_y**2)
            # print('error = ', error)
            y = np.reshape(y, (-1, 1))
            # print('y_test', y)
            out = np.append(out, np.array(y), axis=0)

            dy = np.empty((0, 1), float)
            # y = np.arctanh(y)
            delta_y = (self.Y[i] - y)
            # print('self.Y[i]', self.Y[i])
            y = np.reshape(y, (-1, 1))
            # dy = np.append(dy, np.array(delta_y), axis=0)
            # self.backward_pass(y, dy, learning_rate)

            # p = np.concatenate([inputs[1:seq_len], y], axis=0)
            # inputs = np.empty((0,seq_len), float)
            # inputs = np.append(inputs, p)
            # inputs = np.reshape(inputs, (-seq_len, 1))

        # error = 0.5 * np.power((out.flatten() - self.Y), 2)
        # print("The total prior error is " + str(np.sum(error)/len(out)))
        return out

    def forward_pass(self, inputs):
        # h = np.zeros((self.Whh.shape[0], 1))
        # self.H = { 0: h }
        self.last_inputs = inputs

        for t in range(0, len(inputs)):
            x = inputs[t]

            # h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            if t == 0:
                out_minus_1 = np.zeros(shape=(len(self.H[t]), 1))
                internal_state_minus_1 = 0

                self.input_activation[t] = tanh(
                    self.W_activation.T * x + np.dot(self.U_activation, out_minus_1) + self.b_activation).reshape(
                    self.hidden_layer_size)
                # print("Input Activation: ", self.input_activation[t])

                self.input_gate[t] = sigmoid(
                    self.W_input.T * x + np.dot(self.U_input, out_minus_1) + self.b_input).reshape(
                    self.hidden_layer_size)
                # print("Input Gate: ", self.input_gate[t])

                self.forget_gate[t] = sigmoid(
                    self.W_forget.T * x + np.dot(self.U_forget, out_minus_1) + self.b_forget).reshape(
                    self.hidden_layer_size)
                # print("Forget Gate: ", self.forget_gate[t])

                self.output_gate[t] = sigmoid(
                    self.W_output.T * x + np.dot(self.U_output, out_minus_1) + self.b_output).reshape(
                    self.hidden_layer_size)
                # print("Output Gate: ", self.output_gate[t])

                self.internal_state[t] = (
                            np.multiply(self.input_activation[t], self.input_gate[t]) + np.multiply(self.forget_gate[t],
                                                                                                    internal_state_minus_1)).reshape(
                    self.hidden_layer_size)
                # print("Internal State: ", self.internal_state[t])

                self.H[t] = (np.multiply(tanh(self.internal_state[t]), self.output_gate[t])).reshape(
                    self.hidden_layer_size)

            else:
                self.input_activation[t] = tanh(
                    self.W_activation.T * x + np.dot(self.U_activation, self.H[t - 1]).reshape(self.hidden_layer_size,
                                                                                               1) + self.b_activation).reshape(
                    self.hidden_layer_size)
                # print("Input Activation: ", self.input_activation[t])

                self.input_gate[t] = sigmoid(
                    self.W_input.T * x + np.dot(self.U_input, self.H[t - 1]).reshape(self.hidden_layer_size,
                                                                                     1) + self.b_input).reshape(
                    self.hidden_layer_size)
                # print("Input Gate: ", self.input_gate[t])

                self.forget_gate[t] = sigmoid(
                    self.W_forget.T * x + np.dot(self.U_forget, self.H[t - 1]).reshape(self.hidden_layer_size,
                                                                                       1) + self.b_forget).reshape(
                    self.hidden_layer_size)
                # print("Forget Gate: ", self.forget_gate[t])

                self.output_gate[t] = sigmoid(
                    self.W_output.T * x + np.dot(self.U_output, self.H[t - 1]).reshape(self.hidden_layer_size,
                                                                                       1) + self.b_output).reshape(
                    self.hidden_layer_size)
                # print("Output Gate: ", self.output_gate[t])

                self.internal_state[t] = (
                            np.multiply(self.input_activation[t], self.input_gate[t]) + np.multiply(self.forget_gate[t],
                                                                                                    self.internal_state[
                                                                                                        t - 1])).reshape(
                    self.hidden_layer_size)
                # print("Internal State: ", self.internal_state[t])

                self.H[t] = (np.multiply(tanh(self.internal_state[t]), self.output_gate[t])).reshape(
                    self.hidden_layer_size)

            # self.H[i + 1] = h
            y = np.tanh(np.dot(self.Why, self.H[t]) + self.by)
            # print('yyy', y)
        return y

    def backward_pass(self, out, d_y, learn_rate):
        # print('Wa', self.W_activation)
        # print('Wa shape', self.W_activation.shape)
        # self.compute_output_delta(out)
        T = len(self.last_inputs) - 1
        # last_inputs = np.empty((0,1), float)
        # last_inputs = last_inputs[0:self.last_inputs]
        # print('l_inp', last_inputs)

        d_Why = np.dot(d_y, self.H[T].T)
        # print('dy', d_y.shape)
        # print('Hn', self.H[n].T.shape)
        # print('dWhy', d_Why.shape)
        d_by = d_y

        # d_Whh = np.zeros(self.Whh.shape)
        # d_Wxh = np.zeros(self.Wxh.shape)
        # d_bh = np.zeros(self.bh.shape)

        # d_h = np.dot(self.Why.T, d_y)
        delta_h = np.zeros_like(self.H[0])
        d_internal_state = np.zeros_like(self.H[0])

        d_a = d_i = d_f = d_o = np.zeros_like(self.H[0])
        d_Ua = d_Ui = d_Uf = d_Uo = np.zeros_like(self.H[0])
        d_Wa = d_Wi = d_Wf = d_Wo = np.zeros_like(self.H[0])
        d_ba = d_bi = d_bf = d_bo = np.zeros_like(self.H[0])

        # Backpropagate through time.
        for t in reversed(range(T)):
            d_h = d_y + delta_h

            if t == T:
                d_internal_state = d_h * self.output_gate[t] * (1 - tanh(self.internal_state[t]) ** 2)
            else:
                d_Ua = d_Ua + d_a * self.H[t]
                d_Ui = d_Ui + d_i * self.H[t]
                d_Uf = d_Uf + d_f * self.H[t]
                d_Uo = d_Uo + d_o * self.H[t]

                d_internal_state = d_h * self.output_gate[t] * (
                            1 - tanh(self.internal_state[t]) ** 2) + d_internal_state * self.forget_gate[t + 1]

            d_a = d_internal_state * self.input_gate[t] * (1 - (self.input_activation[t]) ** 2)
            # d_a = np.reshape(d_a, (-1, 1))
            # print('d_a: ', d_a)
            # print('d_a shape: ', d_a.shape)

            d_i = d_internal_state * self.input_activation[t] * self.input_gate[t] * (1 - self.input_gate[t])

            d_f = d_internal_state * self.internal_state[t - 1] * self.forget_gate[t] * (1 - self.forget_gate[t])

            d_o = d_h * tanh(self.internal_state[t]) * self.output_gate[t] * (1 - self.output_gate[t])

            last_input = np.empty((0, 1), float)
            last_input = np.append(last_input, self.last_inputs[t])
            last_input = np.reshape(last_input, (-1, 1))
            # print('last_input', last_input)
            # print('last_input shape', last_input.shape)
            d_Wa = d_Wa + d_a * last_input
            d_Wi = d_Wi + d_i * last_input
            d_Wf = d_Wf + d_f * last_input
            d_Wo = d_Wo + d_o * last_input

            d_ba = d_ba + d_a
            d_bi = d_bi + d_i
            d_bf = d_bf + d_f
            d_bo = d_bo + d_o

            delta_ha = self.U_activation * d_a
            delta_hi = self.U_input * d_i
            delta_hf = self.U_forget * d_f
            delta_ho = self.U_output * d_o

            delta_h = delta_ha + delta_hi + delta_hf + delta_ho
            # d_c = (1 - (np.tanh(self.input_activation[t])**2)) * self.output_gate[t] * d_h

            # An intermediate value: dL/dh * (1 - h^2)
            # temp = ((1 - self.H[t + 1] ** 2) * d_h)
            # print('temp', temp)

            # dL/db = dL/dh * (1 - h^2)
            # d_bh += temp

            # dL/dWhh = dL/dh * (1 - h^2) * h_{t-1}
            # d_Whh += temp @ self.H[t].T

            # dL/dWxh = dL/dh * (1 - h^2) * x
            # d_Wxh += temp @ self.last_inputs[t].T

            # Next dL/dh = dL/dh * (1 - h^2) * Whh
            # d_h = self.Whh @ temp

        # Clip to prevent exploding gradients.
        for d in [d_Why, d_by, d_a, d_i, d_f, d_o, d_Ua, d_Ui, d_Uf, d_Uo, d_Wa, d_Wi, d_Wf, d_Wo, d_ba, d_bi, d_bf,
                  d_bo]:
            np.clip(d, -1, 1, out=d)

        # Update weights and biases using gradient descent.
        # self.Whh -= learn_rate * d_Whh
        # self.Wxh -= learn_rate * d_Wxh
        self.Why = self.Why - learn_rate * d_Why
        # self.bh -= learn_rate * d_bh
        self.by = self.by + learn_rate * d_by

        self.W_activation = self.W_activation - learn_rate * d_Wa
        self.W_input = self.W_input - learn_rate * d_Wi
        self.W_forget = self.W_forget - learn_rate * d_Wf
        self.W_output = self.W_output - learn_rate * d_Wo

        self.U_activation = self.U_activation - learn_rate * d_Ua
        self.U_input = self.U_input - learn_rate * d_Ui
        self.U_forget = self.U_forget - learn_rate * d_Uf
        self.U_output = self.U_output - learn_rate * d_Uo

        self.b_activation = self.b_activation - learn_rate * d_ba
        self.b_input = self.b_input - learn_rate * d_bi
        self.b_forget = self.b_forget - learn_rate * d_bf
        self.b_output = self.b_output - learn_rate * d_bo


def sigmoid(x):
    x = np.array(x, dtype=np.float32)

    return 1 / (1 + np.exp(-x))


# tanh activation function

def tanh(x):
    # print("Before X: ", x)
    # print("After X: ", (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)))
    x = np.array(x, dtype=np.float32)

    return np.tanh(x)


def fill_mean(arr):
    for i in range(0, len(arr)):
        if arr[i][0] == 0:
            arr[i][0] = (arr[i - 1][0] + arr[i + 1][0]) / 2
    return arr


def preprocess(X):
    # replaces '.' with '0'
    fill_NaN = SimpleImputer(missing_values='.', strategy='constant', fill_value='0')
    imputed_X = pd.DataFrame(fill_NaN.fit_transform(X))
    imputed_X.columns = X.columns
    imputed_X.index = X.index

    # change {DATE, NASDAQCOM} datatypes to {datetime, float}
    # and make DATE the index
    imputed_X = imputed_X.astype({'NASDAQCOM': 'float64'})
    imputed_X['DATE'] = pd.to_datetime(imputed_X.DATE, format='%Y-%m-%d')
    imputed_X.index = imputed_X['DATE']
    imputed_X.drop(['DATE'], inplace=True, axis=1)
    # print('imputed_X: ', imputed_X)

    # replaces 0 with mean
    # fill_zero = SimpleImputer(missing_values=0)
    # proc_X = pd.DataFrame(fill_zero.fit_transform(imputed_X))
    imputed_arr = imputed_X.to_numpy()
    proc_arr = fill_mean(imputed_arr)
    proc_X = pd.DataFrame(proc_arr, columns=['NASDAQCOM'])
    proc_X.columns = imputed_X.columns
    proc_X.index = imputed_X.index

    return proc_X


if __name__ == "__main__":
    df = pd.read_csv('NASDAQCOM.csv', header=0)
    proc_df = preprocess(df)

    # plot the processed data
    # plt.figure(figsize=(16,8))
    # plt.plot(proc_df['NASDAQCOM'], label='Close Price history')
    # plt.show()

    # take only the values of the dataset and standardize them from 0-1
    dataset = proc_df.values
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(dataset)

    # take first 80% as our training set, next 20% as our validation set
    # _80p = round(0.8*len(dataset))
    _80p = len(dataset) - 7
    train = scaled_data[0:_80p, :]
    valid = scaled_data[(_80p):, :]
    test_data = scaled_data[(_80p - seq_len):, :]

    # only take blocks of seq_len days as info that will affect our prediction
    x_train, y_train = [], []
    for i in range(seq_len, len(train)):
        x_train.append(scaled_data[i - seq_len:i, 0])  # seq_len days / arr entry
        y_train.append(scaled_data[i, 0])  # 61st day prediction
    x_train, y_train = np.array(x_train), np.array(y_train)
    # print('xtrain', x_train[0:4])
    # print('ytrain', y_train[0:4])

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # TRAINING
    rnn = RNN(x_train, y_train)
    # preds, h = rnn.train()

    a = rnn.train(train)
    tr_pr = scaler.inverse_transform(a)
    first_sixty = np.zeros((seq_len, 1))
    training_price = np.empty((0, 1), float)
    training_price = np.append(training_price, first_sixty)
    training_price = np.append(training_price, tr_pr)
    training_price = np.reshape(training_price, (-1, 1))
    # print('len a', len(a))
    # print('len training price', len(training_price))

    # print('preds:\n', preds[0:4])
    # closing_price = scaler.inverse_transform(preds)

    # TESTING
    # only take blocks of seq_len days as info that will affect our prediction
    inputs = scaled_data[len(dataset) - len(valid) - seq_len:]
    inputs = inputs.reshape(-1, 1)
    # print('len inputs', len(inputs))
    x_test, y_test = [], []
    for i in range(seq_len, len(test_data)):
        x_test.append(test_data[i - seq_len:i, 0])  # seq_len days / arr entry
        y_test.append(test_data[i, 0])  # 61st day prediction
    x_test, y_test = np.array(x_test), np.array(y_test)
    # print('xtest', x_test[259-seq_len])
    # print('xtest', x_test[259])
    # print('len x_test', len(x_test))
    # print('len x_test[0]', len(x_test[0]))

    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    ######################### WORK HERE ##################################
    # test_data = np.append(train[:-seq_len], valid)
    # print('len valid', len(valid))
    # print('test_data', test_data)
    # print('len testdata', len(test_data))
    b = rnn.tr(train)
    b_price = scaler.inverse_transform(b)
    # print('len b', len(b))
    # print('len training price', len(training_price))
    first_sixty = np.zeros((seq_len, 1))
    tr_price = np.empty((0, 1), float)
    tr_price = np.append(tr_price, first_sixty)
    tr_price = np.append(tr_price, b_price)
    tr_price = np.reshape(tr_price, (-1, 1))
    ######################################################################
    # print('Prediction Error=', np.sum(error)/len(preds)*100)
    preds = rnn.test(test_data, x_test, y_test)
    closing_price = scaler.inverse_transform(preds)

    # calculate root mean squared error
    # rms=np.sqrt(np.mean(np.power((valid-closing_price),2)))
    # print('rms= ', rms)

    # plot our predictions
    train = proc_df[:_80p]
    train['TrainingPrice'] = training_price
    # train['bPrice'] = tr_price
    y_true = y_pred = rmse = acc = 0
    for i in range(len(train)):
        y_true = train['NASDAQCOM'][i]
        y_pred = train['TrainingPrice'][i]
        acc += 100 - np.absolute(y_true - y_pred) * 100 / y_true
        rmse += np.sqrt(((y_true - y_pred) ** 2) / len(train))
    acc = acc / len(train)
    rmse = rmse / len(train)

    out = proc_df[_80p:]
    out['Predictions'] = closing_price
    print(out)
    y_true = y_pred = rmse = acc = 0
    for i in range(len(out)):
        y_true = out['NASDAQCOM'][i]
        y_pred = out['Predictions'][i]
        acc += 100 - np.absolute(y_true - y_pred) * 100 / y_true
        rmse += np.sqrt(((y_true - y_pred) ** 2) / len(out))
    acc = acc / len(out)
    rmse = rmse / len(out)

    print()
    print('train accuracy = ', acc)
    print('train rmse = ', rmse)
    print('test accuracy = ', acc)
    print('test rmse = ', rmse)
    # plt.plot(proc_df['NASDAQCOM'])
    plt.plot(train[['NASDAQCOM', 'TrainingPrice']])
    plt.plot(out[['NASDAQCOM', 'Predictions']])
    plt.show()