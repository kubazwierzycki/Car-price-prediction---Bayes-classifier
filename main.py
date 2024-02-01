import numpy as np
import pandas as pd

#load and divide data to X_train, y_train, X_test, y_test
def load_data():
    data = pd.read_csv("audi.csv")  # audi.csv / ford.csv
    data = data[["model", "year", "price", "transmission", "mileage", "fuelType", "tax", "mpg", "engineSize"]]
    data = data.dropna().reset_index(drop=True)
    test_idx = np.random.choice(range(data.shape[0]), round(0.2*data.shape[0]), replace=False)
    data_test = data.iloc[test_idx, :]
    data_train = data.drop(test_idx, axis=0)
    X_train = data_train.drop("price", axis=1).to_numpy()
    y_train = data_train["price"].to_numpy()
    X_test = data_test.drop("price", axis=1).to_numpy()
    y_test = data_test["price"].to_numpy()
    return (X_train, y_train), (X_test, y_test)

#return list of index of top 3 values from list 
def max_index3(tab):
    value = [0, 0, 0]
    index_k = [0, 0, 0]
    for i in range(len(tab)):
        if value[0] <= tab[i]:
            index_k[2] = index_k[1]
            value[2] = value[1]
            index_k[1] = index_k[0]
            value[1] = value[0]
            index_k[0] = i
            value[0] = tab[i]
        elif value[1] <= tab[i]:
            index_k[2] = index_k[1]
            value[2] = value[1]
            index_k[1] = i
            value[1] = tab[i]
        elif value[2] <= tab[i]:
            index_k[2] = i
            value[2] = tab[i]
    return index_k

#round number to k
def round_to(x, k):
    x = int((x + (k / 2)) / k)
    return x * k

#return index of proper price range
def index_price(prices, price):
    for i in range(len(prices)):
        if (prices[i][0] <= price) and (prices[i][1] >= price):
            return i
    return 0

#use train data to build data model
def learn_data(X_train, y_train, prices):
    learnt_X = []
    temp = []
    for i in range(len(prices)):
        learnt_X.append([])
        temp.append([])
        for j in range(len(X_train[0])):
            learnt_X[i].append([])
    for i in range(len(X_train)):
        (temp[index_price(prices, y_train[i])]).append(X_train[i])
    for i in range(len(prices)):
        for j in range(len(X_train[0])):
            x = []
            for k in range(len(temp[i])):
                if j == 6:  #miles per gallon
                    x.append(round_to(temp[i][k][j], 10))
                elif j == 3:    #mileage
                    x.append(round_to(temp[i][k][j], 10000))
                else:
                    x.append(temp[i][k][j])
            a, b = np.unique(x, return_counts=True)
            learnt_X[i][j].append(a)
            learnt_X[i][j].append(b)

    return learnt_X

#return probabilities of being in each class
def bayes_classifier(learnt_X, car, n):
    probability = []

    for i in range(len(learnt_X)):
        b = sum(learnt_X[i][0][1])
        p = b / n
        for j in range(len(learnt_X[i])):
            if p == 0:
                break
            a = 0.0
            for k in range(len(learnt_X[i][j][0])):
                if j == 6:  #miles per gallon
                    if learnt_X[i][j][0][k] == round_to(car[j], 10):
                        a = learnt_X[i][j][1][k]
                        break
                elif j == 3:    #mileage
                    if learnt_X[i][j][0][k] == round_to(car[j], 10000):
                        a = learnt_X[i][j][1][k]
                        break
                else:
                     if learnt_X[i][j][0][k] == car[j]:
                        a = learnt_X[i][j][1][k]
                        break
            a += 0.01;  # add 0.01 to 'a' to avoid setting probability to zero and small enough to not change efficiency of the model
            p *= (a / b)
        probability.append(p)
    return probability

#return mean price according to top 3 probabilities
def mean_price(probability, prices):
    index = max_index3(probability)
    
    left_value = 0
    right_value = 0
    for i in range(3):
        left_value += (prices[index[i]][0] * probability[index[i]])
        right_value += (prices[index[i]][1] * probability[index[i]])

    left_value /= (probability[index[0]] + probability[index[1]] + probability[index[2]])
    right_value /= (probability[index[0]] + probability[index[1]] + probability[index[2]])

    return left_value, right_value

#check if price value is between the range
def compare_price(left_value, right_value, price):
    return (left_value <= price) and (right_value >= price)

#return an error from predicted price range
def calculate_error(left_value, right_value, value):
    if left_value > value:
        return left_value - value
    elif right_value < value:
        return value - right_value
    return 0.0

#run algorithm for each data in set, print mean error and return accuracy
def train_Bayess(X, y, learnt_X, X_train, prices):
    correct_answer = 0.0
    wrong_answer = 0.0
    error_rate = 0.0
    for i in range(len(X)):
        probability = bayes_classifier(learnt_X, X[i], len(X_train))
        left_value, right_value = mean_price(probability, prices)
        if compare_price(left_value, right_value, y[i]):
            correct_answer += 1
        else:
            wrong_answer += 1
        error_rate += calculate_error(left_value, right_value, y[i]) / y[i]
    print("Mean error [%]:", (error_rate / len(X)) * 100)
    return correct_answer / (correct_answer + wrong_answer)

############################################################################################

N = [1000, 2500, 5000, 10000]

for n in N:
    print("Price range:", n)
    (X_train, y_train), (X_test, y_test) = load_data()

    prices = []
    price_diffrence = n
    for i in range(int(150000 / price_diffrence)):
        prices.append([i * price_diffrence, (i+1) * price_diffrence])

    # learnt_X[prices][features][feature_name/feature_count][values]
    learnt_X = learn_data(X_train, y_train, prices)

    #run for train data
    print("Train data:")
    print("Correct answers [%]:", train_Bayess(X_train, y_train, learnt_X, X_train, prices) * 100)
    #run for test data
    print("Test data:")
    print("Correct answers [%]:", train_Bayess(X_test, y_test, learnt_X, X_train, prices) * 100)