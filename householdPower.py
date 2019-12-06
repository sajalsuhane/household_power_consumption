# load and clean-up data
from numpy import nan
from numpy import isnan
from numpy import split
from numpy import array
from pandas import read_csv
from pandas import to_numeric
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
 
# fill missing values with a value at the same time one day ago
def fill_missing(values):
    one_day = 60 * 24
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            if isnan(values[row, col]):
                values[row, col] = values[row - one_day, col]
 
# load all data
dataset = read_csv('household_power_consumption.txt', sep=';', header=0, low_memory=False, infer_datetime_format=True, parse_dates={'datetime':[0,1]}, index_col=['datetime'])

# mark all missing values
dataset.replace('?', nan, inplace=True)

# make dataset numeric
dataset = dataset.astype('float32')

# fill missing
fill_missing(dataset.values)

# add a column for for the remainder of sub metering
values = dataset.values
dataset['sub_metering_4'] = (values[:,0] * 1000 / 60) - (values[:,4] + values[:,5] + values[:,6])

# save updated dataset
dataset.to_csv('household_power_consumption.csv')

# resample data to daily
daily_groups = dataset.resample('D')
daily_data = daily_groups.sum()

# summarize
print(daily_data.shape)
print(daily_data.head())

# save
daily_data.to_csv('household_power_consumption_days.csv')

# evaluate one or more weekly forecasts against expected values
def evaluate_forecasts(actual, predicted):
        scores = list()
    # calculate an RMSE score for each day
        for i in range(actual.shape[1]):
    # calculate mse
            mse  = mean_squared_error(actual[:, i], predicted[:, i])
    # calculate rmse
            rmse = sqrt(mse)
    # store
            scores.append(rmse)
    # calculate overall RMSE
        s = 0
        for row in range(actual.shape[0]):
            for col in range(actual.shape[1]):
                s += (actual[row, col] - predicted[row, col])**2
                score = sqrt(s / (actual.shape[0] * actual.shape[1]))
        return score, scores

# split a univariate dataset into train/test sets
def split_dataset(data):
# split into standard weeks
    train, test = data[1:-328], data[-328:-6]
# restructure into windows of weekly data
    train = array(split(train, len(train)/7))
    test = array(split(test, len(test)/7))
    return train, test

# load the new file
dataset = read_csv('household_power_consumption_days.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
train, test = split_dataset(dataset.values)
# validate train data
print(train.shape)
print(train[0, 0, 0], train[-1, -1, 0])
# validate test
print(test.shape)
print(test[0, 0, 0], test[-1, -1, 0])

# evaluate a single model
def evaluate_model(model_func, train, test):
# history is a list of weekly data
    history = [x for x in train]
# walk-forward validation over each week
    predictions = list()
    for i in range(len(test)):
# predict the week
        yhat_sequence = model_func(history)
# store the predictions
        predictions.append(yhat_sequence)
# get real observation and add to history for predicting the next week
        history.append(test[i, :])
        predictions = array(predictions)
# evaluate predictions days for each week
        score, scores = evaluate_forecasts(test[:, :, 0], predictions)
    return score, scores

# summarize scores
def summarize_scores(name, score, scores):
    s_scores = ', '.join(['%.1f' % s for s in scores])
    print('%s: [%.3f] %s' % (name, score, s_scores))

# convert windows of weekly multivariate data into a series of total power
def to_series(data):
# extract just the total power from each week
    series = [week[:, 0] for week in data]
# flatten into a single series
    series = array(series).flatten()
    return series

# load the new file
dataset = read_csv('household_power_consumption_days.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
# split into train and test
train, test = split_dataset(dataset.values)
# convert training data into a series
series = to_series(train)
# plots
pyplot.figure()
lags = 365
# acf
axis = pyplot.subplot(2, 1, 1)
plot_acf(series, ax=axis, lags=lags)
# pacf
axis = pyplot.subplot(2, 1, 2)
plot_pacf(series, ax=axis, lags=lags)
# show plot
pyplot.show()

# arima forecast
def arima_forecast(history):
    # convert history into a univariate series
    series = to_series(history)
    # define the model
    model = ARIMA(series, order=(7,0,0))
    # fit the model
    model_fit = model.fit(disp=False)
    # make forecast
    yhat = model_fit.predict(len(series), len(series)+6)
    return yhat

# load the new file
dataset = read_csv('household_power_consumption_days.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
# split into train and test
train, test = split_dataset(dataset.values)
# define the names and functions for the models we wish to evaluate
models = dict()
models['arima'] = arima_forecast
# evaluate each model
days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
for name, func in models.items():
    # evaluate and get scores
    score, scores = evaluate_model(func, train, test)
    # summarize scores
    summarize_scores(name, score, scores)
    # plot scores
    pyplot.plot(days, scores, marker='o', label=name)
# show plot
pyplot.legend()
pyplot.show()
