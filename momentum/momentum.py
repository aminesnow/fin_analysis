import yfinance as yf
import pandas as pd
from ta import add_all_ta_features
from ta.utils import dropna
from catboost import CatBoostClassifier
import math
import numpy as np
from lightgbm import LGBMClassifier
import argparse


def to_signal(returns):
    if returns >= 0.03:
        return 1
    if returns <= 0.0:
        return -1
    return 0

def std_x_y(df):
    #scaler = StandardScaler()
    X = df.drop(columns=['signal']).to_numpy()
    #scaler.fit(X)
    #X = scaler.fit_transform(X)
    Y = df['signal'].to_numpy()
    return X, Y

def benchmark(test_df):
    capital = ini_capital
    test_arr = test_df[['Close', 'signal_pred']].reset_index().to_numpy()
    capital = capital * (1 + (test_arr[len(test_arr)-1][1]-test_arr[0][1])/test_arr[0][1])
    print('returns_benchmark: {}'.format((capital-ini_capital)/ini_capital))



def backtest(test_df):
    capital = ini_capital
    shares = 0
    hold_duration = 0
    max_hold = 30

    test_arr = test_df[['Close', 'signal_pred']].reset_index().to_numpy()

    for idx, data in enumerate(test_arr):
        date, price, sig = data[0].date(), data[1], data[2]

        if (capital + shares * price) < price:
            print('chapter 11')
            break

        if shares == 0 and sig == 1:
            quant = math.floor(capital/price)
            shares = quant
            capital = capital - shares * price

        if shares > 0:
            hold_duration += 1
            if sig == -1 or hold_duration >= max_hold or idx == len(test_arr)-1:
                capital = capital + shares * price
                shares = 0
        
    print('returns_backtest: {}'.format((capital-ini_capital)/ini_capital))

def get_stock_data(sym):
    ticker = yf.Ticker(sym)  #PRX.AS ASML.AS
    raw_df = ticker.history(start='2010-01-01', end="2022-01-01", interval="1d")

    raw_df['returnsM'] = raw_df[['Close']].pct_change(periods=30)
    df = raw_df[['Open', 'High', 'Low', 'Close', 'Volume', 'returnsM']]
    df['returnsM_shift'] = df['returnsM'].shift(periods=-30)
    df['signal'] = df.apply(lambda r: to_signal(r['returnsM_shift']), axis=1)
    df.drop(columns=['returnsM', 'returnsM_shift'], inplace=True)

    # Clean NaN values
    df = dropna(df)

    # Add all ta features
    df = add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume")

    df = df.fillna(method='backfill')
    df = pd.concat([df[['Open', 'High', 'Low', 'Close', 'Volume', 'signal']], df.loc[:,df.columns.str.startswith('momentum')|df.columns.str.startswith('trend')]], join = 'outer', axis = 1)

    train_df = df.loc[:'2016-01-01']
    val_df = df.loc['2016-06-01':'2019-01-01']
    test_df = df.loc['2019-01-01':'2021-11-01']

    return train_df, val_df, test_df


ini_capital = 10000
rounds = 2000

if __name__=="__main__":
    # parse args
    parser = argparse.ArgumentParser(description="train mnist",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n-estimators', type=int, default=10,
                        help='the number of estimators')
    parser.add_argument('--learning-ate', type=float, default=0.01,
                    help='the learning rate') 
    parser.add_argument('--num-leaves', type=int, default=52,
                    help='number of leaves')
    parser.add_argument('--max-depth', type=int, default=12,
                help='max depth')
    params = {
        'boosting_type': 'goss',
        'n_estimators': 10000,
        'learning_rate': 0.005,
        'num_leaves': 52,
        'max_depth': 12,
        'subsample_for_bin': 24000,
        'reg_alpha': 0.45,
        'reg_lambda': 0.48,
        'colsample_bytree': 0.5,
        'min_split_gain': 0.025,
        'subsample': 1
    }

    #clf = CatBoostClassifier(
    #    iterations=3000, 
    #    learning_rate=0.08,
        #custom_loss=['AUC', 'Accuracy']
    #)


    clf = LGBMClassifier(**params)

    x_train, y_train = [], []
    x_val, y_val = [], []
    x_test, y_test = [], []

    symbols = ['TSLA', 'NFLX', 'AMD']

    tests = {}

    for s in symbols:
        print(s)
        train_df, val_df, test_df = get_stock_data(s)

        tests[s] = test_df

        x, y = std_x_y(train_df)
        if len(x_train) == 0:
            x_train = x
            y_train = y
        else:
            x_train = np.append(x_train, x, axis=0)
            y_train = np.append(y_train, y, axis=0)

        x, y = std_x_y(val_df)
        if len(x_val) == 0:
            x_val = x
            y_val = y
        else:
            x_val = np.append(x_val, x, axis=0)
            y_val = np.append(y_val, y, axis=0)
    
    clf.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=1000, early_stopping_rounds=rounds)


    for s in symbols:
        test_df = tests[s]
        x_test, y_test = std_x_y(test_df)

        test_df['signal_pred'] = clf.predict(x_test)

        backtest(test_df)
        benchmark(test_df)