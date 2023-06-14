import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error,r2_score





def evaluation(y_test, y_pred, test_pred):
    print("MSE",round(mean_squared_error(y_test,y_pred), 3))
    print("RMSE",round(np.sqrt(mean_squared_error(y_test,y_pred)), 3))
    print("MAE",round(mean_absolute_error(y_test,y_pred), 3))
    print("MAPE",round(mean_absolute_percentage_error(y_test,y_pred), 3))
    print("R2 Score : ", round(r2_score(y_test,y_pred), 3))

    plt.scatter(y_pred, y_test, color='red', marker='o')
    plt.scatter(y_test, y_test, color='blue')
    plt.plot(y_test, y_test, color='lime')
    plt.show()

    test_pred['Close_Prediction'] = y_pred

    print(test_pred)




def data_preparation(df):
    print(df.shape)
    print(df.head)
    df.dropna(inplace=True)

    print(df.shape)

    df.info()

    print(df.describe().T)

    return df





def price_everyday(df):
    df['Date'] = pd.to_datetime(df['Date'])

    plt.figure(figsize=(10, 6))

    plt.plot(df['Date'], df['Open'], label='Open')
    plt.plot(df['Date'], df['Close'], label='Close')
    plt.plot(df['Date'], df['High'], label='High')
    plt.plot(df['Date'], df['Low'], label='Low')

    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Stock Price over Time')
    plt.legend()

    # plt.savefig('./img/price_everyday.png', dpi=300, bbox_inches='tight')

    plt.show()





def volume_chart(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Tìm các thời điểm có khối lượng tăng đột biến
    threshold = df['Volume'].mean() + 2 * df['Volume'].std()
    anomaly_dates = df[df['Volume'] > threshold].index

    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['Volume'])
    plt.scatter(anomaly_dates, df.loc[anomaly_dates]['Volume'], color='red', label='Volume Spike')
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.title('Volume Chart with Anomalies')
    plt.legend()
    plt.show()

    # plt.savefig('./img/volume_chart.png', dpi=300, bbox_inches='tight')




def candlestick_chart(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    mpf.plot(df, type='candle', volume=True, figsize=(10, 6))


