import utils
import training
import pandas as pd
from sklearn.model_selection import train_test_split




path = "./data/NFLX.csv"

df = pd.read_csv(path)


"""
Lam sach data
"""
df = utils.data_preparation(df=df)
df_0 = df


"""
Data visualization
"""
# utils.candlestick_chart(df=df_0)

# utils.price_everyday(df=df)

# utils.volume_chart(df=df)



"""
Chia du lieu thanh tap training va testing
"""
train, test = train_test_split(df, test_size = 0.2)

test_pred = test.copy()

x_train = train[['Open', 'High', 'Low', 'Volume']].values
x_test = test[['Open', 'High', 'Low', 'Volume']].values

y_train = train['Close'].values
y_test = test['Close'].values



"""
Training mo hinh, 4 thuat toan regression
"""
training.linear_regression(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, test_pred=test_pred)

# training.support_vector_regression(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, test_pred=test_pred)

# training.gradient_boosting_regression(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, test_pred=test_pred)

# training.random_forest_regression(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, test_pred=test_pred)