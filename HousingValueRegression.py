import pandas as pd
data = pd.read_csv('cal_housing_clean.csv')
data.head()
data.describe()
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
x_data = data.drop(['medianHouseValue'],axis=1)
y_data =  data['medianHouseValue']
x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,test_size=0.3,random_state=101)

scaler = MinMaxScaler()
scaler.fit(x_train)
X_train = pd.DataFrame(data=scaler.transform(x_train),columns = x_train.columns,index=x_train.index)
X_test= pd.DataFrame(data=scaler.transform(x_test),columns = x_test.columns,index=x_test.index)

'''Create Feature Columns'''
data.columns
import tensorflow as tf
age = tf.feature_column.numeric_column('housingMedianAge')
rooms = tf.feature_column.numeric_column('totalRooms')
bedrooms = tf.feature_column.numeric_column('totalBedrooms')
pop = tf.feature_column.numeric_column('population')
households = tf.feature_column.numeric_column('households')
income = tf.feature_column.numeric_column('medianIncome')

feat_cols = [ age,rooms,bedrooms,pop,households,income]
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train ,batch_size=10,num_epochs=1000,
                                            shuffle=True)
model = tf.estimator.DNNRegressor(hidden_units=[6,6,6],feature_columns=feat_cols)
model.train(input_fn=input_func,steps=25000)
predict_input_func = tf.estimator.inputs.pandas_input_fn(
      x=X_test,
      batch_size=10,
      num_epochs=1,
      shuffle=False)
pred_gen = model.predict(predict_input_func)
predictions = list(pred_gen)
final_preds = []
for pred in predictions:
    final_preds.append(pred['predictions'])
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test,final_preds)**0.5













