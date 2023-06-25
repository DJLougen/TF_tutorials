import pandas as pd
import numpy as np

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# Load in data
data = pd.read_csv("KAggle data/ds_salaries.csv")

data.head()
# work_year: The year the salary was paid.
# experience_level: The experience level in the job during the year
# employment_type: The type of employment for the role
# job_title: The role worked in during the year.
# salary: The total gross salary amount paid.
# salary_currency: The currency of the salary paid as an ISO 4217 currency code.
# salaryinusd: The salary in USD
# employee_residence: Employee's primary country of residence in during the work year as an ISO 3166 country code.
# remote_ratio: The overall amount of work done remotely
# company_location: The country of the employer's main office or contracting branch
# company_size: The median number of people that worked for the company during the year


  inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)
# Split into features and labels
data_feature = data.copy()
data_label = data.pop('salary_currency') # predicting salary of DS

#Pre-process
inputs = {} 
    #For loop to match data to type
for name, column in data.items():
  dtype = column.dtype
  if dtype == object:
    dtype = tf.string
  else:
    dtype = tf.float32

#Pack features into array
data_feature = np.array(data_feature)

#Regression model 
model = tf.keras.Sequential([
    layers.Dense(64),
    layers.Dense(1)
])

#Compile data 
model.compile(loss = tf.keras.losses.MeanSquaredError(),
                optimizer = tf.keras.optimizers.Adam())
#Train model 
model.fit(data_feature, data_label, epochs= 100)