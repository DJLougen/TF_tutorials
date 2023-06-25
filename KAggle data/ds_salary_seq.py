import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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


# Split into features and labels
X = data.iloc[:, :-1]  # Features (with column titles)
y = data.iloc[:, -1]  # Labels

# Test/Train
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=420
)  # Random state = seed


# Define the model
model = keras.models.Sequential()
model.add(keras.layers.Dense(64, input_shape=(X_train.shape[1],), activation="relu"))
# 64 neurons
# Specifies the shape of the input into the first layer
# relu = Rectified Linear Unit
model.add(keras.layers.Dense(64, activation="relu"))  # Takes input from prior layer
model.add(
    keras.layers.Dense(1, activation="sigmoid")
)  # Classification requires 1 neuron output because binary, sigmoid common for binary problems
