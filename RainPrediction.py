# import pandas as pd
# df = pd.read_csv('Rainfall_Data_Germany_Complete.csv')
# r=df.head()
# print(r)

# importing the dependencies
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# loading the data from csv file to a pandas dataframe

Rain_data = pd.read_csv('Rainfall_Data_Germany_Complete.csv')

print(Rain_data.columns)
print(Rain_data.dtypes)

#dropping columns with object type because it can crash the algorithm

Rain_data = Rain_data.drop(columns=['datetime', 'preciptype'])  # Example only — use real names from above

#important columns with object type will be encoded since its needed for prediction, cant be deleted.
Rain_data['name'] = pd.factorize(Rain_data['name'])[0]
# Rain_data['sunrise'] = pd.factorize(Rain_data['sunrise'])[0]
Rain_data['sunrise'] = pd.to_datetime(Rain_data['sunrise'], errors='coerce')
Rain_data['sunrise'] = Rain_data['sunrise'].astype('int64') // 10**9  # Convert to seconds
Rain_data['sunset'] = pd.to_datetime(Rain_data['sunset'], errors='coerce')
Rain_data['sunset'] = Rain_data['sunset'].astype('int64') // 10**9  # Convert to seconds


Rain_data['sunset'] = pd.factorize(Rain_data['sunset'])[0]
Rain_data['conditions'] = pd.factorize(Rain_data['conditions'])[0]
Rain_data['description'] = pd.factorize(Rain_data['description'])[0]
Rain_data['icon'] = pd.factorize(Rain_data['icon'])[0]
Rain_data['stations'] = pd.factorize(Rain_data['stations'])[0]


#errors encountered been handled
Rain_data = Rain_data.drop(columns=['City'], errors='ignore')


# printing the first 5 columns of the dataframe
R= Rain_data.head()
print(R)

# number of rows & columns in the dataframe
N= Rain_data.shape
print(N)

# checking for missing values
M= Rain_data.isnull().sum()
print(M)


# Now extract features and labels
X = Rain_data.drop(columns=['precip']).values  # Replace with your target column name
Y = Rain_data['precip'].values

print("loook")
print(X)
print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state = 2)




#Training the Linear Regression model

model = LinearRegression()
model.fit(X_train, Y_train)


# Printing the parameter values (weights & bias)
print('weights = ', model.coef_)   # The weights (coefficients) for each feature
print('bias = ', model.intercept_) # The bias (intercept)

test_data_prediction = model.predict(X_test)
print(test_data_prediction)


# Its selecting index of the column for x-axis
feature_index = 11  # For example, precipcover hence it says about the x axis
print(Rain_data.columns[feature_index])
X_test_feature = X_test[:, feature_index]

# plot the scatter plot
plt.scatter(X_test_feature, Y_test, color='red', label="Actual")

# Plot the predictions
plt.plot(X_test_feature, test_data_prediction, color='blue', label="Predicted")

# Adding labels and title
plt.xlabel(Rain_data.columns[feature_index])
plt.ylabel('Precipitation (mm)')
plt.title('Actual vs Predicted Precipitation')

# Add legend
plt.legend()

# Show the plot
plt.show()
