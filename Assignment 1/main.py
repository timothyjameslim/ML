import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

# load the csv here (ds = dataset)
ds = pd.read_csv("Iris_Assignment 1.csv", header=0, index_col=0)

# ------------- DATA PREPROCESSING  -------------

# Question 1
# 1a)
# find missing data values
for i in range(len(ds)):
    for j in ds.columns:
        if pd.isnull(ds.loc[ds.index[i],j]):
            print(f"ID: '{ds.index[i]}' Column: '{j}' is blank")
            # This will print all the IDs with missing data

# remove the missing data
ds.dropna(inplace=True)
print("\n--------- The above data with missing fields has been removed ---------\n")
# I removed the data because the columns is data sensitive
# if I filled it with other numbers it may affect the final result

# 1b)
# find duplicate records
duplicates = ds[ds.duplicated(keep=False)]
print(duplicates, "\n--------- The above duplicated data has been removed ---------\n")
# I removed the data as duplicate data may cause over-fitting

# remove the duplicated data
ds.drop_duplicates(inplace=True)

# Question 2
ds.boxplot()
plt.savefig('boxplot.jpg', format='jpg')
# plt.show()
# 2a) boxplot plots the outliers as circles within the diagram.
# 2b) The diagram shows 4 outliers within the sepalWidthCm.

# Question 3
plt.figure()
sns.set_style("ticks")
sns.pairplot(ds, hue="Species")
plt.savefig('scatter.jpg', format='jpg')
# plt.show()
# From the scatter plot, we can visually group the various species.
# Iris-Setosa generally have the smallest Petal Length (< 2cm) and Petal Width (estimated, <0.8cm)
# Iris-Versicolor tend have Petal Width within(estimated, 1 to 1.5 cm) and Petal Length (estimated, 3 to 5 cm)
# Iris-Virginica generally have the largest Petal Width (estimated, >1.5cm)and larger Petal Length (estimated, >4.8cm)

# Question 4
plt.figure()
trainMap = ds.pivot_table(index="Species", aggfunc="median")
sns.heatmap(trainMap, annot=True)
plt.savefig('heatmap.jpg', format='jpg')
# plt.show()

# 4a)
# Positively correlated are features such as the Sepal width and Sepal Length
# where the range between the species are within 2.8 to 3.4 and 5 to 6.5 respectively.

# Negatively correlated are features such as the Petal Length and Petal Width where the ranges between the species
# are at least 1cm apart and 0.7cm to 1.1cm apart.

# 4b)
# Based on Petal Length, we should be able to determine the species
# of the Iris alone, with Petal Width to support the decisions.

# KNN Starts here :)

# # Encode Data
# label_encoder = LabelEncoder()
# ds['Species'] = label_encoder.fit_transform(ds['Species'])
#
# # Normalize Data
# nds = ds.copy()
# columns_to_normalize = nds.columns.difference(['Species'])
# for column in columns_to_normalize:
#     nds[column] = nds[column] / nds[column].abs().max()
#
# # Decode Data (Used for printing)
# nds['Species'] = label_encoder.inverse_transform(nds['Species'])

# Standard Scaler
nds = ds.copy()
std_scale = StandardScaler()
normalise_c = nds.columns.difference(['Species'])
nds[normalise_c] = std_scale.fit_transform(nds[normalise_c])

print(nds)

# this will split the dataset into train and test (80/20)
x_train, x_test = train_test_split(ds, test_size=0.2, train_size=0.8, random_state=0)
# print("x_train: \n", x_train)
# print("x_test: \n", x_test)
# print("x_train.shape: ", x_train.shape)
# print("x_test.shape: ", x_test.shape)


# need to normalize data first!!!!!!!!!!!!!!!!
