import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import svm


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

# Standard Scaler
nds = ds.copy()
std_scale = StandardScaler()
normalise_c = nds.columns.difference(['Species'])
nds[normalise_c] = std_scale.fit_transform(nds[normalise_c])

x = nds.drop(['Species'], axis=1)
y = nds['Species']
# print(nds)

# this will split the dataset into train and test (80/20)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.8, random_state=0)

# Euclidean KNN, is this the triangle guy?

knn1 = KNeighborsClassifier(n_neighbors=6, weights='uniform', algorithm='auto', metric='euclidean')

# Manhattan's KNN, not a fan of KNN I much more prefer the fish market

knn2 = KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='auto', metric='manhattan')

# Okay N-Fold Validation here we go!
# I'm gonna play around with just the training data so using 80/20

clf = (svm.SVC(kernel='linear', C=1))
clf.fit(x_train, y_train)
best_mean = 0.0
bn = 0
mean_acc = []
n_range = range(2,11)
print("--------------------------------------------------------")
for n in n_range:
    scores = cross_val_score(clf, x_train, y_train, cv=n, scoring='accuracy')
    mean_score = scores.mean()
    mean_acc.append(mean_score)

    # print(f"Accuracy of {n}-Fold: {mean_score:.2f}")
    if mean_score > best_mean:
        best_mean = mean_score
        bn = n
print(f"\nN-Fold ------ Best N: {bn}, Best Accuracy: {best_mean:.2f}\n")
print("--------------------------------------------------------")

# It's time to plot the K-Value against the accuracy! WooHoo
plt.figure()
plt.plot(n_range, mean_acc, marker='o',  linestyle='-', color='r')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Number of Neighbors')
plt.xticks(n_range)
plt.grid(True)
# plt.show()

# Re-Evaluating this fella!
knn1 = KNeighborsClassifier(n_neighbors=bn, weights='uniform', algorithm='auto', metric='euclidean')
knn1.fit(x_train, y_train)
y_pre1 = knn1.predict(x_test)
accuracy1 = knn1.score(x_test, y_test)
print(f"Accuracy: {accuracy1:.2f}, n_neighbour: {bn}\n")
print(classification_report(y_test, y_pre1))
wdata_euc = y_test.index[y_test != y_pre1]
sdata_euc = ds.loc[wdata_euc]
print("----------------Misclassified Flowers-------------------")
print(sdata_euc)
print("--------------------------------------------------------")

# This is the re-evaluation of the manhattan version of knn!
knn2 = KNeighborsClassifier(n_neighbors=bn, weights='uniform', algorithm='auto', metric='manhattan')
knn2.fit(x_train, y_train)
y_pre2 = knn2.predict(x_test)
accuracy2 = knn2.score(x_test, y_test)
print(f"Accuracy: {accuracy2:.2f}, n_neighbour: {bn}\n")
print(classification_report(y_test, y_pre2))
wdata_mat = y_test.index[y_test != y_pre2]
sdata_mat = ds.loc[wdata_mat]
print("----------------Misclassified Flowers-------------------")
print(sdata_mat)
print("--------------------------------------------------------")

# Naive Bayes Section (Credits to MARC SOOOOOOOOO)

gnb = GaussianNB()
gnb.fit(x_train, y_train)

y_pred = gnb.predict(x_test)
accuracy = metrics.accuracy_score(y_test, y_pred) * 100
print(f"\nAccuracy (in %): {accuracy:.2f}")

print(classification_report(y_test, y_pred))
print("--------------------------------------------------------")

misclassified_indices = y_test.index[y_test != y_pred]
misclassified_flowers = ds.loc[misclassified_indices]
print("----------------Misclassified Flowers-------------------")
print(misclassified_flowers)
print("-------------------------END----------------------------")