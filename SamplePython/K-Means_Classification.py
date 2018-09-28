from statsmodels.api import datasets
iris = datasets.get_rdataset("iris")
iris.data.columns = ['Sepal_length', 'Sepal_Width', 'Petal_Length', 'Pental_Width', 'Species']
print(iris.data.head())

print(iris.data.dtypes)

iris.data['count'] = 1
iris.data[['Species', 'count']].groupby('Species').count()

#matplotlib incline
def plot_iris(iris, col1, col2):
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.lmplot(x = col1, y = col2,
                data = iris,
                hue = "Species",
                fit_reg = False)
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title('Iris species shown by color')
    plt.show()

plot_iris(iris.data, 'Petal_Width', 'Sepal_Length')
plot_iris(iris.data, 'Sepal_Width', 'Sepal_Length')



from sklearn.preprocessing import scale
import pandas as pd
num_cols = ['Sepal_length', 'Sepal_Width', 'Petal_Length', 'Pental_Width']
iris_scaled = scale(iris.data[num_cols])
iris_scaled = pd.DataFrame(iris_scaled, columns = num_cols)
print(iris_scaled.describe().round(3))

# translate species to numerical value
levels = {'sentosa':0, 'versicolor':1, 'virginca':2}
iris_scaled['Species'] = [levels[x] for x in iris.data['Species']]
iris_scaled.head()


## Split the data into a training and test set by Bernoulli sampling
from sklearn.model_selection import train_test_split
import numpy as np
np.random.seed(3456)
iris_split = train_test_split(np.asmatrix(iris_scaled), test_size = 75)
iris_train_features = iris_split[0][:, :4]
iris_train_labels = np.ravel(iris_split[0][:, 4])
iris_test_features = iris_split[1][:, :4]
iris_test_labels = np.ravel(iris_split[1][:, 4])
print(iris_train_features.shape)
print(iris_train_labels.shape)
print(iris_test_features.shape)
print(iris_test_labels.shape)



## Define and train the KNN model
from sklearn.neighbors import KNeighborsClassifier
KNN_mod = KNeighborsClassifier(n_neighbors=3)
KNN_mod.fit(iris_train_features, iris_train_labels)

iris_test = pd.DataFrame(iris_test_features, columns = num_cols)
iris_test['predicted'] = KNN_mod.predict(iris_test_features)
iris_test['correct'] = [1 if x == z else 0 for x, z in zip(iris_test['predicted'], iris_test_labels)]
accuracy = 100.0 * float(sum(iris_test['correct'])) / float(iris_test.shape[0])
print(accuracy)



levels = {0:'sentosa', 1:'versicolor', 2:'virginica'}
iris_test['Species'] = [levels[x] for x in iris_test['predicted']]
markers = {1:'^', 0:'o'}
colors = {'setosa':'blue', 'versicolor':'green', 'virginica':'red'}
def plot_shapes(df, col1, col2, markers, colors):
    import matplotlib.pyplot as plt
    import seaborn as sns
    ax = plt.figure(figsize=(6, 6)).gca() # defien plot axis
    for m in markers:
        for c in colors:
            df_temp = df[(df['correct'] == m) & (df['Species'] == c)]
            sns.regplot(x = col1, y = col2,
                        data = df_temp,
                        fit_reg=False,
                        scatter_kws={'color': colors[c]},
                        ax = ax)
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title('Iris species by color')
    return 'Done'

plot_shapes(iris_test, 'Petal_Width', 'Sepal_Length', markers, colors)
plot_shapes(iris_test, 'Sepal_Width', 'Sepal_Length', markers, colors)
