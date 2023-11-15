import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler


path = "C:\\Users\\N\\Documents\\KMP\\Yeast.csv"
dataset = pd.read_csv(path)

#Task 1-----------------------------------------------------------------
missing = dataset.isna()
count_missing = missing.sum().sum()
print(f"К-ть пропущених значень у вихідному датасеті: {count_missing}")

dataset.iloc[:, :-1] = dataset.iloc[:, :-1].apply(lambda col: col.fillna(col.mean()), axis=0) 

missing = dataset.isna()
#count_missing = missing.sum().sum()
#print(f"К-ть пропущених значень після заповнення: {count_missing}")

print("Статистична інформація:")
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print(dataset.describe(),"\n")

print(f"Кількість записів: {len(dataset)}\n")

duplicates = dataset.duplicated()
num_duplicates = duplicates.sum()
print(f"Кількість дублікатів у вихідному датасеті: {num_duplicates}\n")
dataset.drop_duplicates(inplace=True)

features = ', '.join(dataset.columns)
print(f"Кількість ознак класифікації: {len(dataset.columns)}\n")
print("Всі ознаки класифікації:")
print(features)

print("\n Кількість записів у кожному класі", dataset.groupby('name').size(), "\n")

#Task 2-----------------------------------------------------------------

font_size = 8
xticks_orientation = 'vertical'  
yticks_orientation = 'horizontal'
color_wheel={"MIT":"red", 
             "NUC":"blue",
             "ERL":"grey", 
             "ME1":"pink",
             "ME2":"green",
             "ME3":"orange",
             "POX":"purple",
             "VAC":"brown",
             "EXC":"maroon",
             "CYT":"yellow"}
colors = dataset["name"].map(lambda x: color_wheel.get(x))
scatter_matrix(dataset, color=colors)
for ax in plt.gcf().get_axes():
    ax.xaxis.label.set_rotation(xticks_orientation)
    ax.yaxis.label.set_rotation(yticks_orientation)
    ax.yaxis.label.set_ha('right') 
    ax.xaxis.label.set_fontsize(font_size)
    ax.yaxis.label.set_fontsize(font_size)
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(font_size)
plt.show()


#Task 3-----------------------------------------------------------------
data = dataset.iloc[:,:-1].values
clas = dataset.iloc[:,-1].values
np.random.seed(42)
Size_train, Size_test, Class_train, Class_test = \
train_test_split(data,clas,test_size=0.2)

classifier_bayes = GaussianNB()
classifier_bayes.fit(Size_train,Class_train)
test_predict = classifier_bayes.predict(Size_test)
score_bayes = classifier_bayes.score(Size_test, Class_test)
print("Результати класифікації до масштабування, метод наївного Байєса:")
print("Частка правильних передбачень = ", score_bayes, "\n")

#Task 4-----------------------------------------------------------------
print("Звіт щодо продуктивності класифікатора:")
print(classification_report(Class_test,test_predict),"\n")

#Task 5-----------------------------------------------------------------
cm = confusion_matrix(Class_test, test_predict)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=classifier_bayes.classes_)
disp.plot()
plt.title('Матриця неточностей, метод наївного Байєса без масштабування')
plt.show()


#Task 6-----------------------------------------------------------------
np.random.seed(42)
Size_train2, Size_test2, Class_train2, Class_test2 = \
train_test_split(data,clas,test_size=0.2)
scaler = StandardScaler()
scaler.fit(Size_train2)
Size_train2 = scaler.transform(Size_train2)
Size_test2 = scaler.transform(Size_test2)
classifier_bayes2 = GaussianNB()
classifier_bayes2.fit(Size_train2,Class_train2)
test_predict2 = classifier_bayes2.predict(Size_test2)
score_bayes2 = classifier_bayes2.score(Size_test2, Class_test2)
print("Результати класифікації після масштабування, метод наївного Байєса:")
print("Частка правильних передбачень = ", score_bayes2, "\n")

print("Звіт щодо продуктивності класифікатора:")
print(classification_report(Class_test2,test_predict2),"\n")

cm2 = confusion_matrix(Class_test2, test_predict2)
disp = ConfusionMatrixDisplay(confusion_matrix=cm2,display_labels=classifier_bayes2.classes_)
disp.plot()
plt.title('Матриця неточностей, метод наївного Байєса з масштабуванням')
plt.show()


#Task 7-----------------------------------------------------------------
Size_train, Size_test, Class_train, Class_test = \
train_test_split(data,clas,test_size=0.2)
scaler = StandardScaler()
scaler.fit(Size_train)
Size_train = scaler.transform(Size_train)
Size_test = scaler.transform(Size_test)

classifier_tree = DecisionTreeClassifier(max_depth=3, min_samples_split=2)
classifier_tree.fit(Size_train, Class_train)
test_predict_tree = classifier_tree.predict(Size_test)
score = classifier_tree.score(Size_test,Class_test)
print("Частка правильних передбачень для методу дерева рішень з max_depth=3, min_samples_split =2 = ", score, "\n")
print("Звіт щодо продуктивності класифікатора:")
print(classification_report(Class_test,test_predict_tree),"\n")
unique_classes = sorted(set(Class_train) | set(Class_test))
cm = confusion_matrix(Class_test, test_predict_tree, labels=unique_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_classes)
disp.plot()
plt.title('Матриця неточностей')
plt.show()


#Task 9-----------------------------------------------------------------
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

def find_nearest_neighbors(test_data, train_data, train_labels):
    nearest_neighbors = []
    for test_instance in test_data:
        distances = [euclidean_distance(test_instance, train_instance) for train_instance in train_data]
        nearest_neighbor_index = np.argmin(distances)
        nearest_neighbor_label = train_labels[nearest_neighbor_index]
        nearest_neighbors.append(nearest_neighbor_label)
    return nearest_neighbors

selected_objects = []
selected_labels = []
for class_label in set(Class_test):
    class_indices = np.where(Class_test == class_label)[0][:3]
    selected_objects.extend(Size_test[class_indices])
    selected_labels.extend(Class_test[class_indices])

nearest_neighbors = find_nearest_neighbors(selected_objects, Size_train, Class_train)

for i, (test_label, nearest_neighbor) in enumerate(zip(selected_labels, nearest_neighbors)):
    print(f"Об'єкт {i+1} з класу {test_label} має найближчого сусіда з класу {nearest_neighbor}")