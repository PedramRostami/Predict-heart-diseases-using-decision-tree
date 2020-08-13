import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import tree
from os import system



data_path = '../dataset/heart.csv'

def preprocessing():
    df = pd.read_csv(data_path, encoding='utf-8')
    df['fbs'] = df['fbs'].astype('int64')

    for i in range(10):
        df.loc[(df['oldpeak'] >= float(i)) & (df['oldpeak'] < float(i + 1)), 'oldpeak'] = i
    df['oldpeak'] = df['oldpeak'].astype('int64')

    for i in range(10):
        thalach_range = range(70 + (i * 20), 70 + ((i + 1) * 20))
        df.loc[df.thalach.isin(thalach_range), 'thalach'] = i
    df['thalach'] = df['thalach'].astype('int64')

    for i in range(10):
        chol_range = range(100 + (i * 50), 100 + ((i + 1) * 50))
        df.loc[df.chol.isin(chol_range), 'chol'] = i
    df['chol'] = df['chol'].astype('int64')

    for i in range(13):
        trestbps_range = range(90 + (i * 10), 90 + ((i + 1) * 10))
        df.loc[df.trestbps.isin(trestbps_range), 'trestbps'] = i
    df['trestbps'] = df['trestbps'].astype('int64')

    for i in range(10):
        years = range(i * 10, (i + 1) * 10)
        df.loc[df.age.isin(years), 'age'] = i
    df['age'] = df['age'].astype('int64')

    cleans_up = {
        "sex": {"male": 0, "female": 1},
        "cp": {"none": 0, "medium": 1, "weak": 2, "severe": 3},
        "exang": {"no": 0, "yes": 1},
        "thal": {"normal": 0, "fixed_defect": 1, "eversable_defect": 2},
        "target": {"no": 0, "yes": 1}
    }
    df.replace(cleans_up, inplace=True)
    df['sex'] = df['sex'].astype('int64')
    df['cp'] = df['cp'].astype('int64')
    df['exang'] = df['exang'].astype('int64')
    df['exang'] = df['exang'].astype('int64')
    df['target'] = df['target'].astype('int64')
    return df

def get_test_train():
    data = preprocessing()
    data = data.drop(data.columns[0], axis=1)
    train_data = data.sample(frac= 0.8, random_state= 200)
    test_data = data.drop(train_data.index)
    train_data = train_data.reset_index(drop= True)
    test_data = test_data.reset_index(drop= True)
    return (train_data, test_data)


def decisoin_tree_test1() :
    train_data, test_data = get_test_train()
    feature_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    train_data_x = train_data[feature_cols]
    train_data_y = train_data.target
    test_data_x = test_data[feature_cols]
    test_data_y = test_data.target

    clf = DecisionTreeClassifier()
    clf = clf.fit(train_data_x, train_data_y)
    y_pred = clf.predict(test_data_x)
    print("Accuracy : ", metrics.accuracy_score(test_data_y, y_pred))
    dotfile = open("tree.dot", 'w')
    tree.export_graphviz(clf, out_file=dotfile, feature_names=feature_cols,
                                    class_names=['0', '1'], filled=True, rounded=True,
                                    special_characters=True)
    dotfile.close()
    system("dot -Tpng tree.dot -o dtree1.png")

def decisoin_tree_test2() :
    train_data, test_data = get_test_train()
    feature_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    train_data_x = train_data[feature_cols]
    train_data_y = train_data.target
    test_data_x = test_data[feature_cols]
    test_data_y = test_data.target

    clf = DecisionTreeClassifier(criterion='entropy')
    clf = clf.fit(train_data_x, train_data_y)
    y_pred = clf.predict(test_data_x)
    print("Accuracy : ", metrics.accuracy_score(test_data_y, y_pred))
    dotfile = open("tree.dot", 'w')
    tree.export_graphviz(clf, out_file=dotfile, feature_names=feature_cols,
                                    class_names=['0', '1'], filled=True, rounded=True,
                                    special_characters=True)
    dotfile.close()
    system("dot -Tpng tree.dot -o dtree2.png")

def decisoin_tree_test3() :
    train_data, test_data = get_test_train()
    feature_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    train_data_x = train_data[feature_cols]
    train_data_y = train_data.target
    test_data_x = test_data[feature_cols]
    test_data_y = test_data.target

    clf = DecisionTreeClassifier(min_samples_leaf=10)
    clf = clf.fit(train_data_x, train_data_y)
    y_pred = clf.predict(test_data_x)
    print("Accuracy : ", metrics.accuracy_score(test_data_y, y_pred))
    dotfile = open("tree.dot", 'w')
    tree.export_graphviz(clf, out_file=dotfile, feature_names=feature_cols,
                                    class_names=['0', '1'], filled=True, rounded=True,
                                    special_characters=True)
    dotfile.close()
    system("dot -Tpng tree.dot -o dtree3.png")

def decisoin_tree_test4() :
    train_data, test_data = get_test_train()
    feature_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    train_data_x = train_data[feature_cols]
    train_data_y = train_data.target
    test_data_x = test_data[feature_cols]
    test_data_y = test_data.target

    clf = DecisionTreeClassifier(max_depth= 3)
    clf = clf.fit(train_data_x, train_data_y)
    y_pred = clf.predict(test_data_x)
    print("Accuracy : ", metrics.accuracy_score(test_data_y, y_pred))
    dotfile = open("tree.dot", 'w')
    tree.export_graphviz(clf, out_file=dotfile, feature_names=feature_cols,
                                    class_names=['0', '1'], filled=True, rounded=True,
                                    special_characters=True)
    dotfile.close()
    system("dot -Tpng tree.dot -o dtree4.png")


if __name__ == '__main__':
    decisoin_tree_test1()
    decisoin_tree_test2()
    decisoin_tree_test3()
    decisoin_tree_test4()

