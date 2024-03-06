from sklearn import svm
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn import metrics
from preprocessing import preprocessing
from chefboost import Chefboost as chef
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import MinMaxScaler
from imblearn.metrics import sensitivity_specificity_support
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import  classification_report

def main():
    person1 = "adam"
    person2 = "andrew"
    person3 = "john"
    person4 = "stephen"
    person5 = "waleed"

    adam(person1)
    andrew(person2)
    john(person3)
    stephen(person4)
    waleed(person5)

def adam(person):
    # ********************************************* ADAM *********************************************
    print("**************************************** ADAM ****************************************")
    adamPrep = preprocessing(person)
    adam_x = adamPrep.getPersonDataFrame  # get all features as adam_x as data frame
    # print(adam_x.info())

    adam_y = adam_x["label"]  # get the label from adam_x as adam_y
    del adam_x['label']  # delete label column from adam_x

    # normalize data
    scaler = MinMaxScaler()
    adam_x_normalized = pd.DataFrame(data=scaler.fit_transform(adam_x.iloc[:, :]), columns=adam_x.columns)

    naiveBayesClassifier(adam_x_normalized, adam_y)
    SVM(adam_x_normalized, adam_y)
    # c45DecisionTree(adam_x_normalized, adam_y)
    NN(1, len(adam_x.columns), adam_x_normalized, adam_y, 95)
    NN(2, len(adam_x.columns), adam_x_normalized, adam_y, 95)

def andrew(person):
    # ********************************************* ANDREW *********************************************
    print("**************************************** ANDREW ****************************************")
    andrewPrep = preprocessing(person)
    andrew_x = andrewPrep.getPersonDataFrame  # get all features as adam_x as data frame
    # print(andrew_x.info())

    andrew_y = andrew_x["label"]  # get the label from adam_x as adam_y
    del andrew_x['label']  # delete label column from adam_x

    # normalize data
    scaler = MinMaxScaler()
    andrew_x_normalized = pd.DataFrame(data=scaler.fit_transform(andrew_x.iloc[:, :]), columns=andrew_x.columns)

    naiveBayesClassifier(andrew_x_normalized, andrew_y)
    SVM(andrew_x_normalized, andrew_y)
    #c45DecisionTree(andrew_x_normalized, andrew_y)
    NN(1, len(andrew_x.columns), andrew_x_normalized, andrew_y, 95)
    NN(2, len(andrew_x.columns), andrew_x_normalized, andrew_y, 95)

def john(person):
    # ********************************************* JOHN *********************************************
    print("**************************************** JOHN ****************************************")
    johnPrep = preprocessing(person)
    john_x = johnPrep.getPersonDataFrame  # get all features as adam_x as data frame
    print(john_x.info())

    john_y = john_x["label"]  # get the label from adam_x as adam_y
    del john_x['label']  # delete label column from adam_x

    # normalize data
    scaler = MinMaxScaler()
    john_x_normalized = pd.DataFrame(data=scaler.fit_transform(john_x.iloc[:, :]), columns=john_x.columns)

    naiveBayesClassifier(john_x_normalized, john_y)
    SVM(john_x_normalized, john_y)
    #c45DecisionTree(john_x_normalized, john_y)
    NN(1, len(john_x.columns), john_x_normalized, john_y, 95)
    NN(2, len(john_x.columns), john_x_normalized, john_y, 95)

def stephen(person):
    # ********************************************* STEPHEN *********************************************
    print("**************************************** STEPHEN ****************************************")
    stephenPrep = preprocessing(person)
    stephen_x = stephenPrep.getPersonDataFrame  # get all features as adam_x as data frame
    print(stephen_x.info())

    stephen_y = stephen_x["label"]  # get the label from adam_x as adam_y
    del stephen_x['label']  # delete label column from adam_x

    # normalize data
    scaler = MinMaxScaler()
    stephen_x_normalized = pd.DataFrame(data=scaler.fit_transform(stephen_x.iloc[:, :]), columns=stephen_x.columns)

    naiveBayesClassifier(stephen_x_normalized, stephen_y)
    SVM(stephen_x_normalized, stephen_y)
    #  c45DecisionTree(stephen_x_normalized, stephen_y)
    NN(1, len(stephen_x.columns), stephen_x_normalized, stephen_y, 98)
    NN(2, len(stephen_x.columns), stephen_x_normalized, stephen_y, 98)

def waleed(person):
    # ********************************************* WALEED *********************************************
    print("**************************************** WALEED ****************************************")
    waleedPrep = preprocessing(person)
    waleed_x = waleedPrep.getPersonDataFrame  # get all features as adam_x as data frame
    print(waleed_x.info())

    waleed_y = waleed_x["label"]  # get the label from adam_x as adam_y
    del waleed_x['label']  # delete label column from adam_x

    # normalize data
    scaler = MinMaxScaler()
    waleed_x_normalized = pd.DataFrame(data=scaler.fit_transform(waleed_x.iloc[:, :]), columns=waleed_x.columns)

    naiveBayesClassifier(waleed_x_normalized, waleed_y)
    SVM(waleed_x_normalized, waleed_y)
    #c45DecisionTree(waleed_x_normalized, waleed_y)
    NN(1, len(waleed_x.columns), waleed_x_normalized, waleed_y, 98)
    NN(2, len(waleed_x.columns), waleed_x_normalized, waleed_y, 98)


def naiveBayesClassifier(adam_x_normalized, adam_y):
    print("------------------------- NAIVE BAYES CLASSIFIER -------------------------")

    n = 5
    sss_normalized = StratifiedShuffleSplit(n_splits=n, test_size=0.15, random_state=0)
    sss_normalized.get_n_splits(adam_x_normalized, adam_y)

    accuracyTot = 0
    acc_tot =  pre_mic_tot =  pre_mac_tot = re_mic_tot = re_mac_tot = f1_mic_tot = f1_mac_tot = error = 0
    result_mic =  result_mac = [0,0]
    
    for train_index, test_index in sss_normalized.split(adam_x_normalized, adam_y):
        x_train, x_test = adam_x_normalized.loc[train_index], adam_x_normalized.loc[test_index]
        y_train, y_test = adam_y[train_index], adam_y[test_index]

        # Create a Gaussian Classifier
        gnb = GaussianNB()

        # Train the model using the training sets
        gnb.fit(x_train, y_train)

        # Predict the response for test dataset
        y_pred = gnb.predict(x_test)

        accuracyNB = metrics.accuracy_score(y_test, y_pred)

        accuracy, precision_micro, precision_macro, recall_micro, recall_macro, f1_micro, f1_macro,result_micro,result_macro =  calcPerformance(y_test, y_pred)

        acc_tot += accuracy
        pre_mic_tot += precision_micro
        pre_mac_tot += precision_macro
        re_mic_tot += recall_micro
        re_mac_tot += recall_macro
        f1_mic_tot += f1_micro
        f1_mac_tot += f1_macro
        result_mic[0] += result_micro[0]
        result_mic[1] += result_micro[1]
        result_mac[0] += result_macro[0]
        result_mac[1] += result_macro[1]
        
        error += 1 - accuracy
      
        
        if accuracyNB > accuracyTot:
            x_train_high = x_train
            x_test_high = x_test
            y_train_high = y_train
            y_test_high = y_test
            accuracyTot = accuracyNB

    print('Accuracy: ', acc_tot /n)
    
    print('Error: ', error /n)
    
    print('Precision micro: ', pre_mic_tot /n)

    print('Precision macro: ', pre_mac_tot /n)

    print('Recall micro: ', re_mic_tot /n)

    print('Recall macro: ', re_mac_tot /n)

    print('f1 micro:', f1_mic_tot /n)

    print('f1 macro:', f1_mac_tot /n)

    print('sensitivity micro: ', result_mic[0]/n)
    print('specificity micro: ', result_mic[1]/n)

    print('sensitivity macro: ', result_mac[0]/n)
    print('specificity macro: ', result_mac[1]/n)
    
  
    
    gnb = GaussianNB()
    gnb.fit(x_train_high, y_train_high)
    y_pred = gnb.predict(x_test_high)

    # calculate accuracy, precision, recall, F1 score
    plot_confusion_matrix(y_test_high, y_pred)
    print("")


def SVM(adam_x_normalized, adam_y):
    print("------------------------- SUPPORT VECTOR MACHINES -------------------------")
    
    n=5
    sss_normalized = StratifiedShuffleSplit(n_splits=n, test_size=0.15, random_state=0)
    sss_normalized.get_n_splits(adam_x_normalized, adam_y)

    accuracyTot = 0
    acc_tot =  pre_mic_tot =  pre_mac_tot = re_mic_tot = re_mac_tot = f1_mic_tot = f1_mac_tot = error = 0
    result_mic =  result_mac = [0,0]
    
    for train_index, test_index in sss_normalized.split(adam_x_normalized, adam_y):
        x_train, x_test = adam_x_normalized.loc[train_index], adam_x_normalized.loc[test_index]
        y_train, y_test = adam_y[train_index], adam_y[test_index]

      #  print("best:", svc_param_selection(x_train, y_train, 5))   # to find the best parameters

        clf = svm.SVC(kernel='poly', C=0.1, gamma=1)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)

        accuracySVM = metrics.accuracy_score(y_test, y_pred)
        
        accuracy, precision_micro, precision_macro, recall_micro, recall_macro, f1_micro, f1_macro,result_micro,result_macro =  calcPerformance(y_test, y_pred)

        acc_tot += accuracy
        error += 1- accuracy
        pre_mic_tot += precision_micro
        pre_mac_tot += precision_macro
        re_mic_tot += recall_micro
        re_mac_tot += recall_macro
        f1_mic_tot += f1_micro
        f1_mac_tot += f1_macro
        result_mic[0] += result_micro[0]
        result_mic[1] += result_micro[1]
        result_mac[0] += result_macro[0]
        result_mac[1] += result_macro[1]
        
        if accuracySVM > accuracyTot:
            x_train_high = x_train
            x_test_high = x_test
            y_train_high = y_train
            y_test_high = y_test
            accuracyTot = accuracySVM

    print('Accuracy: ', acc_tot /n)
    
    print('Error: ', error /n)
    
    print('Precision micro: ', pre_mic_tot /n)

    print('Precision macro: ', pre_mac_tot /n)

    print('Recall micro: ', re_mic_tot /n)

    print('Recall macro: ', re_mac_tot /n)

    print('f1 micro:', f1_mic_tot /n)

    print('f1 macro:', f1_mac_tot /n)

    print('sensitivity micro: ', result_mic[0]/n)
    print('specificity micro: ', result_mic[1]/n)

    print('sensitivity macro: ', result_mac[0]/n)
    print('specificity macro: ', result_mac[1]/n)
    
    clf = svm.SVC(kernel='poly', C=0.1, gamma=1)
    clf.fit(x_train_high, y_train_high)
    y_pred = clf.predict(x_test_high)
    plot_confusion_matrix(y_test_high, y_pred)


def svc_param_selection(X, y, nfolds):
    # RBF => best: {'C': 10, 'gamma': 0.1} %0.36    linear => best: {'C': 10, 'gamma': 0.001} %0.67
    # poly => best: {'C': 0.001, 'gamma': 0.1}  %0.64    sigmoid => best: {'C': 0.001, 'gamma': 0.01} %0.0065

    # poly => best: {'C': 0.1, 'gamma': 1} 0.8859649122807017         linear => best: {'C': 10, 'gamma': 0.001} 0.8771929824561403
    # rbf =>  best: {'C': 10, 'gamma': 0.1} 0.8596491228070176         sigmoid => best: {'C': 10, 'gamma': 0.1} 0.38596491228070173
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma': gammas}
    grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_


# ---------------------------TREE METHODS------------------------------------------------------
def c45DecisionTree(adam_x_normalized, adam_y):

    sss_normalized = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=0)
    sss_normalized.get_n_splits(adam_x_normalized, adam_y)

    for train_index, test_index in sss_normalized.split(adam_x_normalized, adam_y):
        x_train, x_test = adam_x_normalized.loc[train_index], adam_x_normalized.loc[test_index]
        y_train, y_test = adam_y[train_index], adam_y[test_index]

        # add label column with Decision tag to x_train to give the data to tree algortihms
        train_data = x_train.copy()
        train_data['Decision'] = y_train

        test_data = x_test.copy()
        test_data['Decision'] = y_test

        config = {'algorithm': 'C4.5'}
        model = chef.fit(train_data, config)

        # prediction = chef.predict(model, test_data)
        x_test = test_data.iloc[:, :-1]
        y_test = test_data.iloc[:, -1]
        y_pred = list()
        for i in x_test.values.tolist():
            y_pred.append(chef.predict(model, i))

        y_pred = np.array(y_pred)

        calcPerformance(y_test, y_pred)
        plot_confusion_matrix(y_test,y_pred)



'''
def CARTDecisionTree(train_data, test_data):
    config = {'algorithm': 'CART'}
    model = chef.fit(train_data, config)

    prediction = chef.predict(model, test_data)

    return model, prediction
'''

# ------------------------Neural Networks-----------------------------------------------------

def NN(noOfLayers, noOfFeatures, adam_x_normalized, adam_y, labelSize):

    n = 5
    sss_normalized = StratifiedShuffleSplit(n_splits=n, test_size=0.15, random_state=0)
    sss_normalized.get_n_splits(adam_x_normalized, adam_y)

    accuracyTot = 0
    acc_tot = pre_mic_tot = pre_mac_tot = re_mic_tot = re_mac_tot = f1_mic_tot = f1_mac_tot = error = 0
    result_mic = result_mac = [0, 0]

    for train_index, test_index in sss_normalized.split(adam_x_normalized, adam_y):
        x_train, x_test = adam_x_normalized.loc[train_index], adam_x_normalized.loc[test_index]
        y_train, y_test = adam_y[train_index], adam_y[test_index]

        if noOfLayers == 1:
            model = Sequential()
            # first hidden layer
            model.add(Dense(95, input_dim=noOfFeatures, activation='tanh'))
            # output layer
            model.add(Dense(labelSize, activation='softmax'))

        elif noOfLayers == 2:
            model = Sequential()
            # first hidden layer
            model.add(Dense(95, input_dim=noOfFeatures, activation='tanh'))
            # second hidden layer
            model.add(Dense(95, activation='tanh'))
            # output layer , 1 ->dimensions(noOfclasses) of output layer

            model.add(Dense(labelSize, activation='softmax'))

        # For a multi-class classification problem
        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        x_train = np_utils.normalize(x_train.to_numpy())
        x_test = np_utils.normalize(x_test.to_numpy())

        # encode class values as integers
        encoder = LabelEncoder()
        encoder.fit(y_train)
        encoded_Y = encoder.transform(y_train).T
        # Convert labels to categorical one-hot encoding
        one_hot_labels_train = keras.utils.to_categorical(encoded_Y, num_classes=labelSize)

        encoder.fit(y_test)
        encoded_Y = encoder.transform(y_test).T
        # Convert labels to categorical one-hot encoding
        one_hot_labels_test = keras.utils.to_categorical(encoded_Y, num_classes=labelSize)

        history = model.fit(x_train, one_hot_labels_train, epochs=500, batch_size=32,
                            validation_data=(x_test, one_hot_labels_test), verbose=0)

        # print(history.history.keys())
        # score = model.evaluate(x, one_hot_labels, batch_size=32)

        plt.figure(1)

        # summarize history for accuracy

        plt.subplot(2, 1, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
      #  plt.show()

        # summarize history for loss

        plt.subplot(2, 1, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        y_pred = model.predict_classes(x_test, batch_size=32)

        accuracy, precision_micro, precision_macro, recall_micro, recall_macro, f1_micro, f1_macro, result_micro, result_macro = calcPerformance(
            encoded_Y, y_pred)

        accuracyNN = metrics.accuracy_score(encoded_Y, y_pred)

        acc_tot += accuracy
        error += 1 - accuracy
        pre_mic_tot += precision_micro
        pre_mac_tot += precision_macro
        re_mic_tot += recall_micro
        re_mac_tot += recall_macro
        f1_mic_tot += f1_micro
        f1_mac_tot += f1_macro
        result_mic[0] += result_micro[0]
        result_mic[1] += result_micro[1]
        result_mac[0] += result_macro[0]
        result_mac[1] += result_macro[1]

        if accuracyNN > accuracyTot:
            x_train_high = x_train
            x_test_high = x_test
            y_train_high = y_train
            y_encodedTest_high = encoded_Y
            y_pred_high = y_pred
            accuracyTot = accuracyNN


    print('Accuracy: ', acc_tot / n)

    print('Error: ', error / n)

    print('Precision micro: ', pre_mic_tot / n)

    print('Precision macro: ', pre_mac_tot / n)

    print('Recall micro: ', re_mic_tot / n)

    print('Recall macro: ', re_mac_tot / n)

    print('f1 micro:', f1_mic_tot / n)

    print('f1 macro:', f1_mac_tot / n)

    print('sensitivity micro: ', result_mic[0] / n)
    print('specificity micro: ', result_mic[1] / n)

    print('sensitivity macro: ', result_mac[0] / n)
    print('specificity macro: ', result_mac[1] / n)

    plot_confusion_matrix(y_encodedTest_high, y_pred_high)

'''
def NN_param_selection(X, y):
    model = Sequential()
    model.add(Dense(45, input_dim=49, activation='relu'))
    # output layer
    model.add(Dense(95, activation='sigmoid'))
    neurons = [30, 45, 60, 75, 95]
    epochs = [500, 800, 1000]
    batch_size = [80, 100, 200]
    activation = ['relu', 'tanh', 'sigmoid', 'linear', 'hard_sigmoid']
    param_grid = {'epochs': epochs, 'batch_size': batch_size, "units": neurons, "activation:": activation}
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, scoring='neg_log_loss')
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_
'''

def calcPerformance(y_test, y_pred):  # parametrelere bak!!!!!

    # accuracy: (tp + tn) / (p + n)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    #print('Accuracy: ', accuracy)
    
    error = 1-accuracy
    #print("Error: ",error)
    # precision tp / (tp + fp)
    precision_micro = metrics.precision_score(y_test, y_pred, average='micro', zero_division=1)
    #print('Precision micro: ', precision_micro)

    precision_macro = metrics.precision_score(y_test, y_pred, average='macro', zero_division=1)
    #print('Precision macro: ', precision_macro)

    # recall: tp / (tp + fn)
    recall_micro = metrics.recall_score(y_test, y_pred, average='micro')
    #print('Recall micro: ', recall_micro)

    recall_macro = metrics.recall_score(y_test, y_pred, average='macro')
    #print('Recall macro: ', recall_macro)

    # f1: 2 tp / (2 tp + fp + fn)
    f1_micro = metrics.f1_score(y_test, y_pred, average='micro')
    #print('f1 micro:', f1_micro)

    f1_macro = metrics.f1_score(y_test, y_pred, average='macro')
   # print('f1 macro:', f1_macro)

    result_micro = sensitivity_specificity_support(y_test, y_pred, average='micro')  # support: result[2]
   # print('sensitivity micro: ', result_micro[0])
    #print('specificity micro: ', result_micro[1])

    result_macro = sensitivity_specificity_support(y_test, y_pred, average='macro')  # support: result[2]
   # print('sensitivity macro: ', result_macro[0])
   # print('specificity macro: ', result_macro[1])

    
    '''
    cm = confusion_matrix(y_test, y_pred, labels=y_test)
    print(cm)
    df = pd.DataFrame(
            data=confusion_matrix(y_test, y_pred, labels=y_test),
            columns=y_test,
            index=y_pred
        )
    print(df)
    '''
    
    return accuracy, precision_micro, precision_macro, recall_micro, recall_macro, f1_micro, f1_macro,result_micro,result_macro

def plot_confusion_matrix(y_test, y_pred):
    df_confusion = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=False)
    df_conf_norm = df_confusion / df_confusion.sum(axis=1)
    
 #   plot_cm(df_confusion)
    plot_cm(df_conf_norm)
    
    
def plot_cm(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
    matplotlib.rc('figure', figsize=(100, 10))
    plt.matshow(df_confusion, cmap='Reds') # imshow
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    print("\nDimension of confusion matrix {}".format(df_confusion.shape))
    print("\nPredicted labels:\n {}".format(df_confusion.columns))
    print("\nActual labels: \n{}".format(df_confusion.index))
    # plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    # plt.yticks(tick_marks, df_confusion.index)
    # plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel('Predicted')

    plt.show()


if __name__ == '__main__':
    main()
