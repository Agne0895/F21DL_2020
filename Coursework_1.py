'''
F21DL Data Mining and Machine Learning Coursework 1

Team PG_9 members: Akshay Rajieve Krishnan, Agne Zainyte, Jessica Yip
'''

# Importing required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing required datasets using pandas library

df_x_train_full = pd.read_csv("http://www.macs.hw.ac.uk/~ek19/data/x_train_gr_smpl.csv")
df_y_train_full = pd.read_csv("http://www.macs.hw.ac.uk/~ek19/data/y_train_smpl.csv")

df_y_train_0 = pd.read_csv("http://www.macs.hw.ac.uk/~ek19/data/y_train_smpl_0.csv")
df_y_train_1 = pd.read_csv("http://www.macs.hw.ac.uk/~ek19/data/y_train_smpl_1.csv")
df_y_train_2 = pd.read_csv("http://www.macs.hw.ac.uk/~ek19/data/y_train_smpl_2.csv")
df_y_train_3 = pd.read_csv("http://www.macs.hw.ac.uk/~ek19/data/y_train_smpl_3.csv")
df_y_train_4 = pd.read_csv("http://www.macs.hw.ac.uk/~ek19/data/y_train_smpl_4.csv")
df_y_train_5 = pd.read_csv("http://www.macs.hw.ac.uk/~ek19/data/y_train_smpl_5.csv")
df_y_train_6 = pd.read_csv("http://www.macs.hw.ac.uk/~ek19/data/y_train_smpl_6.csv")
df_y_train_7 = pd.read_csv("http://www.macs.hw.ac.uk/~ek19/data/y_train_smpl_7.csv")
df_y_train_8 = pd.read_csv("http://www.macs.hw.ac.uk/~ek19/data/y_train_smpl_8.csv")
df_y_train_9 = pd.read_csv("http://www.macs.hw.ac.uk/~ek19/data/y_train_smpl_9.csv")

print("Reading input files...")
print("Creating dataframes...")

# transforming dataframes into ndarray using numpy library

x_train_full = np.array(df_x_train_full)
y_train_full = np.array(df_y_train_full)

y_train_0 = np.array(df_y_train_0)
y_train_1 = np.array(df_y_train_1)
y_train_2 = np.array(df_y_train_2)
y_train_3 = np.array(df_y_train_3)
y_train_4 = np.array(df_y_train_4)
y_train_5 = np.array(df_y_train_5)
y_train_6 = np.array(df_y_train_6)
y_train_7 = np.array(df_y_train_7)
y_train_8 = np.array(df_y_train_8)
y_train_9 = np.array(df_y_train_9)

print("Transforming dataframes into numeric arrays...")

# setting random seed

np.random.seed(1)

print("Initialising random seed to be 1...")

# normalising dataset

x_train_full = x_train_full.astype('float')/255

print("x_train_full ndarray is being normalised...")


'''
Naive Bayes methods are a set of supervised learning algorithms based on applying Bayes’ theorem 
with the “naive” assumption of conditional independence between every pair of features given the value 
of the class variable.
(Source: scikit-learn.org)
'''

# Naive Bayes classifier using dataset with full features

from sklearn.model_selection import train_test_split

x_train, x_test , y_train, y_test = train_test_split(x_train_full, y_train_full, test_size = 0.33, random_state = 7)

from sklearn.naive_bayes import MultinomialNB

naive_bayes = MultinomialNB()

# training the classifier

naive_bayes.fit(x_train, y_train)
print("Naive Bayes is training...")

# predicting classes using testing dataset

naive_bayes_full_dataset_predict = naive_bayes.predict(x_test)
print("Naive Bayes is classifying testing data...")

# evaluating the classifier

from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report, accuracy_score

naive_bayes_full_dataset_confusion_matrix = confusion_matrix(y_test, naive_bayes_full_dataset_predict)

print("Calculating confusion matrix for full dataset Naive Bayes...\n")
print(naive_bayes_full_dataset_confusion_matrix)

# plotting confusion matrix

titles_options = [("Confusion matrix, without normalization for dataset with all features", None),
                  ("Normalized confusion matrix for dataset with all features", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(naive_bayes, x_test, y_test,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()

naive_bayes_full_dataset_classification_report = classification_report(y_test, naive_bayes_full_dataset_predict)

print("\nClassification report: \n")
print(naive_bayes_full_dataset_classification_report)

naive_bayes_full_dataset_accuracy_score = accuracy_score(y_test, naive_bayes_full_dataset_predict)

print("Accurracy score: \n")
print(naive_bayes_full_dataset_accuracy_score)

print("====================================================================")

'''
Splitting given dataset into 10 separate dataset files for each class to find the correlation between attributes and class value
'''
df_train_0 = pd.concat([df_x_train_full, df_y_train_0], axis=1)
df_train_1 = pd.concat([df_x_train_full, df_y_train_1], axis=1)
df_train_2 = pd.concat([df_x_train_full, df_y_train_2], axis=1)
df_train_3 = pd.concat([df_x_train_full, df_y_train_3], axis=1)
df_train_4 = pd.concat([df_x_train_full, df_y_train_4], axis=1)
df_train_5= pd.concat([df_x_train_full, df_y_train_5], axis=1)
df_train_6 = pd.concat([df_x_train_full, df_y_train_6], axis=1)
df_train_7 = pd.concat([df_x_train_full, df_y_train_7], axis=1)
df_train_8 = pd.concat([df_x_train_full, df_y_train_8], axis=1)
df_train_9 = pd.concat([df_x_train_full, df_y_train_9], axis=1)

# calculating absolute correlation between attributes and class

train_smpl_0 = df_train_0.corr(method = 'pearson').tail(1).transpose().abs().sort_values(by='0',ascending=False).iloc[1:,:]
train_smpl_1 = df_train_1.corr(method = 'pearson').tail(1).transpose().abs().sort_values(by='0',ascending=False).iloc[1:,:]
train_smpl_2 = df_train_2.corr(method = 'pearson').tail(1).transpose().abs().sort_values(by='0',ascending=False).iloc[1:,:]
train_smpl_3 = df_train_3.corr(method = 'pearson').tail(1).transpose().abs().sort_values(by='0',ascending=False).iloc[1:,:]
train_smpl_4 = df_train_4.corr(method = 'pearson').tail(1).transpose().abs().sort_values(by='0',ascending=False).iloc[1:,:]
train_smpl_5 = df_train_5.corr(method = 'pearson').tail(1).transpose().abs().sort_values(by='0',ascending=False).iloc[1:,:]
train_smpl_6 = df_train_6.corr(method = 'pearson').tail(1).transpose().abs().sort_values(by='0',ascending=False).iloc[1:,:]
train_smpl_7 = df_train_7.corr(method = 'pearson').tail(1).transpose().abs().sort_values(by='0',ascending=False).iloc[1:,:]
train_smpl_8 = df_train_8.corr(method = 'pearson').tail(1).transpose().abs().sort_values(by='0',ascending=False).iloc[1:,:]
train_smpl_9 = df_train_9.corr(method = 'pearson').tail(1).transpose().abs().sort_values(by='0',ascending=False).iloc[1:,:]

train_sample5_0 = train_smpl_0.head(5)
train_sample5_1 = train_smpl_1.head(5)
train_sample5_2 = train_smpl_2.head(5)
train_sample5_3 = train_smpl_3.head(5)
train_sample5_4 = train_smpl_4.head(5)
train_sample5_5 = train_smpl_5.head(5)
train_sample5_6 = train_smpl_6.head(5)
train_sample5_7 = train_smpl_7.head(5)
train_sample5_8 = train_smpl_8.head(5)
train_sample5_9 = train_smpl_9.head(5)

train_sample10_0 = train_smpl_0.head(10)
train_sample10_1 = train_smpl_1.head(10)
train_sample10_2 = train_smpl_2.head(10)
train_sample10_3 = train_smpl_3.head(10)
train_sample10_4 = train_smpl_4.head(10)
train_sample10_5 = train_smpl_5.head(10)
train_sample10_6 = train_smpl_6.head(10)
train_sample10_7 = train_smpl_7.head(10)
train_sample10_8 = train_smpl_8.head(10)
train_sample10_9 = train_smpl_9.head(10)

train_sample20_0 = train_smpl_0.head(20)
train_sample20_1 = train_smpl_1.head(20)
train_sample20_2 = train_smpl_2.head(20)
train_sample20_3 = train_smpl_3.head(20)
train_sample20_4 = train_smpl_4.head(20)
train_sample20_5 = train_smpl_5.head(20)
train_sample20_6 = train_smpl_6.head(20)
train_sample20_7 = train_smpl_7.head(20)
train_sample20_8 = train_smpl_8.head(20)
train_sample20_9 = train_smpl_9.head(20)

def remove_duplicates(my_list):
  return list(dict.fromkeys(my_list))

train_labels5 = remove_duplicates(train_sample5_0.T.columns.values.tolist() + train_sample5_1.T.columns.values.tolist()
                                  + train_sample5_2.T.columns.values.tolist() + train_sample5_3.T.columns.values.tolist()
                                  + train_sample5_4.T.columns.values.tolist() + train_sample5_5.T.columns.values.tolist()
                                  + train_sample5_6.T.columns.values.tolist()+ train_sample5_7.T.columns.values.tolist()
                                  + train_sample5_8.T.columns.values.tolist() + train_sample5_9.T.columns.values.tolist())

train_labels10 = remove_duplicates(train_sample10_0.T.columns.values.tolist() + train_sample10_1.T.columns.values.tolist()
                                   + train_sample10_2.T.columns.values.tolist() + train_sample10_3.T.columns.values.tolist()
                                   + train_sample10_4.T.columns.values.tolist()+ train_sample10_5.T.columns.values.tolist()
                                   + train_sample10_6.T.columns.values.tolist()+ train_sample10_7.T.columns.values.tolist()
                                   + train_sample10_8.T.columns.values.tolist() + train_sample10_9.T.columns.values.tolist())

train_labels20 = remove_duplicates(train_sample20_0.T.columns.values.tolist() + train_sample20_1.T.columns.values.tolist()
                                   + train_sample20_2.T.columns.values.tolist() + train_sample20_3.T.columns.values.tolist()
                                   + train_sample20_4.T.columns.values.tolist()+ train_sample20_5.T.columns.values.tolist()
                                   + train_sample20_6.T.columns.values.tolist()+ train_sample20_7.T.columns.values.tolist()
                                   + train_sample20_8.T.columns.values.tolist() + train_sample20_9.T.columns.values.tolist())

# creating dataset with approximately 50 features

train_samples5 = pd.DataFrame([])
for elements in train_labels5:
  index = str(elements)
  train_samples5 = pd.concat((train_samples5, df_x_train_full[index]), axis=1)

print("Creating dataset with approximately 50 features...")
print("The number of features after removing duplicates: {0}".format(train_samples5.shape[1]))

# creating dataset with approximately 100 features

train_samples10 = pd.DataFrame([])
for elements in train_labels10:
  index = str(elements)
  train_samples10 = pd.concat((train_samples10, df_x_train_full[index]), axis=1)

print("Creating dataset with approximately 100 features...")
print("The number of features after removing duplicates: {0}".format(train_samples10.shape[1]))

# creating dataset with approximately 200 features

train_samples20 = pd.DataFrame([])
for elements in train_labels20:
  index = str(elements)
  train_samples20 = pd.concat((train_samples20, df_x_train_full[index]), axis=1)

print("Creating dataset with approximately 200 features...")
print("The number of features after removing duplicates: {0}".format(train_samples20.shape[1]))

# applying Naive Bayes to newly computed datasets

x_train5, x_test5 , y_train5, y_test5 = train_test_split(train_samples5, y_train_full, test_size = 0.33, random_state = 7)

# training the classifier

naive_bayes.fit(x_train5, y_train5)
print("Naive Bayes is training using 5 feature dataset...")

# predicting classes using testing dataset
naive_bayes_5_feature_dataset_predict = naive_bayes.predict(x_test5)
print("Naive Bayes is classifying testing data using 5 feature dataset...")

# evaluating classifier

naive_bayes_5_feature_dataset_confusion_matrix = confusion_matrix(y_test5, naive_bayes_5_feature_dataset_predict)

print("Calculating confusion matrix for 5 feature dataset Naive Bayes...\n")
print(naive_bayes_5_feature_dataset_confusion_matrix)

# plotting confusion matrix

titles_options = [("Confusion matrix, without normalization for dataset with 5 features", None),
                  ("Normalized confusion matrix for dataset with 5 features", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(naive_bayes, x_test5, y_test5,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()

naive_bayes_5_feature_dataset_classification_report = classification_report(y_test5, naive_bayes_5_feature_dataset_predict)

print("\nClassification report for 5 feature dataset: \n")
print(naive_bayes_5_feature_dataset_classification_report)

naive_bayes_5_feature_dataset_accuracy_score = accuracy_score(y_test5, naive_bayes_5_feature_dataset_predict)

print("Accurracy score for 5 feature dataset: \n")
print(naive_bayes_5_feature_dataset_accuracy_score)

print("====================================================================")

x_train10, x_test10 , y_train10, y_test10 = train_test_split(train_samples10, y_train_full, test_size = 0.33, random_state = 7)

# training the classifier

naive_bayes.fit(x_train10, y_train10)
print("Naive Bayes is training using 10 feature dataset...")

# predicting classes using testing dataset
naive_bayes_10_feature_dataset_predict = naive_bayes.predict(x_test10)
print("Naive Bayes is classifying testing 10 feature data...")

# evaluating classifier

naive_bayes_10_feature_dataset_confusion_matrix = confusion_matrix(y_test10, naive_bayes_10_feature_dataset_predict)

print("Calculating confusion matrix for 10 feature dataset Naive Bayes...\n")
print(naive_bayes_10_feature_dataset_confusion_matrix)

# plotting confusion matrix

titles_options = [("Confusion matrix, without normalization for dataset with 10 features", None),
                  ("Normalized confusion matrix for dataset with 10 features", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(naive_bayes, x_test10, y_test10,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()

naive_bayes_10_feature_dataset_classification_report = classification_report(y_test10, naive_bayes_10_feature_dataset_predict)

print("\nClassification report for 10 feature dataset: \n")
print(naive_bayes_10_feature_dataset_classification_report)

naive_bayes_10_feature_dataset_accuracy_score = accuracy_score(y_test10, naive_bayes_10_feature_dataset_predict)

print("Accurracy score for 10 feature dataset: \n")
print(naive_bayes_10_feature_dataset_accuracy_score)

print("====================================================================")

x_train20, x_test20 , y_train20, y_test20 = train_test_split(train_samples20, y_train_full, test_size = 0.33, random_state = 7)

# training the classifier

naive_bayes.fit(x_train20, y_train20)
print("Naive Bayes is training using 20 feature dataset...")

# predicting classes using testing dataset
naive_bayes_20_feature_dataset_predict = naive_bayes.predict(x_test20)
print("Naive Bayes is classifying testing 20 feature data...")

# evaluating classifier

naive_bayes_20_feature_dataset_confusion_matrix = confusion_matrix(y_test20, naive_bayes_20_feature_dataset_predict)

print("Calculating confusion matrix for 20 feature dataset Naive Bayes...\n")
print(naive_bayes_20_feature_dataset_confusion_matrix)

# plotting confusion matrix

titles_options = [("Confusion matrix, without normalization for dataset with 20 features", None),
                  ("Normalized confusion matrix for dataset with 20 features", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(naive_bayes, x_test20, y_test20,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()

naive_bayes_20_feature_dataset_classification_report = classification_report(y_test20, naive_bayes_20_feature_dataset_predict)

print("\nClassification report for 20 feature dataset: \n")
print(naive_bayes_20_feature_dataset_classification_report)

naive_bayes_20_feature_dataset_accuracy_score = accuracy_score(y_test20, naive_bayes_20_feature_dataset_predict)

print("Accurracy score for 20 feature dataset: \n")
print(naive_bayes_20_feature_dataset_accuracy_score)

print("====================================================================")

# Initialising K-Means clustering

from sklearn.cluster import KMeans

# random decision - K from 1 to 30
k = 30

print("Starting clustering using K-Means algorithm...")

mean_accuracy = np.zeros((k - 1))
std_accuracy = np.zeros((k - 1))

for i in range(1, k):
    print("Calculating clusters when k is {0}".format(i))

    # training K-Means
    k_means = KMeans(n_clusters= i, random_state = 5).fit(x_train)

    # testing K-Means
    k_means_predict = k_means.predict(x_test)

    centroid_array = k_means.cluster_centers_

    # evaluation metrics
    mean_accuracy[i - 1] = accuracy_score(y_test, k_means_predict)
    std_accuracy[i - 1] = np.std(k_means_predict == y_test) / np.sqrt(k_means_predict.shape[0])

print("\nThe mean of accuracy is: ")
print(mean_accuracy)
print("\nThe SD of accuracy is: ")
print(std_accuracy)

plt.plot(range(1,k),mean_accuracy,'g')
plt.fill_between(range(1,k),mean_accuracy - 1 * std_accuracy,mean_accuracy + 1 * std_accuracy, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3 std'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbours (K)')
plt.tight_layout()
plt.show()

print("\nThe best accuracy was with", mean_accuracy.max(), "with k=", mean_accuracy.argmax()+1)
print("===================================================================================")

print("Starting clustering using K-Means algorithm with reduced feature set: 5 features...")

mean_accuracy_5 = np.zeros((k - 1))
std_accuracy_5 = np.zeros((k - 1))

for i in range(1, k):
    print("Calculating clusters when k is {0}".format(i))

    # training K-Means
    k_means_5 = KMeans(n_clusters= i, random_state = 5).fit(x_train5)

    # testing K-Means
    k_means_predict_5 = k_means_5.predict(x_test5)

    centroid_array_5 = k_means_5
    # evaluation metrics
    mean_accuracy_5[i - 1] = accuracy_score(y_test5, k_means_predict_5)
    std_accuracy_5[i - 1] = np.std(k_means_predict_5 == y_test5) / np.sqrt(k_means_predict_5.shape[0])

print("\nThe mean of accuracy is: ")
print(mean_accuracy_5)
print("\nThe SD of accuracy is: ")
print(std_accuracy_5)

plt.plot(range(1,k),mean_accuracy,'g')
plt.fill_between(range(1,k),mean_accuracy_5 - 1 * std_accuracy_5,mean_accuracy_5 + 1 * std_accuracy_5, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3 std'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbours (K)')
plt.tight_layout()
plt.show()

print("\nThe best accuracy was with", mean_accuracy_5.max(), "with k=", mean_accuracy_5.argmax()+1)
print("===================================================================================")

print("Starting clustering using K-Means algorithm with reduced feature set: 10 features...")

mean_accuracy_10 = np.zeros((k - 1))
std_accuracy_10 = np.zeros((k - 1))

for i in range(1, k):
    print("Calculating clusters when k is {0}".format(i))

    # training K-Means
    k_means_10 = KMeans(n_clusters= i, random_state = 5).fit(x_train10)

    # testing K-Means
    k_means_predict_10 = k_means_10.predict(x_test10)

    centroid_array_10 = k_means_10.cluster_centers_

    # evaluation metrics
    mean_accuracy_10[i - 1] = accuracy_score(y_test10, k_means_predict_10)
    std_accuracy_10[i - 1] = np.std(k_means_predict_10 == y_test10) / np.sqrt(k_means_predict_10.shape[0])

print("\nThe mean of accuracy is: ")
print(mean_accuracy_10)
print("\nThe SD of accuracy is: ")
print(std_accuracy_10)

plt.plot(range(1,k),mean_accuracy_10,'g')
plt.fill_between(range(1,k),mean_accuracy_10 - 1 * std_accuracy_10,mean_accuracy_10 + 1 * std_accuracy_10, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3 std'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbours (K)')
plt.tight_layout()
plt.show()

print("\nThe best accuracy was with", mean_accuracy_10.max(), "with k=", mean_accuracy_10.argmax()+1)
print("===================================================================================")

print("Starting clustering using K-Means algorithm with reduced features: 20 features...")

mean_accuracy_20 = np.zeros((k - 1))
std_accuracy_20 = np.zeros((k - 1))

for i in range(1, k):
    print("Calculating clusters when k is {0}".format(i))

    # training K-Means
    k_means_20 = KMeans(n_clusters= i, random_state = 5).fit(x_train20)

    # testing K-Means
    k_means_predict_20 = k_means_20.predict(x_test20)

    centroid_array_20 = k_means_20.cluster_centers_

    # evaluation metrics
    mean_accuracy_20[i - 1] = accuracy_score(y_test20, k_means_predict_20)
    std_accuracy_20[i - 1] = np.std(k_means_predict_20 == y_test20) / np.sqrt(k_means_predict_20.shape[0])

print("\nThe mean of accuracy is: ")
print(mean_accuracy_20)
print("\nThe SD of accuracy is: ")
print(std_accuracy_20)

plt.plot(range(1,k),mean_accuracy_20,'g')
plt.fill_between(range(1,k),mean_accuracy_20 - 1 * std_accuracy_20,mean_accuracy_20 + 1 * std_accuracy_20, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3 std'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbours (K)')
plt.tight_layout()
plt.show()

print("\nThe best accuracy was with", mean_accuracy_20.max(), "with k=", mean_accuracy_20.argmax()+1)
print("===================================================================================")

# implementing Gaussain Mixture (EM soft clustering)

from sklearn.mixture import GaussianMixture

mean_accuracy_gmm = np.zeros((k - 1))
std_accuracy_gmm = np.zeros((k - 1))

print("Clustering using Gaussian Mixture Model...")

for i in range(1,30):
    gaussian_mixture_full_dataset = GaussianMixture(n_components=i)
    print("GMM for {0}...".format(i))
    gaussian_mixture_full_dataset.fit(x_train)
    gaussian_mixture_full_dataset_predict = gaussian_mixture_full_dataset.predict(x_test)

    mean_accuracy_gmm[i - 1] = accuracy_score(y_test, gaussian_mixture_full_dataset_predict)
    std_accuracy_gmm[i - 1] = np.std(gaussian_mixture_full_dataset_predict == y_test) / np.sqrt(gaussian_mixture_full_dataset_predict.shape[0])

    # plt.scatter(x_train[:, 0], x_train[:, 1], c=gaussian_mixture_full_dataset_predict, s=40, cmap='viridis');

print("\nThe mean of accuracy is: ")
print(mean_accuracy_gmm)
print("\nThe SD of accuracy is: ")
print(std_accuracy_gmm)

print("\nThe best accuracy was with", mean_accuracy_gmm.max(), "with k=", mean_accuracy_gmm.argmax()+1)

probs_gmm = gaussian_mixture_full_dataset.predict_proba(x_train)
print("Gaussian Model probabilities: \n")
print(probs_gmm[:5].round(3))
print("==================================================================================")





