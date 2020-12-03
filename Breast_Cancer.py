from sklearn.datasets import load_breast_cancer
breast_cancer_data = load_breast_cancer()

from sklearn.model_selection import train_test_split
training_data, validation_data, training_labels, validation_labels = train_test_split(breast_cancer_data.data, breast_cancer_data.target, test_size = 0.2, random_state = 100)

from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt

l = []

for k in range(1, 101):
    classifier = KNeighborsClassifier(n_neighbors = k)
    classifier.fit(training_data, training_labels)
    accuracy = classifier.score(validation_data, validation_labels)
    l.append(accuracy)
    
best_accuracy = max(l)
best_k = l.index(best_accuracy) + 1

print(best_k)

k_list = []

for i in range(1, 101):
    k_list.append(i)
    
plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.title("Breast Cancer Classifier Accuracy")  
  
plt.plot(k_list, l)
plt.show()

