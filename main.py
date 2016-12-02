from sklearn.neighbors import KNeighborsClassifier
from dataset_loader import DatasetLoader
import numpy as np
import random
import json
import os

NUMBER_OF_EXECUTIONS = 30
K_VARIATION = 6


def datasets():
    return os.listdir('./datasets/')


def save_results(dataset, accuracies):
    data = {
        'accuracies': accuracies,
        'average': np.mean(accuracies),
        'standard deviation': np.std(accuracies),
        'number of executions': len(accuracies)
    }
    if not os.path.exists('./results/'):
        os.makedirs('./results/')
    with open('./results/' + dataset + '.json', 'w') as file:
        json.dump(data, file, sort_keys=True, indent=4)


def split(tr_examples_ori, tr_labels_ori, split_size=0.67):
        tr_examples = np.array(tr_examples_ori)
        tr_labels = np.array(tr_labels_ori)
        training_examples = []
        training_labels = []
        test_examples = []
        test_labels = []
        test_size = int(split_size * len(tr_labels))

        test_indexes = random.sample(xrange(len(tr_labels)), test_size)

        test_examples = np.array(tr_examples[test_indexes])
        test_labels = np.array(tr_labels[test_indexes])
        for index in xrange(len(tr_labels)):
            if index not in test_indexes:
                training_examples.append(tr_examples[index])
                training_labels.append(tr_labels[index])
        training_examples = np.array(training_examples)
        training_labels = np.array(training_labels)
        return training_examples, training_labels, test_examples, test_labels


def accuracy(model, test_examples, test_labels):
    return model.score(test_examples, test_labels)


for dataset in datasets():
    loader = DatasetLoader('./datasets/' + dataset)
    accuracies = []
    for k in xrange(1, K_VARIATION + 1, 1):
        for _ in xrange(NUMBER_OF_EXECUTIONS):
            train_examples, train_labels, test_examples, test_labels = split(
                loader.examples, loader.labels
            )
            classifier = KNeighborsClassifier(n_neighbors=k)
            classifier.fit(train_examples, train_labels)
            accuracies.append(accuracy(classifier, test_examples, test_labels))
    save_results(dataset, accuracies)
