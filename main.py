import numpy as np
import time
from keras.datasets import mnist
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt


# Function that reads the mnist dataset, normalizes it by dividing it by 255, flattens it from 28*28 arrays to 784
# element 1d arrays and returns the data.
def read_normalize_flatten_mnist():
    # Load the dataset using the applicable keras function.
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Normalize the train and test data by dividing it by 255.
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    # Flatten the train and test data from 28*28 2d arrays to 784 element 1d arrays.
    x_train = x_train.reshape(x_train.shape[0], 784)
    x_test = x_test.reshape(x_test.shape[0], 784)
    # Return the data.
    return x_train, y_train, x_test, y_test


# Function that sets the label of samples with even numbers as labels to 0 and the label of samples with odd numbers
# as labels to 1.
def make_even_odd_mnist(y_train, y_test):
    # Replace the even labels in the train labels with 0 and replace the rest of the train labels with 1.
    y_train = np.where(y_train % 2 == 0, 0, 1)
    # Replace the even labels in the test labels with 0 and replace the rest of the test labels with 1.
    y_test = np.where(y_test % 2 == 0, 0, 1)
    # Return the train and test labels.
    return y_train, y_test


# Function that takes the train data and labels and the test data and labels as input, trains the nearest class centroid
# classifier with the train data and labels and prints the train and test data accuracy of the classifier.
def ncc(x_train, y_train, x_test, y_test):
    # Create the nearest class centroid classifier and train it with the train data and labels.
    ncc_classifier = NearestCentroid()
    print(f"NCC calculations started...")
    ncc_classifier.fit(x_train, y_train)
    # Calculate and print the train and test data accuracies.
    training_accuracy = ncc_classifier.score(x_train, y_train)
    print(f"-Train accuracy: {training_accuracy} ({round(training_accuracy * 100, 2)}%).")
    test_accuracy = ncc_classifier.score(x_test, y_test)
    print(f"-Test accuracy: {test_accuracy} ({round(test_accuracy * 100, 2)}%).\n")


# Function that takes the train data and labels, the test data and labels and the number of neighbors k as input, trains
# the k nearest neighbors classifier with the train data and labels for the k value given as input and prints the train
# and test data accuracy of the classifier.
def knn(x_train, y_train, x_test, y_test, k):
    # Create the k nearest neighbor classifier and train it with the train data and labels.
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    print(f"KNN calculations for k={k} started...")
    knn_classifier.fit(x_train, y_train)
    # Calculate and print the train and test data accuracies.
    training_accuracy = knn_classifier.score(x_train, y_train)
    print(f"-Train accuracy: {training_accuracy} ({round(training_accuracy * 100, 2)}%).")
    test_accuracy = knn_classifier.score(x_test, y_test)
    print(f"-Test accuracy: {test_accuracy} ({round(test_accuracy * 100, 2)}%).\n")


# Function that takes the train data and labels and the test data and labels as input, trains the custom support vector
# classifier with the train data and labels and prints the train and test data accuracy of the classifier as well as the
# time its training lasted in seconds.
def custom_svc(x_train, y_train, x_test, y_test, sample_num):
    # Sample the training set for sample_num samples.
    idx = np.random.permutation(sample_num)
    # Create the support vector classifier and train it with the sampled train data and labels.
    svc_classifier = SVC()
    print(f"SVC calculations for {sample_num} samples started...")
    start_time = time.time()
    svc_classifier.fit(x_train[idx], y_train[idx])
    finish_time = time.time()
    print(f"-Time elapsed for training: {round(finish_time - start_time, 2)} seconds.")
    # Calculate and print the train and test data accuracies.
    training_accuracy = svc_classifier.score(x_train[idx], y_train[idx])
    print(f"-Sampled train accuracy: {training_accuracy} ({round(training_accuracy * 100, 2)}%).")
    test_accuracy = svc_classifier.score(x_test, y_test)
    print(f"-Test accuracy: {test_accuracy} ({round(test_accuracy * 100, 2)}%).")


# Function that runs the code of the second project. Specifically, the first four parameters are used as train and test
# data and labels. Additionally, if any of the parameters ncc_test, knn_test and custom_svc_test is false, then the
# respective classifier's performance is not measured and printed. By default, all three classifiers are tested. Lastly,
# if the parameter sample_num is different from its default value of 60000, then sample_num examples are sampled from
# the training data and labels, which are used to train the custom support vector classifier.
def second_project(x_train, y_train, x_test, y_test, ncc_test=True, knn_test=True, custom_svc_test=True,
                   svc_sample_num=60000):
    # If enabled, measure and print the performance of the nearest class centroid classifier.
    if ncc_test:
        ncc(x_train, y_train, x_test, y_test)

    # If enabled, measure and print the performance of the k nearest neighbor classifier for k=1 and k=3.
    if knn_test:
        for k in [1, 3]:
            knn(x_train, y_train, x_test, y_test, k)

    # If enabled, measure and print the performance and training time of the custom support vector classifier with
    # sample_num sampled training examples of the training data being used to train the classifier.
    if custom_svc_test:
        custom_svc(x_train, y_train, x_test, y_test, svc_sample_num)


# Function that prints and plots the training time, accuracy and test accuracy for the different given max_iter values.
def print_plot_kernel_results(kernel_name, max_iter_values,  training_times, training_accuracies, test_accuracies, ):
    print(f"-Max iterations values: {max_iter_values}")
    print(f"-Training times: {training_times}")
    print(f"-Training accuracies: {training_accuracies}")
    print(f"-Test accuracies: {test_accuracies}")

    plt.figure()
    plt.suptitle(f"{kernel_name.capitalize()} kernel")
    plt.subplot(211)
    plt.plot(max_iter_values, training_accuracies, 'r', max_iter_values, test_accuracies, 'g')
    plt.xlabel("Max_iter")
    plt.ylabel("Accuracy (%)")
    plt.ylim([75, 100])
    plt.legend(["Train", "Test"])

    plt.subplot(212)
    plt.plot(max_iter_values, training_times, 'b')
    plt.xlabel("Max_iter")
    plt.ylabel("Training time (s)")
    plt.ylim([0, 60])
    plt.subplots_adjust(hspace=0.3)
    plt.show()


# Function that prints and plots the training time, accuracy and test accuracy for the different given values
# param_values of the parameter named param_name.
def print_plot_param_results(param_name, param_values, training_times, training_accuracies, test_accuracies, ):
    print(f"-{param_name} values: {param_values}")
    print(f"-Training times: {training_times}")
    print(f"-Training accuracies: {training_accuracies}")
    print(f"-Test accuracies: {test_accuracies}")

    plt.figure()
    plt.subplot(211)
    plt.plot(param_values, training_accuracies, 'r', param_values, test_accuracies, 'g')
    plt.xlabel(param_name)
    plt.ylabel("Accuracy (%)")
    plt.legend(["Train", "Test"])

    plt.subplot(212)
    plt.plot(param_values, training_times, 'b')
    plt.xlabel(param_name)
    plt.ylabel("Training time (s)")
    plt.subplots_adjust(hspace=0.3)
    plt.show()


# Function that calculates the training and test accuracy as well as the training time of the given support vector
# classifier on the given test and train data and appends the results to the corresponding lists passed as parameters.
def append_time_accuracies(x_train, y_train, x_test, y_test, svc_classifier, training_times, training_accuracies,
                           test_accuracies):
    start_time = time.time()
    svc_classifier.fit(x_train, y_train)
    training_times.append(round(time.time() - start_time, 2))
    training_accuracies.append(round(svc_classifier.score(x_train, y_train) * 100, 2))
    test_accuracies.append(round(svc_classifier.score(x_test, y_test) * 100, 2))


# Function that calculates, prints and plots the training time, accuracy and test accuracy of the support vector
# classifier for the given training and test data for each of the kernel type in kernel_values and for the different
# number of max_iter in max_iter_values. As training data, sample_num sampled examples are used from the give training
# samples x_train.
def test_kernels(x_train, y_train, x_test, y_test, sample_num=1000):
    np.random.seed(1)
    svc_classifier = SVC()
    idx = np.random.permutation(sample_num)
    kernel_values = ["linear", "poly", "rbf", "sigmoid"]
    max_iter_values = [10, 50, 100]

    for kernel in kernel_values:
        print(f"\nSVC calculations for {kernel} kernel started...")
        training_times = []
        training_accuracies = []
        test_accuracies = []
        for max_iter in max_iter_values:
            svc_classifier.set_params(kernel=kernel, max_iter=max_iter)
            append_time_accuracies(x_train[idx], y_train[idx], x_test, y_test, svc_classifier, training_times,
                                   training_accuracies, test_accuracies)
        print_plot_kernel_results(kernel, max_iter_values, training_times, training_accuracies, test_accuracies)


# Function that calculates, prints and plots how the training and test accuracy as well as the training time changes as
# the support vector classifier's parameters change, using sample_num sampled examples from the given training samples
# as training data and the given test data. The parameters that are tested are C, gamma and coef_zero with their
# different values saved in C_values, gamma_values and coef_zero_values respectively.
def test_params(x_train, y_train, x_test, y_test, sample_num=1000):
    np.random.seed(1)
    svc_classifier = SVC()
    idx = np.random.permutation(sample_num)
    C_values = [0.001, 0.05, 0.1, 1.0, 1.5]
    gamma_values = [0.0001, "auto", 0.0075, "scale", 0.025]
    coef_zero_values = [0, 0.01, 0.1, 1.0, 10.0]

    if len(C_values) > 0:
        print(f"\nSVC calculations for different C values started...")
        training_times = []
        training_accuracies = []
        test_accuracies = []
        for C in C_values:
            svc_classifier.set_params(C=C)
            append_time_accuracies(x_train[idx], y_train[idx], x_test, y_test, svc_classifier, training_times,
                                   training_accuracies, test_accuracies)
        print_plot_param_results("C", C_values, training_times, training_accuracies, test_accuracies)
        svc_classifier.set_params(C=1.0)

    if len(gamma_values) > 0:
        print(f"\nSVC calculations for different gamma values started...")
        i = 0
        training_times = []
        training_accuracies = []
        test_accuracies = []
        for gamma in gamma_values:
            svc_classifier.set_params(gamma=gamma)
            append_time_accuracies(x_train[idx], y_train[idx], x_test, y_test, svc_classifier, training_times,
                                   training_accuracies, test_accuracies)
            gamma_values[i] = svc_classifier.gamma_val
            i += 1
        print_plot_param_results("Gamma", gamma_values, training_times, training_accuracies, test_accuracies)
        svc_classifier.set_params(gamma="scale")

    if len(coef_zero_values) > 0:
        print(f"\nSVC calculations for different coef_zero values started...")
        training_times = []
        training_accuracies = []
        test_accuracies = []
        for coef_zero in coef_zero_values:
            svc_classifier.set_params(coef_zero=coef_zero)
            append_time_accuracies(x_train[idx], y_train[idx], x_test, y_test, svc_classifier, training_times,
                                   training_accuracies, test_accuracies)
        print_plot_param_results("Coef0", coef_zero_values, training_times, training_accuracies, test_accuracies)
        svc_classifier.set_params(coef_zero=0.0)


# Class that implements the custom support vector classifier.
class SVC:
    # Initializer that takes the support vector's parameters as input and initializes the classifier.
    def __init__(self, C=1.0, kernel="poly", degree=2, gamma="scale", coef_zero=0.0, tol=0.001, eps=0.001, max_iter=25):
        # The regularization parameter, defaults to 1.0.
        self.C = C
        # The type of kernel the classifier will use, defaults to "poly". Possible values are "linear", "rbf", "sigmoid"
        # and "poly". Based on the given string, the respective kernel is used.
        self.kernel = kernel
        # The degree used by the polynomial kernel. Defaults to 2. The other kernels ignore this value. If the given
        # value is not a positive integer, then its default value is used.
        if not (isinstance(degree, int) and degree > 0):
            self.degree = 2
        else:
            self.degree = degree
        # The type of gamma parameter used by the rbf, sigmoid and polynomial kernels. Defaults to "scale".
        self.gamma = gamma
        # The gamma parameter value used by the rbf, sigmoid and polynomial kernels. The other kernels ignore this
        # value.
        self.gamma_val = 0.0
        # The coefficient zero parameter used by the sigmoid and polynomial kernels. Defaults to 0.0. The other kernels
        # ignore this value.
        self.coef_zero = coef_zero
        # The tolerance parameter used by the sequential minimal optimization algorithm. Defaults to 0.001.
        self.tol = tol
        # The error parameter used by the sequential minimal optimization algorithm. Defaults to 0.001.
        self.eps = eps
        # The max number of iterations the classifiers is allowed to be trained for. Defaults to 25 to avoid the
        # training lasting too long.
        self.max_iter = max_iter

        # The training samples and labels which are initialized to none.
        self.x, self.y = None, None
        # The number of samples and the number of their attributes. Both are initialized to none.
        self.sample_num, self.attribute_num = None, None
        # The matrix of the classifier's alpha Lagrange multipliers which is initialized to none.
        self.alphas = None
        # The classifier's bias value (named threshold in the SMO paper) which is initialized to None.
        self.bias = None
        # The matrix of the non-bound samples' (the samples whose Lagrange multipliers are neither 0 nor C) indexes
        # which is initialized to none.
        self.non_bound_indexes = None
        # The label of the positive and negative class. Both are initialized to none.
        self.positive_class_label, self.negative_class_label = None, None

        # The matrix of the classifier's weights which is equal to alphas * labels and is initialized to none.
        self.weights = None
        # The pairwise kernel cache of the classifier which is initialized to none.
        self.kernels = None
        # The batch kernel cache of the classifier which is initialized to none.
        self.kernels_batch = None
        # The error cache of the classifier's non-bound samples as the sequential minimal optimization algorithm details
        # which is initialized to none.
        self.errors = None

    # Method that trains the classifier with the given input data (x) and labels (y). The algorithm used to train the
    # classifier is the sequential minimal optimization.
    def fit(self, x, y):
        # Save the samples and the updated labels, the updated labels are the same as the inputted ones, except the
        # labels of the first two categories are replaced by -1 and 1 respectively.
        self.x, self.y = np.array(x), self.__update_labels(y)
        # Save the number of samples and the number of their attributes.
        self.sample_num, self.attribute_num = self.x.shape
        # Calculate the float gamma value based on the input given for the gamma parameter on the initializer.
        self.__update_gamma()
        # Initialize the matrix of the classifier's alpha Lagrange multipliers to zeroes, one for each sample.
        self.alphas = np.zeros(self.sample_num)
        # Initialize the classifier's bias (threshold) value to zero.
        self.bias = 0
        # Initialize the matrix of the non-bound samples' indexes to an empty list as at the start there are no
        # non-bound samples.
        self.non_bound_indexes = []

        # Initialize the matrix of the classifier's weights to zeroes, one for each sample.
        self.weights = np.zeros(self.sample_num)
        # Initialize the pairwise kernel cache of the classifier to -1.0 for each cached kernel.
        self.kernels = np.array(np.full((self.sample_num, self.sample_num), -1.0))
        # Initialize the batch kernel cache of the classifier to an empty numpy array for each cached kernel.
        self.kernels_batch = [np.array([]) for _ in range(self.sample_num)]
        # Initialize the error cache of the classifier's support vectors to an empty list.
        self.errors = []

        # Perform the calculation of the lagrange multipliers alpha using the sequential minimal optimization algorithm
        # which is implemented by the __smo() method.
        self.__smo()

    # Method that predicts the labels of the given samples x. For each of the samples in x, first the classifier's
    # output is calculated without using the batch kernel cache and then the correct label is assigned to the sample
    # based on the classifier's output for the specific sample.
    def predict(self, x):
        return [self.__get_class(self.__get_output_single(x[i], -1)) for i in range(len(x))]

    # Method that calculates and returns the accuracy of the classifier's predictions for the given samples x.
    def score(self, x, y):
        # Calculate and save the classifier's predicted labels for the given samples x.
        predictions = self.predict(x)
        # Initialize the correct predictions counter to zero.
        correct_predictions = 0

        # For each correct prediction where the predicted label matches the actual label of the sample, increase the
        # correct predictions counter.
        for i in range(len(predictions)):
            if predictions[i] == y[i]:
                correct_predictions += 1
        # Return the number of correct predictions divided by the total number of samples/labels.
        return correct_predictions / len(y)

    # Method that sets the parameters of the support vector classifier. For each parameter, if the value passed as
    # parameter that corresponds to it is not None, then set it to the given value.
    def set_params(self, C=None, kernel=None, degree=None, gamma=None, coef_zero=None, tol=None, eps=None,
                   max_iter=None):
        if C is not None:
            self.C = C
        if kernel is not None:
            self.kernel = kernel
        if degree is not None:
            if not (isinstance(degree, int) and degree > 0):
                self.degree = 2
            else:
                self.degree = degree
        if gamma is not None:
            self.gamma = gamma
        if coef_zero is not None:
            self.coef_zero = coef_zero
        if tol is not None:
            self.tol = tol
        if eps is not None:
            self.eps = eps
        if max_iter is not None:
            self.max_iter = max_iter

    # Method that implements the sequential minimal optimization algorithm presented by Johh C. Platt in this paper:
    # https://www.microsoft.com/en-us/research/publication/sequential-minimal-optimization-a-fast-algorithm-for-training-support-vector-machines/
    # This specific method implements the main routine detailed in the paper. Any numbers in this form (number), refer
    # to the equations in the above paper. Only parts that have been added to the paper's pseudocode will be commented,
    # parts that remained unchanged will not be commented. This goes for the comments in the rest of this class' methods
    # as well.
    def __smo(self):
        # This is a counter for the amount of iterations that have been performed so far and is initialized to zero.
        iteration_counter = 0
        num_changed = 0
        examine_all = 1

        # The first condition of the and operator is the condition used in the paper, the second one is used to stop the
        # algorithm when the number of iterations it has performed has reached the maximum number of iterations it
        # should perform based on the respective parameter max_iter.
        while (num_changed > 0 or examine_all) and (self.max_iter < 0 or iteration_counter < self.max_iter):
            num_changed = 0

            if examine_all:
                # This loops over all the training samples, as the paper's pseudocode details.
                for i in range(self.sample_num):
                    num_changed += self.__examine_example(i)

            else:
                # This loops over all the training samples whose alpha is not 0 or C, ie the non-bound examples, as the
                # paper's pseudocode details.
                for i in self.non_bound_indexes:
                    num_changed += self.__examine_example(i)

            if examine_all == 1:
                examine_all = 0
            elif num_changed == 0:
                examine_all = 1

            # Increment the iteration counter at the end of each iteration.
            iteration_counter += 1

    # Method that implements the procedure examineExample(i2) detailed in the paper.
    def __examine_example(self, i2):
        y2 = self.y[i2]
        alph2 = self.alphas[i2]

        # Load the error for the sample i2 from the error cache or if it isn't cached calculate and return it.
        e2 = self.__calculate_error(i2)
        r2 = e2 * y2

        if (r2 < -self.tol and alph2 < self.C) or (r2 > self.tol and alph2 > 0):

            if len(self.non_bound_indexes) > 1:
                # Select the index i1 based on the heuristic criteria detailed in the paper's 2.2 chapter. This is
                # performed by the __select_heuristic_i1(e2) method which returns the index i1.
                i1 = self.__select_heuristic_i1(e2)
                if self.__take_step(i1, i2, e2):
                    return 1

            # The paper suggests looping over all non-bound samples' indexes, starting at a random point. The
            # __get_shuffled_list(original_list) implements this. The original list which is the non-bound samples'
            # indexes gets randomly shuffled and is returned.
            shuffled_non_bound_indexes = self.__get_shuffled_list(self.non_bound_indexes)
            # Loop over the shuffled non-bound samples' indexes, as the paper details.
            for i1 in shuffled_non_bound_indexes:
                if self.__take_step(i1, i2, e2):
                    return 1

            # The paper suggests looping over all training sample indexes, starting at a random point. The
            # __get_shuffled_list(original_list) implements this.The original list which is the training sample indexes
            # gets randomly shuffled and is returned.
            shuffled_sample_indexes = self.__get_shuffled_list([*range(self.sample_num)])
            # Loop over the shuffled training sample indexes, as the paper details.
            for i1 in shuffled_sample_indexes:
                if self.__take_step(i1, i2, e2):
                    return 1
        return 0

    # Method that implements the procedure takeStep(i1, i2) detailed in the paper.
    def __take_step(self, i1, i2, e2):
        if i1 == i2:
            return 0

        alph1 = self.alphas[i1]
        y1 = self.y[i1]
        alph2 = self.alphas[i2]
        y2 = self.y[i2]

        # Load the error for the sample i1 from the error cache or if it isn't cached calculate and return it.
        e1 = self.__calculate_error(i1)
        s = y1 * y2

        # Calculate l and h via the paper's equations (13) and (14). The __calculate_l_h(i1, i2) method implements this.
        l, h = self.__calculate_l_h(i1, i2)
        if l == h:
            return 0

        # Load the pairwise kernels needed from the pairwise kernel cache or if any of them are not cached, calculate,
        # cache them and then return them.
        k11 = self.__get_kernel_pairwise(i1, i1)
        k12 = self.__get_kernel_pairwise(i1, i2)
        k22 = self.__get_kernel_pairwise(i2, i2)
        eta = k11 + k22 - 2 * k12

        if eta > 0:
            a2 = alph2 + y2 * (e1 - e2) / eta
            if a2 < l:
                a2 = l
            elif a2 > h:
                a2 = h

        else:
            # Calculate the objective function at l and h via the paper's equations listed at (19). The
            # __calculate_l_h_objective(i1, i2, e1, e2, l, h, k11, k12, k22, s) method implements this.
            l_obj, h_obj = self.__calculate_l_h_objective(i1, i2, e1, e2, l, h, k11, k12, k22, s)
            if l_obj < h_obj - self.eps:
                a2 = l
            elif l_obj > h_obj + self.eps:
                a2 = h
            else:
                a2 = alph2

        if np.abs(a2 - alph2) < self.eps * (a2 + alph2 + self.eps):
            return 0

        a1 = alph1 + s * (alph2 - a2)
        # The next two lines calculate numbers that will be used twice each so that it saves calculation time.
        y1_mul_a1_diff = y1 * (a1 - alph1)
        y2_mul_a2_diff = y2 * (a2 - alph2)
        # Calculate the bias intermediate values b1 and b2 using the paper's equations (20) and (21).
        b1 = e1 + y1_mul_a1_diff * k11 + y2_mul_a2_diff * k12 + self.bias
        b2 = e2 + y1_mul_a1_diff * k12 + y2_mul_a2_diff * k22 + self.bias

        # Update the classifier's bias based on the paper's chapter 2.3.
        if 0 < a1 < self.C:
            self.bias = b1
        elif 0 < a2 < self.C:
            self.bias = b2
        else:
            self.bias = (b1 + b2) / 2

        # Store the new alpha values in the classifier's alpha array.
        self.alphas[i1] = a1
        self.alphas[i2] = a2

        # Update the non-bound indexes based on the changes in the alphas of the samples with indexes i1 and i2.
        self.__update_non_bound_indexes(alph1, self.alphas[i1], i1)
        self.__update_non_bound_indexes(alph2, self.alphas[i2], i2)

        # If either of the alphas changed, update the matrix of the classifier's weights as y * alphas.
        if self.alphas[i1] != alph1 or self.alphas[i2] != alph2:
            self.weights = self.y * self.alphas

        # Update the error cache as the classifier's output - target output y for each of the non-bound samples.
        self.errors = np.array(self.__get_output_batch(self.x[self.non_bound_indexes])) - self.y[self.non_bound_indexes]

        return 1

    # Method that edits the given labels so that the two classes in the labels are replaced by -1 and 1 respectively.
    def __update_labels(self, y):
        # The label of the negative class becomes the first label, while the label of the positive class becomes the
        # first label that is different from the negative class' label.
        self.negative_class_label = y[0]
        i = 0
        while y[i] == self.negative_class_label:
            i += 1
        self.positive_class_label = y[i]
        # The positive labels are replaced by 1, while the negative ones are replaced by -1.
        return np.where(y == self.positive_class_label, 1, -1)

    # Method that calculates and saves the gamma value based on the gamma type. The value of gamma is calculated based
    # on the formula provided here: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    def __update_gamma(self):
        # If the gamma type is not a non-negative integer or float, then calculate its value.
        if not ((isinstance(self.gamma, int) or isinstance(self.gamma, float)) and self.gamma >= 0):
            # If the gamma type is "auto", then gamma_val = 1 / number of the training
            # samples' attributes.
            if self.gamma == "auto":
                self.gamma_val = 1 / self.attribute_num
            # Else, gamma_val = 1 / (number of the training samples' attributes * variance of the training samples).
            else:
                self.gamma_val = 1 / (self.attribute_num * self.x.var())
        # Else if the gamma type is already a non-negative integer or float, save its value to gamma_val.
        else:
            self.gamma_val = float(self.gamma)

    # Method that loads the error for the training sample with index i from the error cache or calculates and returns
    # it.
    def __calculate_error(self, i):
        # Look for the index of the i-th training sample's error in the error cache.
        error_indexes = np.where(self.non_bound_indexes == i)[0]
        # If the index exists, ie the returned list is not empty, then return its respective error.
        if len(error_indexes) >= 1:
            return self.errors[error_indexes[0]]
        # Else if the index does not exist, calculate and return the error of the i-th training sample as the
        # classifier's output for the i-th training sample - the target value (label) for that training sample.
        else:
            return self.__get_output_single(self.x[i], i) - self.y[i]

    # Method that selects the i1 index by following the heuristic criteria detailed in the paper's chapter 2.2. The
    # index i1 that is returned, is the non-bound sample's index which maximizes the value |e1 - e2|, where e1 is the
    # error of the i1 index which is saved in the error cache.
    def __select_heuristic_i1(self, e2):
        return self.non_bound_indexes[np.argmax(np.abs(self.errors - e2))]

    # Method that randomly shuffles a copy of the original list and returns it.
    def __get_shuffled_list(self, original_list):
        # Make a copy of the original list.
        shuffled_list = list(original_list)
        # Shuffle the copy of the original list.
        np.random.shuffle(shuffled_list)
        # Return the shuffled copy of the original list.
        return shuffled_list

    # Method that calculates and returns the l and h values based on the paper's equations (13) and (14).
    def __calculate_l_h(self, i1, i2):
        # If y1 != y2, use the equations (13) to calculate and return l and h.
        if self.y[i1] != self.y[i2]:
            return max(0, self.alphas[i2] - self.alphas[i1]), min(self.C, self.C + self.alphas[i2] - self.alphas[i1])
        # Else use the equations (14) to calculate and return l and h.
        return max(0, self.alphas[i2] + self.alphas[i1] - self.C), min(self.C, self.alphas[i2] + self.alphas[i1])

    # Method that loads the pairwise kernel of the samples i1 and i2 from the pairwise kernel cache after first
    # calculating and caching it if it has not already been cached.
    def __get_kernel_pairwise(self, i1, i2):
        # If the pairwise kernel is not saved in the cache for the pair i1 and i2, then calculate it and save it in the
        # cache.
        if self.kernels[i1][i2] == -1:
            self.kernels[i1][i2] = self.__calculate_kernel(self.x[i1], self.x[i2])
        # Return the pairwise kernel for the pair i1 and i2 saved in the cache.
        return self.kernels[i1][i2]

    # Method that calculates and returns the objective function at l and h based on the paper's equations (19).
    def __calculate_l_h_objective(self, i1, i2, e1, e2, l, h, k11, k12, k22, s):
        f1 = self.y[i1] * (e1 + self.bias) - self.alphas[i1] * k11 - s * self.alphas[i2] * k12
        f2 = self.y[i2] * (e2 + self.bias) - s * self.alphas[i1] * k12 - self.alphas[i2] * k22
        l1 = self.alphas[i1] + s * (self.alphas[i2] - l)
        h1 = self.alphas[i1] + s * (self.alphas[i2] - h)
        l_obj = l1 * f1 + l * f2 + 0.5 * (l1 ** 2) * k11 + 0.5 * (l ** 2) * k22 + s * l * l1 * k12
        h_obj = h1 * f1 + h * f2 + 0.5 * (h1 ** 2) * k11 + 0.5 * (h ** 2) * k22 + s * h * h1 * k12
        return l_obj, h_obj

    # Method that updates the non-bound indexes based on the changes in the alpha of the sample with index i.
    def __update_non_bound_indexes(self, alpha_old, alpha_new, i):
        # If the previous alpha was non-bound and the new alpha is bound, remove the index i from the list of non-bound
        # indexes.
        if 0 < alpha_old < self.C:
            if not (0 < alpha_new < self.C):
                self.non_bound_indexes.remove(i)
        # Else if the previous alpha was bound and the new alpha is non-bound, append the index i to the list of
        # non-bound indexes.
        else:
            if 0 < alpha_new < self.C:
                self.non_bound_indexes.append(i)

    # Method that calculates and returns the classifier's output for each of the samples in x.
    def __get_output_batch(self, x):
        return [self.__get_output_single(x[i], self.non_bound_indexes[i]) for i in range(len(x))]

    # Method that calculates and returns the classifier's output for the sample x. It is calculated using the paper's
    # equation (10), where weights = y * alphas, __get_kernel_batch(self.x, x, i) returns the batch kernel for the x
    # sample and b is the bias (threshold) of the classifier.
    def __get_output_single(self, x, i):
        return np.sum(np.dot(self.weights, self.__get_kernel_batch(self.x, x, i))) - self.bias

    # Method that loads the batch kernel of the sample xi with index i from the batch kernel cache or calculates,
    # caches if needed and returns the batch kernel of the sample xi with index i.
    def __get_kernel_batch(self, x, xi, i):
        # If the index i of the sample is not within the indexes of the cached batch kernels, then calculate and return
        # the batch kernel.
        if i >= self.sample_num or i < 0:
            return self.__calculate_kernel(x, xi)

        # Else if the length of the cached batch kernel is zero, meaning that no kernel has been cached yet, calculate
        # the batch kernel and cache it.
        elif len(self.kernels_batch[i]) == 0:
            self.kernels_batch[i] = self.__calculate_kernel(x, xi)

        # Return the cached kernel since at this point it was already cached earlier.
        return self.kernels_batch[i]

    # Method that returns the correct class label based on the classifier's output. If the given output is non-negative,
    # the positive class' label is returned, else the negative class' label is returned.
    def __get_class(self, output):
        if output >= 0:
            return self.positive_class_label
        return self.negative_class_label

    # Method that calculates and returns the kernel between x1 which can be a single sample or a batch of samples and
    # x2 which is always a single sample. Note that the parameters gamma_val, coef_zero and degree have already had
    # their values set correctly either in the initializer, the set_params or the fit method.
    def __calculate_kernel(self, x1, x2):

        # If the selected kernel is the linear kernel, calculate and return the linear kernel using the formula detailed
        # here: https://scikit-learn.org/stable/modules/svm.html#kernel-functions, for the linear kernel.
        if self.kernel == "linear":
            return np.dot(x1, x2.T)

        # Else if the selected kernel is the rbf kernel, calculate and return the rbf kernel using the formula detailed
        # here: https://scikit-learn.org/stable/modules/svm.html#kernel-functions, for the rbf kernel.
        elif self.kernel == "rbf":
            # If x1 has more than one dimensions, meaning that it's a batch of samples, set the axis for the L2 norm
            # calculation to the second axis.
            if x1.ndim >= 2:
                axis = 1
            # Else if x1 is one dimensional, meaning that it's a single sample, set the axis for the L2 norm calculation
            # to None.
            else:
                axis = None
            # Calculate and return the rbf kernel with the L2 norm being calculated on the applicable axis.
            return np.exp(-self.gamma_val * (np.linalg.norm(x1 - x2, axis=axis) ** 2))

        # Else if the selected kernel is the sigmoid kernel, calculate and return the sigmoid kernel using the formula
        # detailed here: https://scikit-learn.org/stable/modules/svm.html#kernel-functions, for the sigmoid kernel.
        elif self.kernel == "sigmoid":
            return np.tanh(self.gamma_val * np.dot(x1, x2.T) + self.coef_zero)

        # Else if the selected kernel is the polynomial kernel, calculate and return the polynomial kernel using the
        # formula detailed here: https://scikit-learn.org/stable/modules/svm.html#kernel-functions, for the polynomial
        # kernel.
        else:
            return (self.gamma_val * np.dot(x1, x2.T) + self.coef_zero) ** self.degree


# Read the normalized flattened mnist dataset.
X_train, Y_train, X_test, Y_test = read_normalize_flatten_mnist()
# Edit the labels of the normalized flattened mnist dataset so that they split the dataset into even and odd numbers.
Y_train, Y_test = make_even_odd_mnist(Y_train, Y_test)
# Run the second project's code with the three classifiers as described in the function's comments.
second_project(X_train, Y_train, X_test, Y_test, ncc_test=True, knn_test=True, custom_svc_test=True,
               svc_sample_num=5000)

# Test the different types of kernels for the SVC and print and plot the results, is commented out by default.
# test_kernels(X_train, Y_train, X_test, Y_test)
# Test different values for the SVC parameters C, gamma and coef_zero and print and plot the results, is commented out
# by default.
# test_params(X_train, Y_train, X_test, Y_test)
