# README: Binary classification with QUBO feature reduction


## Brief Description of the Python 3 Project

The project involves the development of an application to classify data contained in a dataset, referred to as "samples" or "records." This is a binary classification, that is, the assignment of a sample to one of two classes, such as "healthy"/"sick", "low risk/high risk", "friend/enemy". The data to be classified consists of a record, composed of a list of fields, which in our case are numeric.
The initial dataset has samples that are already classified; that is, the class of each sample is known. It must be divided into two subsets. One, called the "training set," is used to train the classifier; the other, called the "test set," is used to evaluate the quality of the classifier trained on the training set.
The dataset can be viewed as a matrix, whose rows are the samples, and whose columns are the fields, also called "features"; the known values for each sample are contained in a binary vector (of 0s and 1s), whose i-th element represents the class of the i-th sample.
More specifically, the proposed classification involves determining whether a person applying for a loan is creditworthy or at risk. However, the program developed must simply process a given numerical dataset with a target column consisting of zeros and ones. Each loan application is characterized by a given number of fields (called "features") that describe the applicant, including personal, employment, and financial data and others.

 The output is a binary value: "0" for a reliable applicant, "1" for a high-risk applicant.
If there are too many fields (e.g., the dataset used for testing has 145 features), before setting up the
classifier, the number of features must be reduced using an optimization procedure.
In summary, the steps to be followed—which will be described in detail below—are:

- Reading the dataset for feature reduction and training.

- Converting all record fields to normalized numerical values. Preliminary removal of features (columns) that contain a percentage of null or undefined values exceeding a given threshold (usually around 90–95%).

- Splitting the dataset into a training set and a test set. This is done by treating the first M samples as the training set and the remaining samples as the test set. M is provided as input to the program.

- Determination of which features to eliminate and which to use for classification, using the data from the training set and a QUBO optimization procedure.

- Removal of unselected features from both the training set and the test set.

- Training a classifier on the records of the reduced training dataset.

- Classification on the test set data, comparison with the true values, and printing of the results.

The application must include a user interface to monitor and display the status of the various processing stages, as well as automated tests to verify the system. The algorithms and approaches are not specified, except for the procedure for calculating the cost function for feature reduction.

## Running

To run the project with the GUI use this command:

- `python gui.py`

## Stress Testing

Individual python modules such as `preprocessing.py` and `feature_selection.py` have been stress-tested using a input csv file containing 1.5 million rows.
Note, however, that file was simply the concatentation of 75 repeats of the given sample dataset - so its statistical properties may be anomalous.
Stress tests last several minute when ran on our development platform (Ubuntu on VMware on Windows 10)

## Testing

To run the tests use this command:

- `python -m pytest`

Note, the command must start from the project root directory.
