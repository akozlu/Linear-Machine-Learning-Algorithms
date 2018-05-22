# Linear-Machine-Learning-Algorithms
Implementation of Perceptron, Winnow and Adagrad-Perceptron, along with their averaged versions on synthetic and real world datasets 

# Datasets 

1.  Synthetic Sparse data:  training, development, test.
2.  Synthetic Dense data:  training, development, test.
3.  Real data set:  Training = CoNLL; Dvelopment = CoNLL and Enron; Test:  CoNLLand Enron.
# Running the Script
If you run the main function without doing any changes, you will get the following:

1. News Dev Accuracy For Averaged Perceptron
2. Email Dev Accuracy for Averaged Perceptron 
3. News Dev Accuracy for SVM 
4. Email Dev Accuracy for SVM
5. Accuracies of 7 seven models for sparse and synthetic development data. (14 train - 14 test accuracies are printed) 

*Each algorithm runs for 10 iterations. This is because averaged versions require some catch up with their normal versions.I talk about this in my report.*

## Implementation Details:

The implementation of Adagrad, Winnow and Perceptron are all under Classifier class. 

If averaged variable is True, average versions of the algorithms are implemented. 


## Other Functions 
|  Functions  | Usage|
| ------------- | ------------- |
| Plottng Learning Curves of 7 models | Uncomment lines 944-945 |
| Hyperparameter Tuning for Winnow | Uncomment line 937 |
| Hyperparameter Tuning for Adagrad | Uncomment line 936 |

The pdf file includes all the experiments I did with seven linear learning algorithms.

Thank you for reading, please let me know if anyone has problems running the script. 
