# CS632
Introduction to Deep Learning

#Please when running the scripts try first $python myKnn.py -h (Part 1), where -h stands for help, a menu will appear guiding you for all possible options of running the script.
$python parser.py -h (part 2) for options available for sencond part of the assignment.

Part 1 Conseptual Part

1) For knn classification normalizing the scale of the features is of utmost importance due to the nature of the algorithm in action. Knn classification is performed over a group of examples by measuring the distance in between them, thus the scale of the features impact the measurement of the distances hence the classification of hte instances themselves.

2)The difference of numerical and categorical features is that numerical as the name implies are numerical values and include measurements as counts, percentages or numbers while categorical features are descriptions of groups or elements and are string values. For knn classification, categorical variables have to be converted to numerical using various trasformation techniques as LabelEncoder for labels, or single words and BagOfWords or WOrd2Vec or Glove for sentences.

3)The importance of testing data is for validating a trained model by measuring how accurate it can be on unseen data. Due to the fact the training data either is memorized (Lazy Learners) or a function is learned out of the instances contained in the training dataset, without the testing data there would be no other way of measuring the accuracy of the model unless we try it on unseen data.

4)Supervised refers to the methodology followed to train our model, since the model is trained by seeing the classes assosiated with each instance it is as the model is trained in a supervised manner.

5)One additional feature that I would include in the iris dataset would have been the height of the plant, and different variations in the colours each plant species can take.

Part 2 Conseptual Part

1)One drawback for using BagOfWords is the fact that the only feature that derives out of the transformation is the frequency of the words within a sentence, most of the times an weak feature in predicting power. Also there is no relationship between the words in a sentence, something can be derived from a Glove model. One of the biggest strengths however is the simplicity of applying it to a dateset and get started without manually trying to find the predictive power of individual words.

2)
