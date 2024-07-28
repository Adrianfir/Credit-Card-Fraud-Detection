# Fraud-Detection

This repository is dedicated to fraud detection on the CreditCard dataset. 

To find the data set used in this project use the following link:

https://www.kaggle.com/datasets/kartik2112/fraud-detection


============================================================================================

Important Note: In tasks related to fraud detection, the dataset is usually imbalanced, meaning that the emphasis of the machine (deep) learning model would be more on the class with more rows of data in the dataset. To handle this, two approaches can be used:

1 - Undersampling: In this method, a number of rows of data related to the class with the higher number of rows in the dataset(let's say class A). However, the important point to be considered is that each row of data is vital to modelling the data.

	* Tomec Undersampling: A method of undersampling that drops some of the observations of class A which are very close to the observations of class B. So the drops from class A are not random anymore.

2 - Oversampling: In this approach, the amount of data related to class B will be increased. One way is to add duplicates of the existing data to the data set. This means that if D1 has lable B, we add (for example) 3 D1 to the dataset as it happens to be recollected 3 times during the data gathering. In fact, we are keeping the duplicates of the data related to class B. However, this approach is not that usfull as it does not add new values to the model during training. 

	* SMOTE (Synthetic Monitoring Oversampling Technique): This method is quite popular. In this approach, Each data from class B would find their nearest neighbors in the same class. Hence, it is safe to add a point in the middle of them. 

	There are many variants of SMOTE, including Boarderlin-SMOTE, KNN-SMOTE, SVM-SMOTE, and many other variants. Some of them are iterative by nature, meaning that they would keep doing the process after generating new data.

	There is an input to stop the generation process, so we can have even 50-50 data. However, instead we can move from much less representation of class B to less representation of class B. (for example 30 or 40 or ...).

	However, there is a drawback in SMOTE which is making the data in class B look like located in lines.


	* ADASYN (Adaptive Synthetic Oversampling Technique): It follows the fundamentals of SMOTE. Again in an iterative fashion. it generates data for class B "mostly" according to the points that are so close to class A. 



	Note: to apply the explained process the whole dataset (including the train-set and test-set concatenated and formed "df") should be considered. The aforementioned link has both a train-set and a test-set. So, before anything, we have concatenated them to form df and then apply pre-processing, including up-sampling or down-sampling.
	 


============================================================================================
## Results:

Although the results (e.g., f1-score) are slightly better using up-sampling, using down-samplign would bring about much less training time with slightly less accuracy and scores.
