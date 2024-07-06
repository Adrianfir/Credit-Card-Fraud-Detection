# Fraud-Detection

This repository is dedicated for fraud detection on CreditCard dataset. 

To find the data set used in this project use the following link:

https://www.kaggle.com/datasets/kartik2112/fraud-detection


============================================================================================

Important Note: In tasks related to fraud-detection, the dataset is usually imbalanced, meaning that the emphasize of the machine (deep) learning model would be more on the class with more rows of data in the dataset. To handle this, two approach can be used:

1 - Undersampling: In this method, a number of rows of data related to the class with higher number of rows in the dataset(let's say class A). However, the important point to be considered is that each row of data is vital to be used in modeling the data.

	* Tomec Undersampling: A method in undersampling that it drops some of the observations of class A whic are very close to the observation of the class B. So the drops from class A are not randomly anoymore.

2 - Oversampling: In this approach, the number of data related to the class B would be increase. One way is to add duplicates of the existing data to the datset. It means if D1 has lable B, we add (for example) 3  D1 to the datset as it happens to be recallected for 3 times during gathering the data. infact, we are keeping the dublicates of the data related to class B. However, this approach is not that usfull as it does not add new values to the model during training. 

	* SMOTE (Synthetic Monitoring Oversampling Technique): This method is quite popular. In this approach, Each data from class B would find their nearest neighbors in the same class. Hence, it is safe to add a point in the middle of them. 

	There are many variants of SMOTE, sncluding Boarderlin-SMOTE, KNN-SMOTE, SVM-SMOTE, and many other variants. Some of them are interative by nature, meaning that they would keep doing the process after generating new data.

	There is an input to stop the generation process, so we can have even 50-50 data. however instead we can move from much less representation of class B to less representaation of class B. (for example 30 or 40 or ...).

	However, there is a drawback in SMOTE which is making the data in class B looks like located in lines.


	* ADASYN (Adaptive Synthetic Oversampling Technique): It follows the fundamental of SMOTE. Again in an iterative fashion. it generates data for class B "mostly" according to the points that are so close to class A. 



	Note: to apply the explained process the whole dataset (including train-set and test-set concatinated and formed "df") should be considered. The aforementioned link has both train-set and test-set. So, before anything, we have concatinated them to form df and then apply pre-processing, including up-sampling or down-sampling.
	 


============================================================================================
## Results:

Although the results (e.g., f1-score) are slightly better using up-sampling, using down-samplign would bring about much less training time with slightly less accuracy and scores.