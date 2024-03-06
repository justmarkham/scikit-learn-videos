English

# Introduction to Machine Learning with scikit-learn

This video series will teach you how to solve Machine Learning problems using Python's popular scikit-learn library. There are **10 video tutorials** totaling 4.5 hours, each with a corresponding **Jupyter notebook**.

You can [watch the entire series](https://www.youtube.com/playlist?list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A) on YouTube and [view all of the notebooks](http://nbviewer.jupyter.org/github/justmarkham/scikit-learn-videos/tree/master/) using nbviewer.

The series is also available as a [free online course](https://courses.dataschool.io/introduction-to-machine-learning-with-scikit-learn) that includes updated content, quizzes, and a certificate of completion.

[![Watch the first tutorial video](images/youtube.png)](https://www.youtube.com/watch?v=elojMnjn4kk&list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A&index=1 "Watch the first tutorial video")

**Note:** The notebooks in this repository have been updated to use Python 3.9.1 and scikit-learn 0.23.2. The original notebooks (shown in the video) used Python 2.7 and scikit-learn 0.16, and can be downloaded from the [archive branch](https://github.com/justmarkham/scikit-learn-videos/tree/archive). You can read about how I updated the code in this [blog post](https://www.dataschool.io/how-to-update-your-scikit-learn-code-for-2018/).

## Table of Contents

1. What is Machine Learning, and how does it work? ([video](https://www.youtube.com/watch?v=elojMnjn4kk&list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A&index=1), [notebook](01_machine_learning_intro.ipynb))
    - What is Machine Learning?
    - What are the two main categories of Machine Learning?
    - What are some examples of Machine Learning?
    - How does Machine Learning "work"?

2. Setting up Python for Machine Learning: scikit-learn and Jupyter Notebook ([video](https://www.youtube.com/watch?v=IsXXlYVBt1M&list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A&index=2), [notebook](02_machine_learning_setup.ipynb))
    - What are the benefits and drawbacks of scikit-learn?
    - How do I install scikit-learn?
    - How do I use the Jupyter Notebook?
    - What are some good resources for learning Python?

3. Getting started in scikit-learn with the famous iris dataset ([video](https://www.youtube.com/watch?v=hd1W4CyPX58&list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A&index=3), [notebook](03_getting_started_with_iris.ipynb))
    - What is the famous iris dataset, and how does it relate to Machine Learning?
    - How do we load the iris dataset into scikit-learn?
    - How do we describe a dataset using Machine Learning terminology?
    - What are scikit-learn's four key requirements for working with data?

4. Training a Machine Learning model with scikit-learn ([video](https://www.youtube.com/watch?v=RlQuVL6-qe8&list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A&index=4), [notebook](04_model_training.ipynb))
    - What is the K-nearest neighbors classification model?
    - What are the four steps for model training and prediction in scikit-learn?
    - How can I apply this pattern to other Machine Learning models?

5. Comparing Machine Learning models in scikit-learn ([video](https://www.youtube.com/watch?v=0pP4EwWJgIU&list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A&index=5), [notebook](05_model_evaluation.ipynb))
    - How do I choose which model to use for my supervised learning task?
    - How do I choose the best tuning parameters for that model?
    - How do I estimate the likely performance of my model on out-of-sample data?

6. Data science pipeline: pandas, seaborn, scikit-learn ([video](https://www.youtube.com/watch?v=3ZWuPVWq7p4&list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A&index=6), [notebook](06_linear_regression.ipynb))
    - How do I use the pandas library to read data into Python?
    - How do I use the seaborn library to visualize data?
    - What is linear regression, and how does it work?
    - How do I train and interpret a linear regression model in scikit-learn?
    - What are some evaluation metrics for regression problems?
    - How do I choose which features to include in my model?

7. Cross-validation for parameter tuning, model selection, and feature selection ([video](https://www.youtube.com/watch?v=6dbrR-WymjI&list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A&index=7), [notebook](07_cross_validation.ipynb))
    - What is the drawback of using the train/test split procedure for model evaluation?
    - How does K-fold cross-validation overcome this limitation?
    - How can cross-validation be used for selecting tuning parameters, choosing between models, and selecting features?
    - What are some possible improvements to cross-validation?

8. Efficiently searching for optimal tuning parameters ([video](https://www.youtube.com/watch?v=Gol_qOgRqfA&list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A&index=8), [notebook](08_grid_search.ipynb))
    - How can K-fold cross-validation be used to search for an optimal tuning parameter?
    - How can this process be made more efficient?
    - How do you search for multiple tuning parameters at once?
    - What do you do with those tuning parameters before making real predictions?
    - How can the computational expense of this process be reduced?

9. Evaluating a classification model ([video](https://www.youtube.com/watch?v=85dtiMz9tSo&list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A&index=9), [notebook](09_classification_metrics.ipynb))
    - What is the purpose of model evaluation, and what are some common evaluation procedures?
    - What is the usage of classification accuracy, and what are its limitations?
    - How does a confusion matrix describe the performance of a classifier?
    - What metrics can be computed from a confusion matrix?
    - How can you adjust classifier performance by changing the classification threshold?
    - What is the purpose of an ROC curve?
    - How does Area Under the Curve (AUC) differ from classification accuracy?

10. Building a Machine Learning workflow ([video](https://www.youtube.com/watch?v=irHhDMbw3xo&list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A&index=10), [notebook](10_categorical_features.ipynb))
    - Why should you use a Pipeline?
    - How do you encode categorical features with OneHotEncoder?
    - How do you apply OneHotEncoder to selected columns with ColumnTransformer?
    - How do you build and cross-validate a Pipeline?
    - How do you make predictions on new data using a Pipeline?
    - Why should you use scikit-learn (rather than pandas) for preprocessing?

## Bonus Video

At the PyCon 2016 conference, I taught a **3-hour tutorial** that builds upon this video series and focuses on **text-based data**. You can watch the [tutorial video](https://www.youtube.com/watch?v=ZiKMIuYidY0&list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A&index=11) on YouTube.

Here are the topics I covered:

1. Model building in scikit-learn (refresher)
2. Representing text as numerical data
3. Reading a text-based dataset into pandas
4. Vectorizing our dataset
5. Building and evaluating a model
6. Comparing models
7. Examining a model for further insight
8. Practicing this workflow on another dataset
9. Tuning the vectorizer (discussion)

Visit this [GitHub repository](https://github.com/justmarkham/pycon-2016-tutorial) to access the tutorial notebooks and many other recommended resources.

---


Nederlands

# Inleiding tot machine learning met scikit-learn

Deze videoserie leert je hoe je Machine Learning-problemen oplost met behulp van de populaire scikit-learn-bibliotheek van Python. Er zijn **10 videozelfstudies** van in totaal 4,5 uur, elk met een bijbehorend **Jupyter-notebook**. Het notitieboek bevat alles wat je in de video ziet: code, uitvoer, afbeeldingen en opmerkingen.

**Opmerking:** De notebooks in deze repository zijn bijgewerkt om Python 3.9.1 en scikit-learn 0.23.2 te gebruiken. De originele notebooks (getoond in de video) gebruikten Python 2.7 en scikit-learn 0.16 en kunnen worden gedownload van de [archieftak](https://github.com/justmarkham/scikit-learn-videos/tree/archive). Je kunt lezen hoe ik de code heb bijgewerkt in deze [blogpost](https://www.dataschool.io/how-to-update-your-scikit-learn-code-for-2018/).

Je kunt [de hele serie bekijken](https://www.youtube.com/playlist?list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A) op YouTube en [alle notebooks bekijken](http://nbviewer.jupyter.org/github /justmarkham/scikit-learn-videos/tree/master/) met nbviewer.

[![Bekijk de eerste instructievideo](images/youtube.png)](https://www.youtube.com/watch?v=elojMnjn4kk&list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A&index=1 "Bekijk de eerste instructievideo")

Zodra je deze videoserie hebt voltooid, raad ik aan om je in te schrijven voor mijn online cursus, [Machine Learning with Text in Python](https://www.dataschool.io/learn/), om een ​​beter begrip te krijgen van scikit-learn en natuurlijke taal Verwerken.

## Inhoudsopgave

1. Wat is machine learning en hoe werkt het? ([video](https://www.youtube.com/watch?v=elojMnjn4kk&list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A&index=1), [notebook](01_machine_learning_intro.ipynb))
    - Wat is machinaal leren?
    - Wat zijn de twee hoofdcategorieën van Machine Learning?
    - Wat zijn enkele voorbeelden van Machine Learning?
    - Hoe "werkt" machine learning?

2. Python instellen voor machine learning: scikit-learn en Jupyter Notebook ([video](https://www.youtube.com/watch?v=IsXXlYVBt1M&list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A&index=2), [notebook](02_machine_learning_setup.ipynb ))
    - Wat zijn de voor- en nadelen van scikit-learn?
    - Hoe installeer ik scikit-learn?
    - Hoe gebruik ik de Jupyter Notebook?
    - Wat zijn enkele goede bronnen om Python te leren?

3. Aan de slag in scikit-learn met de beroemde irisdataset ([video](https://www.youtube.com/watch?v=hd1W4CyPX58&list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A&index=3), [notebook](03_getting_startedynb_iris.
    - Wat is de beroemde iris-dataset en hoe verhoudt deze zich tot Machine Learning?
    - Hoe laden we de irisdataset in scikit-learn?
    - Hoe beschrijven we een dataset met behulp van Machine Learning-terminologie?
    - Wat zijn de vier belangrijkste vereisten van scikit-learn voor het werken met gegevens?

4. Een Machine Learning-model trainen met scikit-learn ([video](https://www.youtube.com/watch?v=RlQuVL6-qe8&list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A&index=4), [notebook](04_model_training.ipynb))
    - Wat is het classificatiemodel van K-nearest buren?
    - Wat zijn de vier stappen voor modeltraining en -voorspelling in scikit-learn?
    - Hoe kan ik dit patroon toepassen op andere Machine Learning-modellen?

5. Machine Learning-modellen vergelijken in scikit-learn ([video](https://www.youtube.com/watch?v=0pP4EwWJgIU&list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A&index=5), [notebook](05_model_evaluation.ipynb))
    - Hoe kies ik welk model ik gebruik voor mijn begeleide leertaak?
    - Hoe kies ik de beste afstemmingsparameters voor dat model?
    - Hoe schat ik de waarschijnlijke prestaties van mijn model op out-of-sample data?

6. Datawetenschapspijplijn: pandas, seaborn, scikit-learn ([video](https://www.youtube.com/watch?v=3ZWuPVWq7p4&list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A&index=6), [notebook](06_linear_regression.ipynb)
    - Hoe gebruik ik de panda-bibliotheek om gegevens in Python te lezen?
    - Hoe gebruik ik de seaborn-bibliotheek om gegevens te visualiseren?
    - Wat is lineaire regressie en hoe werkt het?
    - Hoe train en interpreteer ik een lineair regressiemodel in scikit-learn?
    - Wat zijn enkele evaluatiestatistieken voor regressieproblemen?
    - Hoe kies ik welke functies ik in mijn model wil opnemen?

7. Kruisvalidatie voor het afstemmen van parameters, modelselectie en functieselectie ([video](https://www.youtube.com/watch?v=6dbrR-WymjI&list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A&index=7), [notebook](07_cross_validation .ipynb))
    - Wat is het nadeel van het gebruik van de trein/test split-procedure voor modelevaluatie?
    - Hoe overwint K-fold kruisvalidatie deze beperking?
    - Hoe kan kruisvalidatie worden gebruikt voor het selecteren van afstemmingsparameters, het kiezen tussen modellen en het selecteren van functies?
    - Wat zijn enkele mogelijke verbeteringen aan kruisvalidatie?

8. Efficiënt zoeken naar optimale afstemmingsparameters ([video](https://www.youtube.com/watch?v=Gol_qOgRqfA&list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A&index=8), [notebook](08_grid_search.ipynb))
    - Hoe kan K-voudige kruisvalidatie worden gebruikt?