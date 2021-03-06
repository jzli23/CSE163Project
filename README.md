# Reproducing Results
1. To obtain the necessary files, click [here](https://github.com/jzli23/CSE163Project) to clone our repository to your local environment.
2. You will need the libraries:
    1. _pandas_
    2. _matplotlib_
    3. _seaborn_
    4. _spacy_
    5. _sklearn_
        1. From sklearn, will you need the _mean_absolute_error, DecisionTreeRegressor, and train_testsplit_ methods.
3. You should now see the following datasets on your local environment:
    1. approval_pollist.csv
    2. approval_topline.csv
    3. big_tweet_data.csv
    4. realdonaldtrump.csv
4. As well as the Python files:
    1. approval_ratings.py
    2. language.py
    3. mentions.py
    4. test_machine.py
5. Running these 3 Python files should result in the following images being produced in your directory:
    1. approval_ratings.py produces:
        1. approval.png
    2. mentions.py
        1. people_of_interest.png
    3. language.py
        1. grammar.png
        2. wordcount.png
        3. charactercount.png
        4. comparisons.png
    4. test_machine.py
        1. test_machine_retweets.png
        2. test_machine_favorites.png
6. If you want to generate another _big_tweet_data_ dataset, uncomment the _"import spacy"_ and _lines 138-143_ in **language.py**.



![Good Luck Image](Good_Luck.jpg)