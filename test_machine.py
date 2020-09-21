"""
CSE 163
Jonathan Li and Chet Kruse

test_machine.py is a python program that validates the proper
hyperparameter to use for langauge.py. The program tests
and saves a graph demonstrating the changes in train and test
error as max_depth changes.
"""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
sns.set()


def process_big_tweets(tweets):
    """
    Passes in an unprocessed big tweets dataset. Returns
    an expanded big tweet dataset that includes improved
    access to details about grammar usage and character count.
    """
    tweets = tweets.copy()
    tweets['grammar_dict'] = \
        tweets['grammar_dict'].apply(lambda x: dict(eval(x)))
    tweets = tweets.join(tweets['grammar_dict'].apply(pd.Series))
    tweets = tweets.drop(['grammar_dict'], axis=1)
    tweets['capital perc'] = tweets['capital perc'] / tweets['character count']

    return tweets


def test_machine(tweets, kind):
    """
    Passes in big tweets dataset as well as a string for the kind
    of label. "kind" can either be "favorites" or "retweets". Saves
    a graph of the training mean average error and the test mean
    average error with respect to the max_depth value under
    test_machine_(kind).png.
    """
    tweets = tweets.copy()
    tweets['date'] = tweets['date'].apply(lambda x: int(x[0:4]))
    train = []
    test = []
    for i in range(2, 100):
        mae_values = machine_learn_tweets(tweets, i, kind)
        train.append(mae_values[0])
        test.append(mae_values[1])

    fig, ax = plt.subplots(1, figsize=(20, 10))

    depth = list(range(2, 100))

    df = pd.DataFrame(list(zip(depth, train, test)),
                      columns=['Depth', 'Train MAE', 'Test MAE'])

    print(df)

    sns.lineplot(x='Depth', y='Train MAE',
                 ax=ax, data=df, color='blue')
    sns.lineplot(x='Depth', y='Test MAE',
                 ax=ax, data=df, color='red')
    ax.set(xlabel='Max Depth Hyperparameter', ylabel='Mean Absolute Error',
           title=kind.upper() + ' Machine Learning Test')

    fig.savefig('images/test_machine_' + kind + '.png')


def machine_learn_tweets(tweets, depth, kind):
    """
    Passes in the big tweets dataset, a depth value, and
    a kind. "kind" can either be "favorites" or "retweets".
    Returns a tuple of the train mean absolute error and the
    test mean absolute error for the given model.
    """
    tweets = tweets[['favorites', 'retweets', 'word count',
                     'NOUN', 'PROPN', 'VERB', 'ADJ', 'ADV',
                     'character count', '! count',
                     'hashtag count', '@ count', 'date']]
    tweets.dropna()
    features = tweets.loc[:, (tweets.columns != 'retweets')
                          & (tweets.columns != 'favorites')]

    labels = tweets[kind]

    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.2)

    model = DecisionTreeRegressor(max_depth=depth)
    model.fit(features_train, labels_train)

    train_predictions = model.predict(features_train)
    train = mean_absolute_error(labels_train, train_predictions)

    test_predictions = model.predict(features_test)
    test = mean_absolute_error(labels_test, test_predictions)

    return (train, test)


def main():
    big_tweets = pd.read_csv('data/big_tweet_data.csv')
    big_tweets = process_big_tweets(big_tweets)
    test_machine(big_tweets, 'retweets')
    test_machine(big_tweets, 'favorites')


if __name__ == '__main__':
    main()
