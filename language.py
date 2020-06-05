"""
CSE 163
Jonathan Li and Chet Kruse

language.py is a program that focuses on answering the question
"Do certain language characteristics differ between Donald Trump’s 
popular tweets and his not so popular ones?" The program processes
Donald Trump's grammar in his tweets, generates plots representing
relationships and distributions, and creates a machine learning
model that attempts to predict tweet success based off of grammar
and tweet structure.
"""

import pandas as pd
import seaborn as sns
# import spacy
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

sns.set()


def process_tweets(tweets, nlp):
    """
    Passes in original tweets dataset and the spacy
    natural language processing tool. Returns a completed
    file named "big_tweet_data.csv" that contains all information
    pertaining to each tweet's structure and grammar usage.
    """
    tweets = tweets.copy()
    tweets = tweets.loc['2016-1-20':]
    tweets['character count'] = tweets['content'].str.len()

    tweets['hashtag list'] = tweets['hashtags'].str.split(',')
    tweets['hashtag list'] = tweets['hashtag list'].fillna('')
    tweets['hashtag count'] = tweets['hashtag list'].apply(lambda x: len(x))

    tweets['@ list'] = tweets['mentions'].str.split(',')
    tweets['@ list'] = tweets['@ list'].fillna('')
    tweets['@ count'] = tweets['@ list'].apply(lambda x: len(x))

    tweets['grammar_dict'] = tweets['content'].apply(lambda x: spacy_p(x, nlp))

    tweets.to_csv('data/big_tweet_data.csv')


def spacy_p(tweet, nlp):
    """
    Passes in an individual tweet and the spacy
    natural language processing tool. Returns a dictionary
    that contains all important information regarding
    the tweet's grammar usage.
    """
    doc = nlp(tweet)
    grammar_var = {'word count': 0,
                   'NOUN': 0,
                   'VERB': 0,
                   'PROPN': 0,
                   'ADJ': 0,
                   'ADV': 0,
                   'capital perc': 0,
                   '! count': 0,
                   '? count': 0}
    grammar_var['word count'] = len(doc)
    factor = 1 / (grammar_var['word count'])
    for token in doc:
        if token.pos_ in grammar_var.keys():
            grammar_var[token.pos_] += factor
        for c in token.shape_:
            if c.isupper():
                grammar_var['capital perc'] += 1
            elif c == '!':
                grammar_var['! count'] += 1
            elif c == '?':
                grammar_var['? count'] += 1
    return grammar_var


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


def create_grammarplot(tweets):
    """
    Passes in the processed big tweets dataset. Saves
    plots regarding the distribution of grammar usage,
    distribution of character count and word count,
    as well as the relationship plots between NOUN
    percentage / character count / PROPER NOUN percentage
    vs. favorites / retweets.
    """
    tweets = tweets.copy()
    # To make the grammar plot
    fig, ax = plt.subplots(1, figsize=(20, 10))
    sns.kdeplot(data=tweets['PROPN'], ax=ax)
    sns.kdeplot(data=tweets['NOUN'], ax=ax)
    sns.kdeplot(data=tweets['VERB'], ax=ax)
    sns.kdeplot(data=tweets['ADJ'], ax=ax)
    sns.kdeplot(data=tweets['ADV'], ax=ax)
    ax.set(xlabel='Part of Speech Ratio', ylabel='Probability Density',
           title='Distribution of Grammar Usage in Tweets')

    fig.savefig('images/grammar.png')

    # To make the charactercount plot
    fig, ax = plt.subplots(1, figsize=(20, 10))
    sns.kdeplot(data=tweets['character count'], ax=ax, shade=True)
    ax.set(xlabel='Tweet Character Count', ylabel='Probability Density',
           title='Distribution of Character Counts in Tweets')

    fig.savefig('images/charactercount.png')

    # To make wordcount plot
    fig, ax = plt.subplots(1, figsize=(20, 10))
    sns.kdeplot(data=tweets['word count'], ax=ax, shade=True)
    ax.set(xlabel='Tweet Word Count', ylabel='Probability Density',
           title='Distribution of Character Counts in Tweets')

    fig.savefig('images/wordcount.png')

    # To make the comparisons plot
    tweets = tweets.resample('W').mean()
    fig, [[ax1, ax2, ax3], [ax4, ax5, ax6]] = \
        plt.subplots(nrows=2, ncols=3, figsize=(20, 10))

    sns.regplot(x='NOUN', y='favorites', marker='+', scatter_kws={"s": 10},
                data=tweets, ax=ax1)
    ax1.set(xlabel='Mean Noun Usage Ratio per Week',
            ylabel='Mean Favorites per Week',
            title='Mean Noun Usage Ratio vs Mean Favorites per Week')

    sns.regplot(x='character count', y='favorites', marker='+',
                scatter_kws={"s": 10}, data=tweets, ax=ax2)
    ax2.set(xlabel='Mean Character Count per Week',
            ylabel='Mean Favorites per Week',
            title='Mean Character Count vs Mean Favorites per Week')

    sns.regplot(x='PROPN', y='favorites', marker='+', scatter_kws={"s": 10},
                data=tweets, ax=ax3)
    ax3.set(xlabel='Mean Proper Noun Usage Ratio per Week',
            ylabel='Mean Favorites per Week',
            title='Mean Proper Noun Usage Ratio vs Mean Favorites per Week')

    sns.regplot(x='NOUN', y='retweets', marker='+', scatter_kws={"s": 10},
                data=tweets, ax=ax4)
    ax4.set(xlabel='Mean Noun Usage Ratio per Week',
            ylabel='Mean Retweets per Week',
            title='Mean Noun Usage Ratio vs Mean Retweets per Week')

    sns.regplot(x='character count', y='retweets', marker='+',
                scatter_kws={"s": 10}, data=tweets, ax=ax5)
    ax5.set(xlabel='Mean Character Count per Week',
            ylabel='Mean Retweets per Week',
            title='Mean Character Count vs Mean Retweets per Week')

    sns.regplot(x='PROPN', y='retweets', marker='+', scatter_kws={"s": 10},
                data=tweets, ax=ax6)
    ax6.set(xlabel='Proper Noun Usage Ratio per Week',
            ylabel='Mean Retweets per Week',
            title='Mean Proper Noun Usage Ratio vs Mean Retweets per Week',
            xlim=(0.08, 0.22), ylim=(0, 35000))

    fig.savefig('images/comparisons.png')


def machine_learn_tweets(tweets):
    """
    Passes in the big tweets dataset. Creates a machine learning
    model that attempts to predict both the favorites and retweets
    of a particular post based off of grammar usage, sentence structure
    and date of creation. Prints the Mean Average Values for both the training
    and test tweets. 
    """
    tweets = tweets.copy()
    tweets['year'] = tweets.index
    tweets['year'] = tweets['year'].map(lambda x: x.year)
    tweets = tweets[['favorites', 'retweets', 'word count',
                     'NOUN', 'PROPN', 'VERB', 'ADJ', 'ADV',
                     'character count', '! count',
                     'hashtag count', '@ count', 'year']]
    tweets.dropna()
    features = tweets.loc[:, (tweets.columns != 'retweets')
                          & (tweets.columns != 'favorites')]

    labels = tweets['retweets']

    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.10)

    model = DecisionTreeRegressor(max_depth=50)
    model.fit(features_train, labels_train)

    train_predictions = model.predict(features_train)
    print('Train MAE Retweets:',
          mean_absolute_error(labels_train, train_predictions))

    test_predictions = model.predict(features_test)
    print('Test  MAE Retweets:',
          mean_absolute_error(labels_test, test_predictions))

    labels = tweets['favorites']

    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.2)

    model = DecisionTreeRegressor(max_depth=15)
    model.fit(features_train, labels_train)

    train_predictions = model.predict(features_train)
    print('Train MAE Favorites:',
          mean_absolute_error(labels_train, train_predictions))

    test_predictions = model.predict(features_test)
    print('Test  MAE Favorites:',
          mean_absolute_error(labels_test, test_predictions))


def main():
    # tweets = pd.read_csv('data/realdonaldtrump.csv',
    #                      index_col='date', parse_dates=True)

    # This command generated big_tweet_data.csv. Takes 10 or so minutes.
    # nlp = spacy.load('en_core_web_sm')
    # process_tweets(tweets, nlp)

    big_tweets = pd.read_csv('data/big_tweet_data.csv',
                             index_col='date', parse_dates=True)
    big_tweets = process_big_tweets(big_tweets)
    print('Tweet Retweet Median: ' + str(big_tweets['retweets'].median()))
    print('Tweet Favorite Median: ' + str(big_tweets['favorites'].median()))

    create_grammarplot(big_tweets)

    machine_learn_tweets(big_tweets)


if __name__ == '__main__':
    main()
