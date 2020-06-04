import pandas as pd
import seaborn as sns
# import spacy
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

sns.set()


def organize_ratings(ratings):
    ratings = ratings[['subgroup', 'grade', 'samplesize',
                       'approve', 'disapprove']]
    ratings = ratings.sort_index()

    # Only care about polls that scored well and dealt with Adults
    approved_grades = ['A+', 'A', 'A-', 'A/B', 'B+', 'B', 'B-', 'B/C']
    isQuality = ratings['grade'].isin(approved_grades)
    isAdults = ratings['subgroup'] == 'All polls'

    ratings = ratings[isQuality & isAdults]

    ratings['weighted_approve'] = ratings['samplesize'] * ratings['approve']
    ratings['weighted_disapprove'] = \
        ratings['samplesize'] * ratings['disapprove']

    ratings = ratings.resample('W').sum()
    ratings['approve'] = ratings['weighted_approve'] / ratings['samplesize']
    ratings['disapprove'] = \
        ratings['weighted_disapprove'] / ratings['samplesize']
    ratings = ratings[['samplesize', 'approve', 'disapprove']]

    return ratings.loc['2016-1-20':'2020-4-15']


def organize_tweets_ratings(tweets):
    tweets['count'] = 1
    tweets = tweets[['retweets', 'favorites', 'count']]

    tweets = tweets.resample('W').sum()
    tweets['retweet_avg'] = tweets['retweets'] / tweets['count']
    tweets['fav_avg'] = tweets['favorites'] / tweets['count']

    return tweets.loc['2017-01-20':'2020-4-15']


def create_plots(approval_ratings):
    approval_ratings = approval_ratings.resample('W').mean()

    fig, [[ax1, ax2, ax3], [ax4, ax5, ax6]] = \
        plt.subplots(nrows=2, ncols=3, figsize=(20, 10))

    sns.regplot(x='approve', y='count',
                data=approval_ratings, ax=ax1)
    ax1.set(xlabel='Avg Approval Rating (Week)',
            ylabel='Weekly Tweet Count',
            title='Approval Ratings per Week vs. Weekly Tweet Count',
            xlim=(35, 45), ylim=(0, 200))

    sns.regplot(x='approve', y='fav_avg',
                data=approval_ratings, ax=ax2)
    ax2.set(xlabel='Avg Approval Rating (Week)',
            ylabel='Weekly Tweet Favorites',
            title='Approval Ratings per Week vs. Weekly Tweet Favorites',
            xlim=(35, 45), ylim=(40000, 160000))

    sns.regplot(x='approve', y='retweet_avg',
                data=approval_ratings, ax=ax3)
    ax3.set(xlabel='Avg Approval Rating (Week)',
            ylabel='Weekly Tweet Retweets',
            title='Approval Ratings per Week vs. Weekly Tweet Retweets',
            xlim=(35, 45), ylim=(5000, 35000))

    sns.regplot(x='disapprove', y='count',
                data=approval_ratings, ax=ax4)
    ax4.set(xlabel='Avg Disapproval Rating (Week)',
            ylabel='Weekly Tweet Count',
            title='Disapproval Ratings per Week vs. Weekly Tweet Count',
            xlim=(40, 60), ylim=(0, 200))

    sns.regplot(x='disapprove', y='fav_avg',
                data=approval_ratings, ax=ax5)
    ax5.set(xlabel='Avg Disapproval Rating (Week)',
            ylabel='Weekly Tweet Favorites',
            title='Disapproval Ratings per Week vs. Weekly Tweet Favorites',
            xlim=(40, 60), ylim=(40000, 160000))

    sns.regplot(x='disapprove', y='retweet_avg',
                data=approval_ratings, ax=ax6)
    ax6.set(xlabel='Avg Disapproval Rating (Week)',
            ylabel='Weekly Tweet Retweets',
            title='Disapproval Ratings per Week vs. Weekly Tweet Retweets',
            xlim=(40, 60), ylim=(5000, 35000))

    fig.savefig('test.png')


def organize_tweets_language(tweets):
    tweets = tweets.loc['2016-1-20':]

    # people_of_interest = ['obama', 'hillary', 'cnn', 'fox']

    fig, ax = plt.subplots(1, figsize=(20, 10))

    obama_data = tweets[tweets['content'].str.lower()
                        .str.contains('obama')]['favorites']
    sns.kdeplot(data=obama_data, ax=ax, clip=(0, 300000), legend=False)

    hillary_data = tweets[tweets['content'].str.lower()
                          .str.contains('hillary')]['favorites']
    sns.kdeplot(data=hillary_data, ax=ax, clip=(0, 300000), legend=False)

    cnn_data = tweets[tweets['content'].str.lower()
                      .str.contains('cnn')]['favorites']
    sns.kdeplot(data=cnn_data, ax=ax, clip=(0, 300000), legend=False)

    fox_data = tweets[tweets['content'].str.lower()
                      .str.contains('fox')]['favorites']
    sns.kdeplot(data=fox_data, ax=ax, clip=(0, 300000), legend=False)

    ax.set(xlabel='Tweet Favorites',
           ylabel='Kernal Density',
           title='Favorites Distribution by Group')
    plt.legend(loc='upper right', labels=['Obama', 'Hillary', 'CNN', 'FOX'])

    fig.savefig('people_of_interest.png')


def create_barplot(tweets):
    tweets = tweets.loc['2016-1-20':]

    persons_of_interest = ['obama', 'hillary', 'cnn', 'fox']

    df = {'Group': ['Obama', 'Hillary', 'CNN', 'FOX News', 'Overall'],
          'Favorite Average': [0, 0, 0, 0, 0],
          'Retweet Average': [0, 0, 0, 0, 0]}
    df = pd.DataFrame(df, columns=['Group', 'Favorite Average'])

    df.loc[4, 'Favorite Average'] = tweets['favorites'].mean()
    df.loc[4, 'Retweet Average'] = tweets['retweets'].mean()

    for i in range(len(persons_of_interest)):
        current = tweets[tweets['content'].str.lower()
                         .str.contains(persons_of_interest[i])]
        df.loc[i, 'Favorite Average'] = current['favorites'].mean()
        df.loc[i, 'Retweet Average'] = current['retweets'].mean()

    fig, [ax1, ax2] = plt.subplots(ncols=2, figsize=(20, 10))
    sns.catplot(x='Group', y='Favorite Average', ax=ax1, kind='bar', data=df)
    ax1.set(xlabel='Groups / Individuals',
            ylabel='Average Number of Favorites',
            title='Group\'s Average Number of Favorites')

    sns.catplot(x='Group', y='Retweet Average', ax=ax2, kind='bar', data=df)
    ax2.set(xlabel='Groups / Individuals',
            ylabel='Average Number of Retweets',
            title='Group\'s Average Number of Retweets')

    fig.savefig('barplots.png')


def process_tweets(tweets, nlp):
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

    tweets.to_csv('big_tweet_data.csv')


def spacy_p(tweet, nlp):
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
    tweets = tweets.copy()
    tweets['grammar_dict'] = \
        tweets['grammar_dict'].apply(lambda x: dict(eval(x)))
    tweets = tweets.join(tweets['grammar_dict'].apply(pd.Series))
    tweets = tweets.drop(['grammar_dict'], axis=1)
    tweets['capital perc'] = tweets['capital perc'] / tweets['character count']

    return tweets


def create_grammarplot(tweets):
    fig, ax = plt.subplots(1, figsize=(20, 10))
    sns.kdeplot(data=tweets['PROPN'], ax=ax)
    sns.kdeplot(data=tweets['NOUN'], ax=ax)
    sns.kdeplot(data=tweets['VERB'], ax=ax)
    sns.kdeplot(data=tweets['ADJ'], ax=ax)
    sns.kdeplot(data=tweets['ADV'], ax=ax)
    ax.set(xlabel='Part of Speech Percentage', ylabel='Kernal Density',
           title='Distribution of Grammar Usage in Tweets')

    fig.savefig('grammar.png')

    fig, ax = plt.subplots(1, figsize=(20, 10))
    sns.kdeplot(data=tweets['character count'], ax=ax, shade=True)
    ax.set(xlabel='Tweet Character Count', ylabel='Kernal Density',
           title='Distribution of Character Counts in Tweets')

    fig.savefig('charactercount.png')

    fig, ax = plt.subplots(1, figsize=(20, 10))
    sns.kdeplot(data=tweets['word count'], ax=ax, shade=True)
    ax.set(xlabel='Tweet Word Count', ylabel='Kernal Density',
           title='Distribution of Word Counts in Tweets')

    fig.savefig('wordcount.png')


def machine_learn_tweets(tweets):
    tweets = tweets.copy()
    tweets = tweets[['favorites', 'retweets', 'word count',
                     'character count', '! count',
                     'hashtag count', '@ count']]
    tweets.dropna()
    features = tweets.loc[:, (tweets.columns != 'retweets')
                          & (tweets.columns != 'favorites')]

    labels = tweets['retweets']

    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.01)

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
        train_test_split(features, labels, test_size=0.01)

    model = DecisionTreeRegressor(max_depth=50)
    model.fit(features_train, labels_train)

    train_predictions = model.predict(features_train)
    print('Train MAE Favorites:',
          mean_absolute_error(labels_train, train_predictions))

    test_predictions = model.predict(features_test)
    print('Test  MAE Favorites:',
          mean_absolute_error(labels_test, test_predictions))


def main():
    tweets = pd.read_csv('CSE_163_Project/realdonaldtrump.csv',
                         index_col='date', parse_dates=True)
    # ratings_extra = pd.read_csv('CSE_163_Project/approval_topline.csv',
    #                             index_col='timestamp', parse_dates=True)
    ratings = pd.read_csv('CSE_163_Project/approval_polllist.csv',
                          index_col='enddate', parse_dates=True)

    ratings = organize_ratings(ratings)
    tweets_ratings = organize_tweets_ratings(tweets)
    approval_ratings = tweets_ratings.join(ratings).dropna()
    create_plots(approval_ratings)

    organize_tweets_language(tweets)
    create_barplot(tweets)

    # This command generated big_tweet_data.csv. Takes 10 or so minutes.
    # nlp = spacy.load('en_core_web_sm')
    # process_tweets(tweets, nlp)

    big_tweets = pd.read_csv('C:/Users/Chet Kruse/Desktop/CSE163/' +
                             'CSE_163_Project/big_tweet_data.csv',
                             index_col='date', parse_dates=True)
    big_tweets = process_big_tweets(big_tweets)
    print(big_tweets['retweets'].median())
    print(big_tweets['favorites'].median())
    # create_grammarplot(big_tweets)

    machine_learn_tweets(big_tweets)


if __name__ == '__main__':
    main()
