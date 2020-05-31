import pandas as pd
import seaborn as sns
import spacy
import matplotlib.pyplot as plt

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
    ratings['weighted_disapprove'] = ratings['samplesize'] * ratings['disapprove']

    ratings = ratings.resample('W').sum()
    ratings['approve'] = ratings['weighted_approve'] / ratings['samplesize']
    ratings['disapprove'] = ratings['weighted_disapprove'] / ratings['samplesize']

    return ratings[['samplesize', 'approve', 'disapprove']].loc['2016-1-20':'2020-4-15']


def organize_tweets_ratings(tweets):
    tweets['count'] = 1
    tweets = tweets[['retweets', 'favorites', 'count']]

    tweets = tweets.resample('W').sum()
    tweets['retweet_avg'] = tweets['retweets'] / tweets['count']
    tweets['fav_avg'] = tweets['favorites'] / tweets['count']

    return tweets.loc['2017-01-20':'2020-4-15']


def create_plots(approval_ratings):
    approval_ratings = approval_ratings.resample('W').mean()

    fig, [[ax1, ax2, ax3], [ax4, ax5, ax6]] = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))

    sns.regplot(x='approve', y='count', data=approval_ratings, ax=ax1)
    ax1.set(xlabel='Avg Approval Rating (Week)', ylabel='Weekly Tweet Count', title='Approval Ratings per Week vs. Weekly Tweet Count',
            xlim=(35, 45), ylim=(0, 200))

    sns.regplot(x='approve', y='fav_avg', data=approval_ratings, ax=ax2)
    ax2.set(xlabel='Avg Approval Rating (Week)', ylabel='Weekly Tweet Favorites', title='Approval Ratings per Week vs. Weekly Tweet Favorites',
            xlim=(35, 45), ylim=(40000, 160000))

    sns.regplot(x='approve', y='retweet_avg', data=approval_ratings, ax=ax3)
    ax3.set(xlabel='Avg Approval Rating (Week)', ylabel='Weekly Tweet Retweets', title='Approval Ratings per Week vs. Weekly Tweet Retweets',
            xlim=(35, 45), ylim=(5000, 35000))

    sns.regplot(x='disapprove', y='count', data=approval_ratings, ax=ax4)
    ax4.set(xlabel='Avg Disapproval Rating (Week)', ylabel='Weekly Tweet Count', title='Disapproval Ratings per Week vs. Weekly Tweet Count',
            xlim=(40, 60), ylim=(0, 200))

    sns.regplot(x='disapprove', y='fav_avg', data=approval_ratings, ax=ax5)
    ax5.set(xlabel='Avg Disapproval Rating (Week)', ylabel='Weekly Tweet Favorites', title='Disapproval Ratings per Week vs. Weekly Tweet Favorites',
            xlim=(40, 60), ylim=(40000, 160000))

    sns.regplot(x='disapprove', y='retweet_avg', data=approval_ratings, ax=ax6)
    ax6.set(xlabel='Avg Disapproval Rating (Week)', ylabel='Weekly Tweet Retweets', title='Disapproval Ratings per Week vs. Weekly Tweet Retweets',
            xlim=(40, 60), ylim=(5000, 35000))

    fig.savefig('test.png')


def organize_tweets_language(tweets):
    tweets = tweets.loc['2016-1-20':]
    print(tweets[tweets['content'] == '@CNNPolitics'])



def main():
    print('cool')

    tweets = pd.read_csv('C:/Users/Chet Kruse/Desktop/CSE163/CSE_163_Project/realdonaldtrump.csv',
                         index_col='date', parse_dates=True)
    ratings_extra = pd.read_csv('C:/Users/Chet Kruse/Desktop/CSE163/CSE_163_Project/approval_topline.csv',
                          index_col='timestamp', parse_dates=True)
    ratings = pd.read_csv('C:/Users/Chet Kruse/Desktop/CSE163/CSE_163_Project/approval_polllist.csv',
                           index_col='enddate', parse_dates=True)

    ratings = organize_ratings(ratings)
    tweets_ratings = organize_tweets_ratings(tweets)
    approval_ratings = tweets_ratings.join(ratings).dropna()
    create_plots(approval_ratings)

    organize_tweets_language(tweets)



if __name__ == '__main__':
    main()