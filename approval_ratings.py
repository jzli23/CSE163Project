import pandas as pd
import seaborn as sb
import spacy

def organize_ratings(ratings):
    ratings = ratings[['subgroup', 'grade', 'samplesize', 'approve', 'disapprove']]
    ratings = ratings.sort_index()

    # Only Care about polls that scored well and dealt with Adults
    approved_grades = ['A+', 'A', 'A-', 'A/B', 'B+', 'B', 'B-', 'B/C']
    isQuality = ratings['grade'].isin(approved_grades)
    isAdults = ratings['subgroup'] == 'All polls'

    ratings = ratings[isQuality & isAdults]

    ratings['weighted_approve'] = ratings['samplesize'] * ratings['approve']
    ratings['weighted_disapprove'] = ratings['samplesize'] * ratings['disapprove']

    ratings = ratings.resample('D').sum()
    ratings['approve'] = ratings['weighted_approve'] / ratings['samplesize']
    ratings['disapprove'] = ratings['weighted_disapprove'] / ratings['samplesize']

    return ratings[['samplesize', 'approve', 'disapprove']].loc['2016-1-20':'2020-4-15']


def organize_tweets(tweets):
    tweets['count'] = 1
    tweets = tweets[['retweets', 'favorites', 'count']]

    tweets = tweets.resample('D').sum()
    tweets['retweet_avg'] = tweets['retweets'] / tweets['count']
    tweets['fav_avg'] = tweets['retweets'] / tweets['count']

    return tweets.loc['2017-01-20':'2020-4-15']

def main():
    print('cool')

    tweets = pd.read_csv('C:/Users/Chet Kruse/Desktop/CSE163/CSE_163_Project/realdonaldtrump.csv',
                         index_col='date', parse_dates=True)
    ratings_extra = pd.read_csv('C:/Users/Chet Kruse/Desktop/CSE163/CSE_163_Project/approval_topline.csv',
                          index_col='timestamp', parse_dates=True)
    ratings = pd.read_csv('C:/Users/Chet Kruse/Desktop/CSE163/CSE_163_Project/approval_polllist.csv',
                           index_col='enddate', parse_dates=True)

    ratings = organize_ratings(ratings)
    print(ratings)

    tweets = organize_tweets(tweets)
    print(tweets)

if __name__ == '__main__':
    main()
