import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()


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

    overall_mean_fav = tweets['favorites'].mean()
    overall_mean_retweet = tweets['retweets'].mean()
    df.loc[4, 'Favorite Average'] = overall_mean_fav
    df.loc[4, 'Retweet Average'] = overall_mean_retweet
    print('Overall Favorite Mean: ' + str(overall_mean_fav))
    print('Overall Retweet Mean: ' + str(overall_mean_retweet))

    for i in range(len(persons_of_interest)):
        current = tweets[tweets['content'].str.lower()
                         .str.contains(persons_of_interest[i])]
        ind_mean_fav = current['favorites'].mean()
        ind_mean_retweet = current['retweets'].mean()
        df.loc[i, 'Favorite Average'] = ind_mean_fav
        df.loc[i, 'Retweet Average'] = ind_mean_retweet
        print(str(df.loc[i, 'Group']) + ' Favorite Average: ' +
              str(ind_mean_fav))
        print(str(df.loc[i, 'Group']) + ' Retweet Average: ' +
              str(ind_mean_retweet))

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


def main():
    tweets = pd.read_csv('CSE163Project/realdonaldtrump.csv',
                         index_col='date', parse_dates=True)

    organize_tweets_language(tweets)
    create_barplot(tweets)


if __name__ == '__main__':
    main()
