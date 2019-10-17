from flaskexample.groupon_scrape import scraping
from flaskexample.preprocess_word2vec import preprocess_df
from flaskexample.preprocess_lda import categorize
from flaskexample.sentiment import pred_sentiment

import flaskexample.utils
import numpy as np


def ModelIt(births):
    # Scraping
    # scraping(births)
    
    # Preprocess scraped tweets
    processed_file_name = preprocess_df("./flaskexample/static/data/groupon_review.txt")
    # processed_file_name = preprocess_df(births)

    # Categorization with LDA
    lda_file_name = categorize(processed_file_name)

    # Prediction
    df = pred_sentiment(lda_file_name)
    len_df = len(df)
    
    count_service = len(df.loc[df['service']>0.02,['service']])
    count_service_pos = np.sum(df['pred_service'].values)
    count_room = len(df.loc[df['room']>0.02,['room']])
    count_room_pos = np.sum(df['pred_room'].values)
    count_loc = len(df.loc[df['loc']>0.02,['loc']])
    count_loc_pos = np.sum(df['pred_loc'].values)

    pct_service = round(100*count_service/len_df,1)
    pct_service_pos = round(100*count_service_pos/count_service,1)
    pct_room = round(100*count_room/len_df,1)
    pct_room_pos = round(100*count_room_pos/count_room,1)
    pct_loc = round(100*count_loc/len_df,1)
    pct_loc_pos = round(100*count_loc_pos/count_loc,1)

    comments = {'pos_service': 'No relevant comments found.',
                'neg_service': 'No relevant comments found.',
                'pos_room': 'No relevant comments found.',
                'neg_room': 'No relevant comments found.',
                'pos_loc': 'No relevant comments found.',
                'neg_loc': 'No relevant comments found.'
                }
    print(df.head(5))
    for i in range(len_df):
        if float(df.loc[i,'service']) > 0.3 and float(df.loc[i,'prob_service']) >= 0.85 and comments['pos_service'] == 'No relevant comments found.':
            comments['pos_service'] = df.loc[i,'org_sent']
        elif float(df.loc[i,'service']) > 0.3 and float(df.loc[i,'prob_service']) <= 0.23 and comments['neg_service'] == 'No relevant comments found.':
            comments['neg_service'] = df.loc[i,'org_sent']
        elif float(df.loc[i,'room']) > 0.3 and float(df.loc[i,'prob_room']) >= 0.85 and comments['pos_room'] == 'No relevant comments found.':
            comments['pos_room'] = df.loc[i,'org_sent']
        elif float(df.loc[i,'room']) > 0.3 and float(df.loc[i,'prob_room']) <= 0.4 and comments['neg_room'] == 'No relevant comments found.':
            comments['neg_room'] = df.loc[i,'org_sent']
        elif float(df.loc[i,'loc']) > 0.3 and float(df.loc[i,'prob_loc']) >= 0.85 and comments['pos_loc'] == 'No relevant comments found.':
            comments['pos_loc'] = df.loc[i,'org_sent']
        elif float(df.loc[i,'loc']) > 0.3 and float(df.loc[i,'prob_loc']) <= 0.4 and comments['neg_loc'] == 'No relevant comments found.':
            comments['neg_loc'] = df.loc[i,'org_sent']

    # Output
    service = str(pct_service)+'% of the reviewers mentioned service. ' + str(pct_service_pos) + '% of them spoke highly of it.'
    room = str(pct_room)+'% of the reviewers mentioned room. ' + str(pct_room_pos) + '% of them spoke highly of it.'
    location_com = str(pct_loc)+'% of the reviewers mentioned location. ' + str(pct_loc_pos) + '% of them spoke highly of it.'

    import matplotlib.pyplot as plt
 
    # Make a dataset:
    height = [pct_service_pos*0.05, pct_room_pos*0.05, pct_loc_pos*0.05]
    bars = ('Service', 'Room', 'Location')
    y_pos = np.arange(len(bars))

    print(height)
    # Create bars
    # Create names on the x-axis
    plt.xticks(y_pos, bars)
    plt.bar(y_pos, height, color=['blue', 'red', 'green'])

    axes = plt.gca()
    axes.set_ylim([0,5])
    plt.suptitle('Predicted Aspect Rating', fontsize=24)
    plt.savefig('./flaskexample/static/data/rate_plot.png')

    return service, room, location_com, comments
