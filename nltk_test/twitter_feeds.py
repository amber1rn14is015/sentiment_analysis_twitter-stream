import tweepy
import sentiment_analysis
import json

#consumer key, consumer secret, access token, access secret.
consumer_key = "-------your consumer key----------"
consumer_secret = "--------your consumer secret----------"
access_token = "---------your access token---------"
access_token_secret = "--------your access token secret---------"

topic = "------the topic you want to get the tweets about------"


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)


class MyStreamListener(tweepy.StreamListener):


    # def on_data(self, data):
    #     # Twitter returns data in JSON format - we need to decode it first
    #     decoded = json.loads(data)
    #
    #     # Also, we convert UTF-8 to ASCII ignoring all bad characters sent by users
    #     print (len(decoded['text'].encode('ascii', 'ignore')))
    #     print ('')
    #     return True

    def on_status(self, status):
        tweet = status.text.split("https://")[0]
        sentiment, confidence = sentiment_analysis.sentiment(tweet)
        print(tweet)
        print('')

        if(confidence*100>=80):
            output = open('twitter-feeds.txt','a')
            output.write(sentiment)
            output.write('\n')
            output.close()


    def on_error(self, status):
        print(status)


myStreamListener = MyStreamListener()
myStream = tweepy.Stream(auth=api.auth, listener=myStreamListener)
myStream.filter(track=[topic], async=True, languages=['en'])
