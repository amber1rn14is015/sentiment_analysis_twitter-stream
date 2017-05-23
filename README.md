# sentiment_analysis_twitter-stream
Analyse the sentiment of the public tweets in either a positive or a negative and plots a graph, about some particular track.



This project requires you to have python 3.x along with the following modules.
So before proceeding forward install the following modules.

  1. nltk
  2. scikit-learn
  3. matplotlob
  4. tweepy
  
After getting the required modules, run sentiment_graph.py and simultaneously run twitter_feeds.py.
twutter_feeds.py will create a new file twitter-feeds.txt that will contain either a 'pos' for positive or 'neg' for negative on each line.
These sentiments are based on the public tweets that have been streamed via twitter api.
Based on these sentiments which are stored in twitter-feeds.txt, sentiment_graph.py will plot the live graph.


Do not forget to enter your twitter cerdentials into twitter_feeds.py.

To get your twitter credentials, go to https://apps.twitter.com

The training of classifiers is still going on. Currently the classifiers are a bit negative sentiments biased.
