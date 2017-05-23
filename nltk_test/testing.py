import sentiment_analysis as sa

text1 = "Hello python, it was nice meeting you. Catch up with you later."
text2 = "That movie was an utter crap. Waste of time and very bad storyline. Bad acting"

print(sa.sentiment(text1))
print(sa.sentiment(text2))