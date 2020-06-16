
# import libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# get user input
x_keywords = input('please enter 5 words, seperated by comma: ')
x_keywords = [x_keywords]

# import data
df3 = pd.read_csv('elton_beatles.csv', index_col=0)
artist_new = df3.index

# ## 2. Apply Bag of Words
cv = CountVectorizer(stop_words='english')
cv.fit(df3['songtext'])

df3_vecs = cv.transform(df3['songtext'])
df3_cv = pd.DataFrame(df3_vecs.todense(), index=artist_new, columns=cv.get_feature_names())

x_keywords = pd.DataFrame(x_keywords, columns=['keywords'])
x_keywords_vecs = cv.transform(x_keywords['keywords'])

# ## 3. Apply TfIDf Transformer
tf = TfidfTransformer()
tf_vecs = tf.fit_transform(df3_vecs)
x_keywords_tf_vecs = tf.transform(x_keywords_vecs)
df3_tf = pd.DataFrame(tf_vecs.todense(), index=artist_new, columns=cv.get_feature_names())
x_keywords_tf = pd.DataFrame(x_keywords_tf_vecs.todense(), columns=cv.get_feature_names())

# ## 3. Run Naive Bayes Model
df3_tf.reset_index(inplace=True)

y = df3_tf['index']
x = df3_tf.drop('index', axis=1)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, random_state=43)

mn = MultinomialNB(alpha=0.4)
mn.fit(xtrain, ytrain)
test_score = mn.score(xtest, ytest).round(3)
training_score = mn.score(xtrain, ytrain).round(3)
y_keywords_pred = mn.predict(x_keywords_tf)
print(x_keywords)
print('training score ', training_score)
print('test score ', test_score)
print('Your words are most likely from a song by: ', y_keywords_pred)
