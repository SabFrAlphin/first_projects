
import argparse
import sys
import re
import requests
from bs4 import BeautifulSoup as soup
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

parser = argparse.ArgumentParser(description='A program downloading lyrics from specified artists.')

parser.add_argument('-a1', '--artist1', type=str, default="Abba",
                    help='name of the first artist')

parser.add_argument('-a2', '--artist2', type=str, default='Beyonce',
                    help='number of the second artist')

args = parser.parse_args()

artist1 = args.artist1
artist2 = args.artist2


# ## 1.Scrape content off lyrics.com
headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
url = 'https://www.lyrics.com'
artist_url = 'https://www.lyrics.com/artist.php?name='+artist1
artist2_url = 'https://www.lyrics.com/artist.php?name='+artist2

#artist_url = 'https://www.lyrics.com/artist.php?name=Elton-John&aid=4617&o=1'
#artist2_url = 'https://www.lyrics.com/artist.php?name=The-Beatles&aid=3644&o=1'

# define variables for code
no_of_songs = 10

container = requests.get(artist_url, headers)
container2 = requests.get(artist2_url, headers)
container_html = soup(container.text, 'html.parser')
container2_html = soup(container2.text, 'html.parser')

link = container_html.find_all('a', href=True)
link2 = container2_html.find_all('a', href=True)

all_links = [x.get('href') for x in link if r'/lyric' in x.get('href')]
all_links2 = [x.get('href') for x in link2 if r'/lyric' in x.get('href')]
df = pd.DataFrame()

# save all songs in one csv
for a in all_links[:no_of_songs]:
    # open required url
    response = requests.get(url+a, headers)
    # get html text from site
    response_html = soup(response.text, 'html.parser')
    # identify the song text
    response_text = response_html.find_all('pre', attrs={"id": "lyric-body-text"})
    # identify the song title
    response_title = response_html.find('h1', attrs={"class": "lyric-title"}).text
    # loop over song text and extract pure text / get rid of html technical things
    for i in response_text:
        song = i.text
        df = df.append([song], ignore_index=True)
# write scraped content to file
df.to_csv('artist1.csv')

df2 = pd.DataFrame()
# save all songs in one csv
for a in all_links2[:no_of_songs]:
    # open required url
    response2 = requests.get(url+a, headers)
    # get html text from site
    response2_html = soup(response2.text, 'html.parser')
    # identify the song text
    response2_text = response2_html.find_all('pre', attrs={"id": "lyric-body-text"})
    # identify the song title
    response2_title = response2_html.find('h1', attrs={"class": "lyric-title"}).text
    # loop over song text and extract pure text / get rid of html technical things
    for i in response2_text:
        song2 = i.text
        df2 = df2.append([song2], ignore_index=True)
# write scraped content to file
df2.to_csv('artist2.csv')


artist_list = [[artist1] * no_of_songs + [artist2] * no_of_songs]
df3 = pd.concat([df, df2], ignore_index=True).set_index(artist_list).rename(columns={0: 'songtext'})
df3['str_compare'] = df3['songtext'].str.slice(0, 20)
df3 = df3.drop_duplicates(subset=['str_compare'], keep='first')
df3 = df3.drop('str_compare', axis=1)

artist_new = df3.index

# ## 2. Apply Bag of Words
cv = CountVectorizer(stop_words='english')
cv.fit(df3['songtext'])

df3_vecs = cv.transform(df3['songtext'])
df3_cv = pd.DataFrame(df3_vecs.todense(), index=artist_new, columns=cv.get_feature_names())


# ## 3. Apply TfIDf Transformer
tf = TfidfTransformer()
tf_vecs = tf.fit_transform(df3_vecs)
df3_tf = pd.DataFrame(tf_vecs.todense().round(2), index=artist_new, columns=cv.get_feature_names())

# ## 3. Run Naive Bayes Model
df3_tf.reset_index(inplace=True)

y = df3_tf['index']
x = df3_tf.drop('index', axis=1)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, random_state=87)

mn = MultinomialNB(alpha=1)
mn.fit(xtrain, ytrain)
test_score = mn.score(xtest, ytest)
training_score = mn.score(xtrain, ytrain)

y_train_pred = mn.predict(xtrain)
print('y_train_pred: ', y_train_pred)

print('training score ', training_score)
print('test score ', test_score)
