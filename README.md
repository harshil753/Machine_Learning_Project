```python
import pandas as pd
import numpy as np
import sklearn as skl
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from functools import reduce
import winsound
```


```python
imdb = pd.read_csv("IMDb movies.csv")
```

    c:\users\harsh\desktop\env\lib\site-packages\IPython\core\interactiveshell.py:3062: DtypeWarning: Columns (3) have mixed types.Specify dtype option on import or set low_memory=False.
      has_raised = await self.run_ast_nodes(code_ast.body, cell_name,
    


```python
imdb.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>imdb_title_id</th>
      <th>title</th>
      <th>original_title</th>
      <th>year</th>
      <th>date_published</th>
      <th>genre</th>
      <th>duration</th>
      <th>country</th>
      <th>language</th>
      <th>director</th>
      <th>...</th>
      <th>actors</th>
      <th>description</th>
      <th>avg_vote</th>
      <th>votes</th>
      <th>budget</th>
      <th>usa_gross_income</th>
      <th>worlwide_gross_income</th>
      <th>metascore</th>
      <th>reviews_from_users</th>
      <th>reviews_from_critics</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>tt0000009</td>
      <td>Miss Jerry</td>
      <td>Miss Jerry</td>
      <td>1894</td>
      <td>1894-10-09</td>
      <td>Romance</td>
      <td>45</td>
      <td>USA</td>
      <td>None</td>
      <td>Alexander Black</td>
      <td>...</td>
      <td>Blanche Bayliss, William Courtenay, Chauncey D...</td>
      <td>The adventures of a female reporter in the 1890s.</td>
      <td>5.9</td>
      <td>154</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>tt0000574</td>
      <td>The Story of the Kelly Gang</td>
      <td>The Story of the Kelly Gang</td>
      <td>1906</td>
      <td>1906-12-26</td>
      <td>Biography, Crime, Drama</td>
      <td>70</td>
      <td>Australia</td>
      <td>None</td>
      <td>Charles Tait</td>
      <td>...</td>
      <td>Elizabeth Tait, John Tait, Norman Campbell, Be...</td>
      <td>True story of notorious Australian outlaw Ned ...</td>
      <td>6.1</td>
      <td>589</td>
      <td>$ 2250</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>tt0001892</td>
      <td>Den sorte drøm</td>
      <td>Den sorte drøm</td>
      <td>1911</td>
      <td>1911-08-19</td>
      <td>Drama</td>
      <td>53</td>
      <td>Germany, Denmark</td>
      <td>NaN</td>
      <td>Urban Gad</td>
      <td>...</td>
      <td>Asta Nielsen, Valdemar Psilander, Gunnar Helse...</td>
      <td>Two men of high rank are both wooing the beaut...</td>
      <td>5.8</td>
      <td>188</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>tt0002101</td>
      <td>Cleopatra</td>
      <td>Cleopatra</td>
      <td>1912</td>
      <td>1912-11-13</td>
      <td>Drama, History</td>
      <td>100</td>
      <td>USA</td>
      <td>English</td>
      <td>Charles L. Gaskill</td>
      <td>...</td>
      <td>Helen Gardner, Pearl Sindelar, Miss Fielding, ...</td>
      <td>The fabled queen of Egypt's affair with Roman ...</td>
      <td>5.2</td>
      <td>446</td>
      <td>$ 45000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>25.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>tt0002130</td>
      <td>L'Inferno</td>
      <td>L'Inferno</td>
      <td>1911</td>
      <td>1911-03-06</td>
      <td>Adventure, Drama, Fantasy</td>
      <td>68</td>
      <td>Italy</td>
      <td>Italian</td>
      <td>Francesco Bertolini, Adolfo Padovan</td>
      <td>...</td>
      <td>Salvatore Papa, Arturo Pirovano, Giuseppe de L...</td>
      <td>Loosely adapted from Dante's Divine Comedy and...</td>
      <td>7.0</td>
      <td>2237</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>31.0</td>
      <td>14.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>




```python
imdb.columns
```




    Index(['imdb_title_id', 'title', 'original_title', 'year', 'date_published',
           'genre', 'duration', 'country', 'language', 'director', 'writer',
           'production_company', 'actors', 'description', 'avg_vote', 'votes',
           'budget', 'usa_gross_income', 'worlwide_gross_income', 'metascore',
           'reviews_from_users', 'reviews_from_critics'],
          dtype='object')




```python
#remove values with no gross income data
imdb=imdb[imdb['usa_gross_income'].isna()==False]
imdb=imdb[imdb['worlwide_gross_income'].isna()==False]
imdb=imdb[imdb['budget'].isna()==False]
imdb.reset_index(drop=True, inplace=True)
```


```python
#Remove dollar sign from revenue and budget
usa=[]
worldwide=[]
budget=[]
for index, row in imdb.iterrows():
    try:
        usa.append(row['usa_gross_income'].split('$ ')[1])
    except:
        usa.append(0)
    try:
        worldwide.append(row['worlwide_gross_income'].split('$ ')[1])
    except:
        worldwide.append(0)
        
    budget.append(row['budget'].split(' ')[1])

imdb['usa_gross_income']=[int(x) for x in usa]
imdb['worlwide_gross_income']=[int(x) for x in worldwide]
imdb['budget']=[int(x) for x in budget]
```


```python
imdb
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>imdb_title_id</th>
      <th>title</th>
      <th>original_title</th>
      <th>year</th>
      <th>date_published</th>
      <th>genre</th>
      <th>duration</th>
      <th>country</th>
      <th>language</th>
      <th>director</th>
      <th>...</th>
      <th>actors</th>
      <th>description</th>
      <th>avg_vote</th>
      <th>votes</th>
      <th>budget</th>
      <th>usa_gross_income</th>
      <th>worlwide_gross_income</th>
      <th>metascore</th>
      <th>reviews_from_users</th>
      <th>reviews_from_critics</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>tt0010323</td>
      <td>Il gabinetto del dottor Caligari</td>
      <td>Das Cabinet des Dr. Caligari</td>
      <td>1920</td>
      <td>1920-02-27</td>
      <td>Fantasy, Horror, Mystery</td>
      <td>76</td>
      <td>Germany</td>
      <td>German</td>
      <td>Robert Wiene</td>
      <td>...</td>
      <td>Werner Krauss, Conrad Veidt, Friedrich Feher, ...</td>
      <td>Hypnotist Dr. Caligari uses a somnambulist, Ce...</td>
      <td>8.1</td>
      <td>55601</td>
      <td>18000</td>
      <td>8811</td>
      <td>8811</td>
      <td>NaN</td>
      <td>237.0</td>
      <td>160.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>tt0012190</td>
      <td>I quattro cavalieri dell'Apocalisse</td>
      <td>The Four Horsemen of the Apocalypse</td>
      <td>1921</td>
      <td>1923-04-16</td>
      <td>Drama, Romance, War</td>
      <td>150</td>
      <td>USA</td>
      <td>None</td>
      <td>Rex Ingram</td>
      <td>...</td>
      <td>Pomeroy Cannon, Josef Swickard, Bridgetta Clar...</td>
      <td>An extended family split up in France and Germ...</td>
      <td>7.2</td>
      <td>3058</td>
      <td>800000</td>
      <td>9183673</td>
      <td>9183673</td>
      <td>NaN</td>
      <td>45.0</td>
      <td>16.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>tt0017136</td>
      <td>Metropolis</td>
      <td>Metropolis</td>
      <td>1927</td>
      <td>1928-10-01</td>
      <td>Drama, Sci-Fi</td>
      <td>153</td>
      <td>Germany</td>
      <td>German</td>
      <td>Fritz Lang</td>
      <td>...</td>
      <td>Alfred Abel, Gustav Fröhlich, Rudolf Klein-Rog...</td>
      <td>In a futuristic city sharply divided between t...</td>
      <td>8.3</td>
      <td>156076</td>
      <td>6000000</td>
      <td>1236166</td>
      <td>1349711</td>
      <td>98.0</td>
      <td>495.0</td>
      <td>208.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>tt0021749</td>
      <td>Luci della città</td>
      <td>City Lights</td>
      <td>1931</td>
      <td>1931-04-02</td>
      <td>Comedy, Drama, Romance</td>
      <td>87</td>
      <td>USA</td>
      <td>English</td>
      <td>Charles Chaplin</td>
      <td>...</td>
      <td>Virginia Cherrill, Florence Lee, Harry Myers, ...</td>
      <td>With the aid of a wealthy erratic tippler, a d...</td>
      <td>8.5</td>
      <td>162668</td>
      <td>1500000</td>
      <td>19181</td>
      <td>46008</td>
      <td>99.0</td>
      <td>295.0</td>
      <td>122.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>tt0027977</td>
      <td>Tempi moderni</td>
      <td>Modern Times</td>
      <td>1936</td>
      <td>1937-03-12</td>
      <td>Comedy, Drama, Family</td>
      <td>87</td>
      <td>USA</td>
      <td>English</td>
      <td>Charles Chaplin</td>
      <td>...</td>
      <td>Charles Chaplin, Paulette Goddard, Henry Bergm...</td>
      <td>The Tramp struggles to live in modern industri...</td>
      <td>8.5</td>
      <td>211250</td>
      <td>1500000</td>
      <td>163577</td>
      <td>457688</td>
      <td>96.0</td>
      <td>307.0</td>
      <td>115.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>8121</th>
      <td>tt9214832</td>
      <td>Emma.</td>
      <td>Emma.</td>
      <td>2020</td>
      <td>2020-03-27</td>
      <td>Comedy, Drama</td>
      <td>124</td>
      <td>UK</td>
      <td>English</td>
      <td>Autumn de Wilde</td>
      <td>...</td>
      <td>Anya Taylor-Joy, Angus Imrie, Letty Thomas, Ge...</td>
      <td>In 1800s England, a well meaning but selfish y...</td>
      <td>6.7</td>
      <td>19858</td>
      <td>10000000</td>
      <td>10055355</td>
      <td>25659965</td>
      <td>71.0</td>
      <td>314.0</td>
      <td>188.0</td>
    </tr>
    <tr>
      <th>8122</th>
      <td>tt9354944</td>
      <td>Jexi</td>
      <td>Jexi</td>
      <td>2019</td>
      <td>2019-10-11</td>
      <td>Comedy, Romance</td>
      <td>84</td>
      <td>USA, Canada</td>
      <td>English</td>
      <td>Jon Lucas, Scott Moore</td>
      <td>...</td>
      <td>Adam Devine, Alexandra Shipp, Rose Byrne, Ron ...</td>
      <td>A comedy about what can happen when you love y...</td>
      <td>6.1</td>
      <td>17038</td>
      <td>5000000</td>
      <td>6546159</td>
      <td>9341824</td>
      <td>39.0</td>
      <td>234.0</td>
      <td>42.0</td>
    </tr>
    <tr>
      <th>8123</th>
      <td>tt9426210</td>
      <td>Weathering with You</td>
      <td>Tenki no ko</td>
      <td>2019</td>
      <td>2019-10-14</td>
      <td>Animation, Drama, Family</td>
      <td>112</td>
      <td>Japan, China</td>
      <td>Japanese</td>
      <td>Makoto Shinkai</td>
      <td>...</td>
      <td>Kotaro Daigo, Nana Mori, Shun Oguri, Sei Hirai...</td>
      <td>A high-school boy who has run away to Tokyo be...</td>
      <td>7.6</td>
      <td>16277</td>
      <td>11100000</td>
      <td>7798743</td>
      <td>193176979</td>
      <td>72.0</td>
      <td>177.0</td>
      <td>110.0</td>
    </tr>
    <tr>
      <th>8124</th>
      <td>tt9779516</td>
      <td>Cosa mi lasci di te</td>
      <td>I Still Believe</td>
      <td>2020</td>
      <td>2020-03-19</td>
      <td>Biography, Drama, Music</td>
      <td>116</td>
      <td>USA</td>
      <td>English</td>
      <td>Andrew Erwin, Jon Erwin</td>
      <td>...</td>
      <td>K.J. Apa, Britt Robertson, Nathan Parsons, Gar...</td>
      <td>The true-life story of Christian music star Je...</td>
      <td>6.5</td>
      <td>6196</td>
      <td>12000000</td>
      <td>9868521</td>
      <td>13681524</td>
      <td>41.0</td>
      <td>151.0</td>
      <td>52.0</td>
    </tr>
    <tr>
      <th>8125</th>
      <td>tt9825006</td>
      <td>Avant qu'on explose</td>
      <td>Avant qu'on explose</td>
      <td>2019</td>
      <td>2019-02-28</td>
      <td>Comedy</td>
      <td>108</td>
      <td>Canada</td>
      <td>French</td>
      <td>Rémi St-Michel</td>
      <td>...</td>
      <td>Étienne Galloy, Amadou Madani Tall, William Mo...</td>
      <td>The Third World War is on the horizon. Despite...</td>
      <td>6.6</td>
      <td>100</td>
      <td>3850000</td>
      <td>119894</td>
      <td>119894</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
<p>8126 rows × 22 columns</p>
</div>




```python
#split multiple items in column into a list of separate items
imdb['genre'] = (imdb['genre'].str.split(', '))
imdb['language'] = (imdb['language'].str.split(', '))
imdb['actors'] = (imdb['actors'].str.split(', '))
imdb['writer'] = (imdb['writer'].str.split(', '))
imdb['director'] = (imdb['director'].str.split(', '))
```


```python
# perform sentiment analysis on description and title
des_scores=[]
title_scores=[]
analyzer = SentimentIntensityAnalyzer()
for x in imdb['description']:
    try:
        des_scores.append(analyzer.polarity_scores(x)['compound'])
    except TypeError:
        des_scores.append(0)

for x in imdb['title']:
    try:
        title_scores.append(analyzer.polarity_scores(x)['compound'])
    except TypeError:
        title_scores.append(0)
imdb['description_score']=des_scores
imdb['title']=title_scores
imdb.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>imdb_title_id</th>
      <th>title</th>
      <th>original_title</th>
      <th>year</th>
      <th>date_published</th>
      <th>genre</th>
      <th>duration</th>
      <th>country</th>
      <th>language</th>
      <th>director</th>
      <th>...</th>
      <th>description</th>
      <th>avg_vote</th>
      <th>votes</th>
      <th>budget</th>
      <th>usa_gross_income</th>
      <th>worlwide_gross_income</th>
      <th>metascore</th>
      <th>reviews_from_users</th>
      <th>reviews_from_critics</th>
      <th>description_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>tt0010323</td>
      <td>0.0</td>
      <td>Das Cabinet des Dr. Caligari</td>
      <td>1920</td>
      <td>1920-02-27</td>
      <td>[Fantasy, Horror, Mystery]</td>
      <td>76</td>
      <td>Germany</td>
      <td>[German]</td>
      <td>[Robert Wiene]</td>
      <td>...</td>
      <td>Hypnotist Dr. Caligari uses a somnambulist, Ce...</td>
      <td>8.1</td>
      <td>55601</td>
      <td>18000</td>
      <td>8811</td>
      <td>8811</td>
      <td>NaN</td>
      <td>237.0</td>
      <td>160.0</td>
      <td>-0.4215</td>
    </tr>
    <tr>
      <th>1</th>
      <td>tt0012190</td>
      <td>0.0</td>
      <td>The Four Horsemen of the Apocalypse</td>
      <td>1921</td>
      <td>1923-04-16</td>
      <td>[Drama, Romance, War]</td>
      <td>150</td>
      <td>USA</td>
      <td>[None]</td>
      <td>[Rex Ingram]</td>
      <td>...</td>
      <td>An extended family split up in France and Germ...</td>
      <td>7.2</td>
      <td>3058</td>
      <td>800000</td>
      <td>9183673</td>
      <td>9183673</td>
      <td>NaN</td>
      <td>45.0</td>
      <td>16.0</td>
      <td>-0.7579</td>
    </tr>
    <tr>
      <th>2</th>
      <td>tt0017136</td>
      <td>0.0</td>
      <td>Metropolis</td>
      <td>1927</td>
      <td>1928-10-01</td>
      <td>[Drama, Sci-Fi]</td>
      <td>153</td>
      <td>Germany</td>
      <td>[German]</td>
      <td>[Fritz Lang]</td>
      <td>...</td>
      <td>In a futuristic city sharply divided between t...</td>
      <td>8.3</td>
      <td>156076</td>
      <td>6000000</td>
      <td>1236166</td>
      <td>1349711</td>
      <td>98.0</td>
      <td>495.0</td>
      <td>208.0</td>
      <td>0.6369</td>
    </tr>
    <tr>
      <th>3</th>
      <td>tt0021749</td>
      <td>0.0</td>
      <td>City Lights</td>
      <td>1931</td>
      <td>1931-04-02</td>
      <td>[Comedy, Drama, Romance]</td>
      <td>87</td>
      <td>USA</td>
      <td>[English]</td>
      <td>[Charles Chaplin]</td>
      <td>...</td>
      <td>With the aid of a wealthy erratic tippler, a d...</td>
      <td>8.5</td>
      <td>162668</td>
      <td>1500000</td>
      <td>19181</td>
      <td>46008</td>
      <td>99.0</td>
      <td>295.0</td>
      <td>122.0</td>
      <td>0.7845</td>
    </tr>
    <tr>
      <th>4</th>
      <td>tt0027977</td>
      <td>0.0</td>
      <td>Modern Times</td>
      <td>1936</td>
      <td>1937-03-12</td>
      <td>[Comedy, Drama, Family]</td>
      <td>87</td>
      <td>USA</td>
      <td>[English]</td>
      <td>[Charles Chaplin]</td>
      <td>...</td>
      <td>The Tramp struggles to live in modern industri...</td>
      <td>8.5</td>
      <td>211250</td>
      <td>1500000</td>
      <td>163577</td>
      <td>457688</td>
      <td>96.0</td>
      <td>307.0</td>
      <td>115.0</td>
      <td>0.0516</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>




```python
#get dummy variables for categories that need them
df = pd.get_dummies(imdb['genre'].apply(pd.Series).stack()).sum(level=0)
df2 = pd.get_dummies(imdb['language'].apply(pd.Series).stack()).sum(level=0)
df3 = pd.get_dummies(imdb['director'].apply(pd.Series).stack()).sum(level=0)
df4 = pd.get_dummies(imdb['writer'].apply(pd.Series).stack()).sum(level=0)
```


```python
df = df.add_prefix('genre_')
df2 = df2.add_prefix('langauge_')
df3 = df3.add_prefix('director_')
df4 = df4.add_prefix('writer_')
```


```python
#add similar column for merging
df['imdb_title_id']=imdb['imdb_title_id']
df2['imdb_title_id']=imdb['imdb_title_id']
df3['imdb_title_id']=imdb['imdb_title_id']
df4['imdb_title_id']=imdb['imdb_title_id']
```


```python
#merge all dataframes into one
dfs = [imdb,df,df2,df3,df4]
df_final = reduce(lambda left,right: pd.merge(left,right,on='imdb_title_id'), dfs)
```


```python
df_final.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>imdb_title_id</th>
      <th>title</th>
      <th>original_title</th>
      <th>year</th>
      <th>date_published</th>
      <th>genre</th>
      <th>duration</th>
      <th>country</th>
      <th>language</th>
      <th>director</th>
      <th>...</th>
      <th>writer_Zoë Lund</th>
      <th>writer_Àlex Pastor</th>
      <th>writer_Álex de la Iglesia</th>
      <th>writer_Álvaro Rodríguez</th>
      <th>writer_Élie Chouraqui</th>
      <th>writer_Émile Gaudreault</th>
      <th>writer_Émile Zola</th>
      <th>writer_Éric Rohmer</th>
      <th>writer_Éric Tessier</th>
      <th>writer_Éric Toledano</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>tt0010323</td>
      <td>0.0</td>
      <td>Das Cabinet des Dr. Caligari</td>
      <td>1920</td>
      <td>1920-02-27</td>
      <td>[Fantasy, Horror, Mystery]</td>
      <td>76</td>
      <td>Germany</td>
      <td>[German]</td>
      <td>[Robert Wiene]</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>tt0012190</td>
      <td>0.0</td>
      <td>The Four Horsemen of the Apocalypse</td>
      <td>1921</td>
      <td>1923-04-16</td>
      <td>[Drama, Romance, War]</td>
      <td>150</td>
      <td>USA</td>
      <td>[None]</td>
      <td>[Rex Ingram]</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>tt0017136</td>
      <td>0.0</td>
      <td>Metropolis</td>
      <td>1927</td>
      <td>1928-10-01</td>
      <td>[Drama, Sci-Fi]</td>
      <td>153</td>
      <td>Germany</td>
      <td>[German]</td>
      <td>[Fritz Lang]</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>tt0021749</td>
      <td>0.0</td>
      <td>City Lights</td>
      <td>1931</td>
      <td>1931-04-02</td>
      <td>[Comedy, Drama, Romance]</td>
      <td>87</td>
      <td>USA</td>
      <td>[English]</td>
      <td>[Charles Chaplin]</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>tt0027977</td>
      <td>0.0</td>
      <td>Modern Times</td>
      <td>1936</td>
      <td>1937-03-12</td>
      <td>[Comedy, Drama, Family]</td>
      <td>87</td>
      <td>USA</td>
      <td>[English]</td>
      <td>[Charles Chaplin]</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 11699 columns</p>
</div>




```python
# get a binary column to represent whether a movie was able to recoup its production costs
profitable = []
for index, row in df_final.iterrows():
    if row['worlwide_gross_income']>row['budget']:
        profitable.append(1)
    else:
        profitable.append(0)
df_final['profitable']=profitable
```


```python
df_final.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>imdb_title_id</th>
      <th>title</th>
      <th>original_title</th>
      <th>year</th>
      <th>date_published</th>
      <th>genre</th>
      <th>duration</th>
      <th>country</th>
      <th>language</th>
      <th>director</th>
      <th>...</th>
      <th>writer_Àlex Pastor</th>
      <th>writer_Álex de la Iglesia</th>
      <th>writer_Álvaro Rodríguez</th>
      <th>writer_Élie Chouraqui</th>
      <th>writer_Émile Gaudreault</th>
      <th>writer_Émile Zola</th>
      <th>writer_Éric Rohmer</th>
      <th>writer_Éric Tessier</th>
      <th>writer_Éric Toledano</th>
      <th>profitable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>tt0010323</td>
      <td>0.0</td>
      <td>Das Cabinet des Dr. Caligari</td>
      <td>1920</td>
      <td>1920-02-27</td>
      <td>[Fantasy, Horror, Mystery]</td>
      <td>76</td>
      <td>Germany</td>
      <td>[German]</td>
      <td>[Robert Wiene]</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>tt0012190</td>
      <td>0.0</td>
      <td>The Four Horsemen of the Apocalypse</td>
      <td>1921</td>
      <td>1923-04-16</td>
      <td>[Drama, Romance, War]</td>
      <td>150</td>
      <td>USA</td>
      <td>[None]</td>
      <td>[Rex Ingram]</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>tt0017136</td>
      <td>0.0</td>
      <td>Metropolis</td>
      <td>1927</td>
      <td>1928-10-01</td>
      <td>[Drama, Sci-Fi]</td>
      <td>153</td>
      <td>Germany</td>
      <td>[German]</td>
      <td>[Fritz Lang]</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>tt0021749</td>
      <td>0.0</td>
      <td>City Lights</td>
      <td>1931</td>
      <td>1931-04-02</td>
      <td>[Comedy, Drama, Romance]</td>
      <td>87</td>
      <td>USA</td>
      <td>[English]</td>
      <td>[Charles Chaplin]</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>tt0027977</td>
      <td>0.0</td>
      <td>Modern Times</td>
      <td>1936</td>
      <td>1937-03-12</td>
      <td>[Comedy, Drama, Family]</td>
      <td>87</td>
      <td>USA</td>
      <td>[English]</td>
      <td>[Charles Chaplin]</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 11700 columns</p>
</div>




```python
#get the feature and label sets
#remove all the features that wouldn't be known until after a movie is produced
X = df_final.drop(['imdb_title_id','title','original_title','date_published',
                   'genre', 'duration', 'country', 'language', 'director', 'writer',
                   'production_company', 'actors', 'description', 'avg_vote', 'votes',
                   'usa_gross_income', 'worlwide_gross_income', 'metascore','reviews_from_users', 
                   'reviews_from_critics', 'description_score'],axis=1)
y = df_final['profitable']
```


```python
corlist=pd.DataFrame(X.corrwith(y),columns=['coor'])
```


```python
corlist.reset_index(inplace=True)
```


```python
df1 = corlist[corlist['coor']!=0]
```


```python
X = X[df1['index']]
X.drop('profitable',axis=1,inplace=True)
```


```python
X.dropna(inplace=True)
```


```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

classifier = RandomForestClassifier()
classifier.fit(X_train,y_train)

model = SelectFromModel(classifier, prefit=True)
X = model.transform(X)
```


```python
# generate more samples from the data to balance out binary outcomes
from imblearn.over_sampling import SMOTE

oversample = SMOTE()
X, y = oversample.fit_resample(X, y)
```


```python
#get a training and test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=0)
```


```python
# standardize the data
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_val = sc.transform(X_val)
X_test = sc.transform(X_test)
```


```python
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier()
classifier.fit(X_train,y_train)
```




    RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                           criterion='gini', max_depth=None, max_features='auto',
                           max_leaf_nodes=None, max_samples=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_jobs=None, oob_score=False, random_state=None,
                           verbose=0, warm_start=False)




```python
# predict values using X test and the new model
y_pred = classifier.predict(X_val)
```


```python
#print the confusion matrix as well as multiple scores to evaluate the model
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, f1_score 
cm = confusion_matrix(y_val, y_pred)
print(cm)
print('Accuracy: ' + str(accuracy_score(y_val, y_pred)))
print('AUC: ' + str(roc_auc_score(y_val, y_pred)))
print('F1 Score: ' + str(f1_score(y_val, y_pred)))
```

    [[722 148]
     [519 427]]
    Accuracy: 0.6327092511013216
    AUC: 0.6406296323297126
    F1 Score: 0.5614727153188692
    


```python
#use grid search cv to run multiple random forest classifier models to find best hyperparameters
from sklearn.model_selection  import GridSearchCV

param_grid = { 
    'n_estimators': [50, 100, 250],
    'max_features': ['auto','sqrt']
}

CV_rfc = GridSearchCV(estimator=classifier, param_grid=param_grid, cv= 5,n_jobs=-1)
CV_rfc.fit(X_val, y_val)
print(CV_rfc.best_estimator_)
```

    RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                           criterion='gini', max_depth=None, max_features='auto',
                           max_leaf_nodes=None, max_samples=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=250,
                           n_jobs=None, oob_score=False, random_state=None,
                           verbose=0, warm_start=False)
    


```python
winsound.Beep(440,250)
```


```python
# predict values using X test and the new model
y_pred = CV_rfc.best_estimator_.predict(X_test)
```


```python
#print the confusion matrix as well as multiple scores to evaluate the model
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, f1_score 
cm = confusion_matrix(y_test, y_pred)
print(cm)
print('Accuracy: ' + str(accuracy_score(y_test, y_pred)))
print('AUC: ' + str(roc_auc_score(y_test, y_pred)))
print('F1 Score: ' + str(f1_score(y_test, y_pred)))
```

    [[667 264]
     [413 472]]
    Accuracy: 0.6272026431718062
    AUC: 0.6248836376655925
    F1 Score: 0.5823565700185072
    


```python
# perform multiple random permutations to find p-value 
from sklearn.model_selection import permutation_test_score
from sklearn.model_selection import StratifiedKFold

clf = CV_rfc.best_estimator_
cv = StratifiedKFold(2, shuffle=True, random_state=0)

score_orig, perm_scores_orig, pvalue_orig = permutation_test_score(
    clf, X_test, y_test, scoring="accuracy", cv=cv, n_permutations=100)
```


```python
pvalue_orig
```




    0.009900990099009901




```python
winsound.Beep(440,250)
```


```python

```
