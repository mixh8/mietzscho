# import pandas
import pandas as pd

#Ask for user input
movie = input("Input the desired movie:\n")



# Similarity matching function
# Input additional info in case of duplocate movies


# remove column limit for printing
pd.set_option('max_columns', None)

# load movies metadata
metadata = pd.read_csv("movies_metadata.csv", low_memory = False)
metadata = metadata[metadata.revenue > 0]

# for index in metadata.index:
#     metadata.loc[index, "release_date"] = str(metadata.loc[index, "release_date"])[:4]



# return the first three rows; .head() 5 rows
metadata.head(3)

## simple recommender

# calculate the mean of vote average column
C = metadata["vote_average"].mean()

# calculate the minimum number of votes needed to be listed in the chart, m
m = metadata["vote_count"].quantile(0.90)

# filter all qualified movies into a new dataframe
q_movies = metadata.copy().loc[metadata["vote_count"] >= m]
    
# we can see that the two dataframes now have different sizes because some movies got "eliminated"
compare = [q_movies.shape, metadata.shape]

# function that computes weighted rating of each movie
def weighted_rating(x, m=m, C=C):
    v = x["vote_count"]
    R = x["vote_average"]
    
    # calculations based on IMDB formula
    return(v/(v+m) * R + m/(m+v) * C)

# define a new score feature and calculate its value using weighted_rating
q_movies["score"] = q_movies.apply(weighted_rating, axis = 1)

# sort movies based on calculated score
q_movies = q_movies.sort_values("score", ascending = False)

# print the top 20 movies
top20 = q_movies[["title", "vote_count", "vote_average", "score"]].head(50)

# content-based recommender

# return plot overviews of 5 first movies
metadata["overview"].head()

# import TfIdfVectorizer from sickit-learn
from sklearn.feature_extraction.text import TfidfVectorizer

# define a TF-IDF Vectorizer Object. Remove all english stop words
tfidf = TfidfVectorizer(stop_words = "english")

# replace NaN with empty string
metadata["overview"] = metadata["overview"].fillna("")

# construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(metadata["overview"])

# return the shape of the TF-IDF matrix
tfidf_matrix.shape

# array mapping from feature integer indices to feature name
names = tfidf.get_feature_names()[5000:5010]
print("Processing ...")

# import linear kernel
from sklearn.metrics.pairwise import linear_kernel

# compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# cosine similarity of first movie
# cosine_sim[1]

# construct a reverse map of indices and movie titles
indices = pd.Series(metadata.index, index = metadata["title"]).drop_duplicates()
    

#function that takes as an input the movie title and returns a list of the most similar movies
def get_recommendations(title, cosine_sim = cosine_sim):
    
    # get index of movie that matches title
    idx = indices[title]

    # get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # sort the pairs based on similarity score
    sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse = True)
    
    # get the scores of 10 most similar movies
    sim_scores = sim_scores[1:11]
    
    # get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    
    # return top 10 most similar movies
    return metadata["title"].iloc[movie_indices]


# Credits, Genres, and Keyword Based Recommender

# load keywords and credits
crdts = pd.read_csv("credits.csv")
kwrds = pd.read_csv("keywords.csv")


# convert IDs to int. Required for merging
kwrds["id"] = kwrds["id"].astype("int")
crdts["id"] = crdts["id"].astype("int")
metadata["id"] = metadata["id"].astype("int")

# merge keywords and credits into main metadata dataframe
metadata = metadata.merge(crdts, on = "id")
metadata = metadata.merge(kwrds, on = "id")

# print the first two movies of newly merged dataframe
# print(metadata.head(2))

# parse the stringified features into their corresponding python objects
from ast import literal_eval

features = ["cast", "crew", "keywords", "genres"]
for feature in features:
    metadata[feature] = metadata[feature].apply(literal_eval)
    
# import NumPy
import numpy as np

def get_director(x):
    for i in x:
        if i["job"] == "Director":
            return i["name"]
    return np.nan

def get_list(x):
    if isinstance(x, list):
        names = [i["name"] for i in x]
        
        # check if more than 3 elements are in the list. If yes, return the first three, if no the entire list
        if len(names) > 3:
            names = names[:3]
        return names
    
    # return empty list in case of missing/malformed data
    return []

# define new director, cast, genres and keywordseatures that are in suitable form
metadata["director"] = metadata["crew"].apply(get_director)

features = ["cast", "keywords", "genres"]
for feature in features:
    metadata[feature] = metadata[feature].apply(get_list)

# print the new features of the first 3 films
#print(metadata[["title", "cast", "director", "keywords", "genres"]].head(3))

# function to convert all strings to lowercase and strip spaces
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        #check if rector exists, if not return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ""
        
# apply the clean_data function to our features
features = ["cast", "keywords", "director", "genres"]

for feature in features:
    metadata[feature] = metadata[feature].apply(clean_data)

# create metadata soup that we will feed to vectorizer
def create_soup(x):
    return " ".join(x["keywords"]) + " " + " ".join(x["cast"]) + " " + x["director"] + " " + " ".join(x["genres"])

# create a new soup feature
metadata["soup"] = metadata.apply(create_soup, axis = 1)

# print first two lines of soup
#print(metadata["soup"].head(2))

# import countVectorizer and create the count matrix
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words = "english")
count_matrix = count.fit_transform(metadata["soup"])

#print(count_matrix.shape)

# compute the cosine similarity matrix based on the count_matrix
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

# reset index of main dataframe and nstruct reverse mapping as before
metadata = metadata.reset_index()
indices = pd.Series(metadata.index, index = metadata["title"])


series = get_recommendations(movie, cosine_sim2)
print("---------------------\n\n")
for movie in series:
    print("Title: {}\n".format(movie))
    print("Overview: {}\n\n".format(metadata.at[indices[movie], "overview"]))
    print("---------------------\n\n")

