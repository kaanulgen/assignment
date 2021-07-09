import json, string, re
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

#------------------------------------------ Functions ------------------------------------------#

def get_data():
    
    # Reading Meta Data
    with open('./data/meta.json','r') as f:
        meta = json.loads(f.read())

    # Normalizing meta data
    meta = pd.json_normalize(meta, record_path =['meta'])
    meta['name'] = meta['name'].str.lower()

    # cleaning text data
    meta["text"] = meta['name'] + " " + meta["category"] + " " + meta["subcategory"]
    meta["cleaned_text"] = meta['text'].astype(str).map(preprocess)

    return meta

def preprocess(text):
    
    # turkish stopwords
    stops = set(stopwords.words('turkish'))
    
    result=[]
    # remove punctuation
    text = re.sub(r'[' + string.punctuation + ']+', ' ', text.lower())
    
    # splitting strings into tokens 
    for token in word_tokenize(text):
        if token not in stops and len(token) > 2: # Token sequence length greater than 2.
            result.append(token)
    return " ".join(i for i in result if not i.isdigit()) # Also reject numbers


def transform_data(data):

    # Convert a collection of raw documents to a matrix of TF-IDF features
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(data['cleaned_text'])

    # Compute cosine similarity between samples in X and Y.
    # Cosine similarity, or the cosine kernel, computes similarity as the normalized dot product of X and Y
    # X = tfidf_matrix, Y = tfidf_matrix
    
    #cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    
    return cosine_sim


def recommend_products(id_, data, transform):

    # create a Pandas Series with indices of all the products present in my dataset.
    indices = pd.Series(data["productid"].index, data["productid"]).drop_duplicates()
    # get the index of the input product that is passed onto my recommend_products() function in the productid parameter.
    index = indices[id_]


    # here we store the Cosine Values of each product with respect to my input product.
    sim_scores = list(enumerate(transform[index]))

    # after getting the cosine values we sort them in reverse order.
    sim_scores = sorted(sim_scores,  key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[0:11] # top ten score
    
    # top 10 product sorted according to my cosine values in a list.
    product_indices = [i[0] for i in sim_scores] # index
    similarity_score = [i[1] for i in sim_scores] # score

    product_id = data['productid'].iloc[product_indices]
    product_name = data['name'].iloc[product_indices]

    recommendation_data = pd.DataFrame(columns=['productid', 'name', 'score',])
    #recommendation_data = pd.DataFrame(columns=['productid'])

    recommendation_data['productid'] = product_id
    recommendation_data['name'] = product_name
    recommendation_data['score'] = similarity_score

    recommendation_data = recommendation_data[recommendation_data.productid != id_]

    return recommendation_data # return the pandas dataFrame with the top 10 product recommendations.


def results(product_id):

    meta = get_data()
    
    transform_result = transform_data(meta)
    
    print("product_id: ", product_id)
    
    if product_id not in meta['productid'].unique():
        return 'Product not in Database'
    
    else:
        recommendations = recommend_products(product_id, meta, transform_result)
        return recommendations.to_dict('records')




