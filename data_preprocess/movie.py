import pandas as pd
import numpy as np
import copy
import dgl
import random
import torch
from collections import Counter
from sklearn.preprocessing import LabelEncoder
import json

path = "/Users/Desktop/data/ml-1m/" # your data path

movies = pd.read_csv(path + "ml1m_extended_movie.csv")
# the file of "ml1m_extended_movie.csv" is obtained by combining original ml-1m and meta data from the web.


def create_kg_from_data(data):
    node_encoder = LabelEncoder()
    relation_encoder = LabelEncoder()
    
    nodes = pd.concat([data['head'], data['tail']]).unique()
    node_encoder.fit(nodes)
    head = node_encoder.transform(data['head'])
    tail = node_encoder.transform(data['tail'])

    relations = data['relation'].unique()
    relation_encoder.fit(relations)
    relation = relation_encoder.transform(data['relation'])
    g = dgl.heterograph({
        ('node', 'relation', 'node'): (head, tail)
    })
    
    g.edges['relation'].data['etype'] = torch.tensor(list(relation))  
    return g, node_encoder, relation_encoder


def node2vec_walk(g, start_node, K):
    walk_2nd_neighbor_dic = {}
    walk = [start_node]
    visited_neighbors = set()
    visited_second_neighbors = set()
    
    one_hop_neighbors = g.successors(start_node).tolist()
    
    for neighbor in one_hop_neighbors:
        walk_2nd_neighbor_dic[neighbor] = []
        walk.append(neighbor)
        visited_neighbors.add(neighbor)
        
        second_order_neighbors = g.successors(neighbor).tolist()
        second_order_neighbors = [n for n in second_order_neighbors if n != start_node]
        if len(second_order_neighbors) > K:
            sampled_neighbors = np.random.choice(second_order_neighbors, K, replace=False)
        else:
            sampled_neighbors = second_order_neighbors
        
        for second_neighbor in sampled_neighbors:
            walk.append(second_neighbor)
            visited_second_neighbors.add(second_neighbor)
            walk_2nd_neighbor_dic[neighbor].append(second_neighbor)
    
    return walk_2nd_neighbor_dic


def get_edge_types(g, walk):
    edge_types = []
    for i in range(len(walk) - 1):
        src, dst = walk[i], walk[i + 1]
        eid = g.edge_ids(src, dst)
        edge_type = g.edata['etype'][eid]
        edge_types.append(edge_type.item())
    return edge_types


movies['genres'] = movies['genres'].combine_first(movies['Genres'])  
movies['release_date'] = pd.to_datetime(movies['release_date'])
movies['release_date'] = movies['release_date'].dt.to_period('M')
movies['release_date'] = movies['release_date'].combine_first(movies['Year']) 
movies['director'] = movies['director'].fillna("unknown")
movies['actors'] = movies['actors'].fillna("unknown")
movies['overview'] = movies['overview'].fillna("unknown")
movies['writer'] = movies['writer'].fillna("unknown")

movie_1stneighbor_text_dic = {}
triples = [] 
for index, row in movies.iterrows():
    title = row['Title'] 
    movie_id = row['MovieID']
    genres = row['genres']  
    director = row['director']
    actors = row['actors'] 
    original_language = row['original_language']
    release_date = str(row['release_date'])
    vote_average = str(row['vote_average'])

    triples.append((title, 'was directed by', director))
    triples.append((director, 'directed', title))
    genre_list = sorted(genres.split('|'))
    if len(genre_list) == 1:
        triples.append((title, 'has genre', genre_list[0])) 
        triples.append((genre_list[0], 'is the genre of', title))  
    else:   
        for i in range(len(genre_list)):
            for j in range(i + 1, len(genre_list)):
                genre1 = genre_list[i]
                genre2 = genre_list[j]
                triples.append((title, 'has genres', f"{genre1} and {genre2}"))
                triples.append((f"{genre1} and {genre2}", 'are the genres of of', title))

    for actor in actors.split('|')[:3]:
        triples.append((title, 'stars', actor)) 
        triples.append((actor, 'starred in', title)) 

    description = (
        f'{title}, directed by {director}, is an {original_language}-language film released in {release_date}. '
        f'The movie falls under the genres of {", ".join(genres.split("|"))}. The main cast of this movie includes {", ".join(actors.split("|"))}.'
    )
    # print(description)

    movie_1stneighbor_text_dic[movie_id] = description

data = pd.DataFrame(triples, columns=['head', 'relation', 'tail'])
data = data.dropna()


graph, node_encoder, relation_encoder = create_kg_from_data(data)
movie_id_name_map = movies.set_index('MovieID')['Title'].to_dict()
movie_2ndneighbor_text_dic = {}
cnt = 0
for item_id in movie_id_name_map.keys():
    cnt += 1
    if cnt % 100 == 0:
        print(cnt)
    total_text = []
    direct_text = []
    act_text = []
    item_name = movie_id_name_map[item_id]
    item_node_id = node_encoder.transform([item_name])[0]
    movie_2ndneighbor_text_dic[item_name] = []
    cur_dic = node2vec_walk(graph, item_node_id, K=10)

    same_genre_movies = []
    
    for neighbor_id in cur_dic.keys():
        
        relation = get_edge_types(graph, walk=[item_node_id, neighbor_id])
        # to text
        neighbor = node_encoder.inverse_transform([neighbor_id])[0]
        relation = relation_encoder.inverse_transform(relation)[0]

        second_order_neighbors = cur_dic[neighbor_id]
        second_order_neighbors_name_list = []
        for second_order_neighbor_id in second_order_neighbors:
            second_order_neighbor = node_encoder.inverse_transform([second_order_neighbor_id])[0]
            second_order_neighbors_name_list.append(second_order_neighbor)
        
        movies_2ndneighbor = ", ".join(second_order_neighbors_name_list)

        if relation == "has genre" or relation == "has genres":
            same_genre_movies += second_order_neighbors_name_list
    
        elif relation == "was directed by":
            text = f"The director of {item_name}, {neighbor}, also directed {movies_2ndneighbor}."
            direct_text.append(text)
     
        elif relation == "stars":
            text = f"The lead actor of this movie, {neighbor}, also starred in {movies_2ndneighbor}."
            act_text.append(text)
        else:
            continue
        
    if len(same_genre_movies) > 10:
        same_genre_movies = np.random.choice(same_genre_movies, 10, replace=False)
    same_genre_movies = ", ".join(same_genre_movies)
    genre_text = f"Movies in the same/similar genres as {item_name} also include: < {same_genre_movies} > ."
    total_text.append(genre_text)
    total_text.append(" ".join(direct_text))
    total_text.append(" ".join(act_text) )
    
    movie_2ndneighbor_text_dic[item_id] = " ".join(total_text) 
    


# if data_type == "item":
#     if data_name == "ml-1m":
#         system_input = "Assume you are an expert in movie recommendation. You will be given a certain movie with its first-order information (in the form of triples) and some second-order relationships (movies related to this movie). Please complete the missing knowledge, summarize the movie and analyze what kind of users would like it. Your response should be a coherent paragraph and no more than 200 words."
#     if data_name == "mind":
#         system_input = "Assume you are an expert in news recommendation. You are given a piece of news. I will give you the basic information of the news and some related news that has the same category or key words with the current news.  Please help me summarize the news and related news, and analyze what kind of users would like reading this type of news. please give your answer in a coherent paragraph under 150 words."
#     elif data_name == "lastfm":
#         system_input = "Assume you are an expert in music recommendation. There is an artist. You already know : 1)the tags that some users have assigned to this artist on Last.fm; 2) Some related artists that share the same tags with this artist. Based on this information and your world knowledge, please summarize the music style and characteristics of this artist, and analyze what kind of users would enjoy listening to their works. Your response should be a coherent paragraph and no more than 150 words."
        
# elif data_type == "user":
#     if data_name == "ml-1m":
#         system_input = "Assume you are an expert in movie recommendation with access to a viewer's movie-watching history, where each entry is formatted as (movie_name: genres: xx, director: xx, main actors: xx, overview: xx). Please analyze and summarize this user's viewing preferences from the aspects of movie genres, directors, and actors. Your response should be a coherent and fluent paragraph, not exceeding 100 words."
#     if data_name == "mind":
#         system_input = "Assume you are an expert in news recommendation with access to a user's news-reading history, where each entry is formatted as (news_name: category: xx, subcategory: xx, abstract: xx). Please analyze and summarize this user's news preferences. Your response should be a coherent and fluent paragraph, not exceeding 100 words."
#     elif data_name == "lastfm":
#         system_input = "Assume you are an expert in music recommendation with access to a listener's artist-listening history, where each entry is formatted as (artist_name: tags assigned by users). Please analyze what type of artist this user likes. Your response should be a coherent and fluent paragraph, not exceeding 100 words."
    

def prompt_generation(id, movie):
    
    movie_basic_info = movie_1stneighbor_text_dic[id]
    movie_higher_info = movie_2ndneighbor_text_dic[id]
    
    analysis = f"""Assume you are an expert in movie recommendation. Now you are given a movie titled {movie}. 
    
    - The basic information from its first-degree neighbors in the movie knowledge graph includes:  {movie_basic_info} 
    - In addition, the sampled second-order information from the knowledge graph is: {movie_higher_info}

    Now, please calibrate and complete the above knowledge, and describe the movie in a coherent paragraph. Your answer should include the movie's title, genres, director, main actors, and any other information you deem helpful for the recommendation system. Please limit your answer to within 200 words.
    """
    return analysis



question_dic = {}
for movie_id, name in movie_id_name_map.items():
    prompt = prompt_generation(movie_id, name)        
    question_dic[movie_id] = prompt

print("The number of requests is: ", len(list(question_dic.keys())))
with open('/Users/Desktop/data/ml-1m/llm_input_item.json', 'w') as f:
    json.dump(question_dic, f)