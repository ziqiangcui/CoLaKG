import pandas as pd
from itertools import combinations
import gc
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import dgl
import random
import torch
from collections import Counter
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import copy
import numpy as np
import asyncio
import aiohttp
import nest_asyncio
import json
import time
import gc
import pickle
import dgl
import random
import torch
from collections import Counter
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import copy
import numpy as np
import asyncio
import aiohttp
import nest_asyncio
import json
import time
import re
import ssl
from itertools import combinations
from collections import deque
import aiofiles
import os

data_type = "item" # item or user
data_name = 'ml-1m' # choose a dataset

input_path = f"../data/{data_name}/llm_input_{data_type}.json"    
write_path = f"../data/{data_name}/llm_response_{data_type}.json" 

if data_type == "item":
    if data_name == "ml-1m":
        system_input = "Assume you are an expert in movie recommendation. You will be given a certain movie with its first-order information (in the form of triples) and some second-order relationships (movies related to this movie). Please complete the missing knowledge, summarize the movie and analyze what kind of users would like it. Your response should be a coherent paragraph and no more than 200 words."
    if data_name == "mind":
        system_input = "Assume you are an expert in news recommendation. You are given a piece of news. I will give you the basic information of the news and some related news that has the same category or key words with the current news.  Please help me summarize the news and related news, and analyze what kind of users would like reading this type of news. please give your answer in a coherent paragraph under 150 words."
    elif data_name == "lastfm":
        system_input = "Assume you are an expert in music recommendation. There is an artist. You already know : 1)the tags that some users have assigned to this artist on Last.fm; 2) Some related artists that share the same tags with this artist. Based on this information and your world knowledge, please summarize the music style and characteristics of this artist, and analyze what kind of users would enjoy listening to their works. Your response should be a coherent paragraph and no more than 150 words."     
elif data_type == "user":
    if data_name == "ml-1m":
        system_input = "Assume you are an expert in movie recommendation with access to a viewer's movie-watching history, where each entry is formatted as (movie_name: genres: xx, director: xx, main actors: xx, overview: xx). Please analyze and summarize this user's viewing preferences from the aspects of movie genres, directors, and actors. Your response should be a coherent and fluent paragraph, not exceeding 100 words."
    if data_name == "mind":
        system_input = "Assume you are an expert in news recommendation with access to a user's news-reading history, where each entry is formatted as (news_name: category: xx, subcategory: xx, abstract: xx). Please analyze and summarize this user's news preferences. Your response should be a coherent and fluent paragraph, not exceeding 100 words."
    elif data_name == "lastfm":
        system_input = "Assume you are an expert in music recommendation with access to a listener's artist-listening history, where each entry is formatted as (artist_name: tags assigned by users). Please analyze what type of artist this user likes. Your response should be a coherent and fluent paragraph, not exceeding 100 words."

# Load the dictionary from the file
with open(input_path, 'r') as f:
    input_dic = json.load(f)

nest_asyncio.apply()

api_key = "xxxx" # input your own api_key
base_url = "https://api.deepseek.com/v1/chat/completions"

requests = [
    {"role": "user", "content": request, "item_key":key}
    for key, request in input_dic.items()
]


concurrent_limit = 100 
semaphore = asyncio.Semaphore(concurrent_limit)

batch_size = 500

max_retries = 5

batch_delay = 2

ssl_context = ssl.create_default_context()

async def fetch_response(session, request, retries=0):
    async with semaphore:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system_input},
                request,
            ],
            "temperature": 0.0,
            "top_p": 0.001,
            "stream": False
        }
        try:
            async with session.post(base_url, headers=headers, data=json.dumps(payload), ssl=ssl_context, timeout=aiohttp.ClientTimeout(total=120)) as response:
                if response.status == 200:
                    result = await response.json()
                    return result['choices'][0]['message']['content']
                else:
                    print(f"Error: Received status code {response.status}")
                    return None
        except (aiohttp.ClientConnectorError, aiohttp.ClientConnectorSSLError, asyncio.TimeoutError) as e:
            if retries < max_retries:
                print(f"Retrying... ({retries + 1}/{max_retries}) due to {e}")
                return await fetch_response(session, request, retries + 1)
            else:
                print(f"Failed after {max_retries} retries due to {e}.")
                return None

async def process_batch(session, batch):
    tasks = [fetch_response(session, request) for request in batch]
    return await asyncio.gather(*tasks)


async def write_responses_to_file(responses, file_path):
    async with aiofiles.open(file_path, 'w') as f:
        await f.write(json.dumps(responses, indent=4))

async def main(file_path):
    start_time = time.time()  
    responses = {} 

    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=concurrent_limit, ssl=ssl_context)) as session:
        for i in range(0, len(requests), batch_size):
          
            print(f"Processing batch {i // batch_size + 1}")
            batch = requests[i:i + batch_size]
            batch_responses = await process_batch(session, batch)
            for request, response in zip(batch, batch_responses):
                item_key = request['item_key']
                responses[item_key] = response

            await write_responses_to_file(responses, file_path)
            print(f"Batch {i // batch_size + 1} completed, sleeping for {batch_delay} seconds")
            await asyncio.sleep(batch_delay)

    end_time = time.time()  
    elapsed_time = end_time - start_time  
    print(f"Total time taken: {elapsed_time} seconds")
    return responses


if os.path.exists(write_path):
    os.remove(write_path)
    print(f"Deleted existing file: {write_path}")

responses = asyncio.run(main(write_path))
print(".......", len(list(responses.keys())), ".......")
