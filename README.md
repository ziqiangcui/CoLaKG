## Introduction
This is the Pytorch implementation for our SIGIR'25 paper: **Comprehending Knowledge Graphs with Large Language Models for Recommender Systems**.

## Environment Dependencies
You can refer to `requirements.txt` for the experimental environment we set to use.

## Run CoLaKG

We use API calls to interact with the LLM to obtain KG comprehension. Batch processing is employed to accelerate this process. Related code is in llm_code/llm_request_api.py. For your convenience, we also have provided the responses of the LLM (i.e., KG subgraph comprehension) for all datasets in the "data" directory.

We use a text embedding model to convert the LLM's responses into embeddings. You can refer to this part of the code in llm_code/get_text_embedding.py. We also provide the generated embeddings in the "data" directory.

To train the recommendation model with the generated semantic embeddings, simply use:

`cd rec_code`

`sh train_movielens.sh`

`sh train_lastfm.sh`

`sh train_mind.sh`