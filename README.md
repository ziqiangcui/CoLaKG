# Introduction
This is the Pytorch implementation for our SIGIR'25 paper CoLaKG: **Comprehending Knowledge Graphs with Large Language Models for Recommender Systems**.

## 🚀 Update
**[2025-09-02]**
We have fixed some broken files: mind_embeddings_simcse_kg_user.pt and mind_embeddings_simcse_kg.pt. Their sizes exceed the upload limit. Please find them in 🔗 [Google Drive Download Link](https://drive.google.com/drive/u/0/folders/14W3TpbO1k9XZ_13gilcdzwTAU5jn8Qel).

**[2025-09-01]** 
We have uploaded the data preprocessing code (in the `data_preprocess` folder) and supplemented the corresponding knowledge source data (in the `data` folder). Some files (e.g. llm_input_user/item.json, original knowledge data) exceed the size limit for GitHub upload. Please download them from the Google Drive link below:

🔗 [Google Drive Download Link](https://drive.google.com/drive/u/0/folders/14W3TpbO1k9XZ_13gilcdzwTAU5jn8Qel)

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
