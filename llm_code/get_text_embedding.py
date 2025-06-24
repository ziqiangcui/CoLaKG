from transformers import AutoTokenizer, AutoModel
import torch
import json

text_answer_path = "data/mind/llm_response_item.json"  # the file path of the LLM's textual output.
text_embedding_path_to_write = "data/mind/mind_embeddings_simcse_kg.pt" # generating the text embeddings w.r.t. LLM's textual output.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-roberta-large")
model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-roberta-large")

with open(text_answer_path, 'r') as file:
    item_txt_dic = json.load(file)
    
item_num = len(set(item_txt_dic.keys()))
texts = [item_txt_dic[str(i)] for i in range(item_num)]
batch_size = 64
all_embeddings = []
model.to(device)

for i in range(0, len(texts), batch_size):
    print(i)
    batch_texts = texts[i:i + batch_size]
    
    inputs_simcse = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
    
    inputs_simcse = {key: value.to(device) for key, value in inputs_simcse.items()}
    
    with torch.no_grad():
        embeddings_simcse = model(**inputs_simcse, output_hidden_states=True, return_dict=True).pooler_output
    
    all_embeddings.append(embeddings_simcse.cpu())


all_embeddings = torch.cat(all_embeddings, dim=0)

torch.save(all_embeddings, text_embedding_path_to_write)
print("inference over", all_embeddings.shape)
