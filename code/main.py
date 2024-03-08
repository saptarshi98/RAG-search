import os

from datasets import load_dataset
from pinecone import Pinecone, PodSpec
import ast
from openai import OpenAI
from tqdm.auto import tqdm
import pandas as pd
from utils import Utils

API_KEY = '44ac9d5a-f2e9-48c2-9f12-039a04bcaca8'
OPENAI_API_KEY= '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'


#Setting up Pinecone DB connection

util = Utils(API_KEY, OPENAI_API_KEY)
pinecone = Pinecone(api_key = util.pinecone_api_key)

INDEX_NAME = util.create_index_name('dl-ai', util.openai_api_key)
if INDEX_NAME in [index.name for index in pinecone.list_indexes()]:
    pinecone.delete_index(INDEX_NAME)


# if INDEX_NAME in [index for index in pinecone.list_indexes()]:
#    pinecone.delete_index(INDEX_NAME)

pinecone.create_index(name=INDEX_NAME,
                      dimension=1536,
                      metric='cosine',
                      spec=PodSpec(
                         environment='gcp-starter'
                      ))

index = pinecone.Index(INDEX_NAME)


#Load dataset
max_articles_num = 10000
article = pd.read_csv('..\data\wiki.csv', nrows=max_articles_num)

#Upsert the Embeddings of the articles to Pinecone DB
prepped = []

for i, row in tqdm(article.iterrows(), total=article.shape[0]):
    meta = ast.literal_eval(row['metadata'])
    prepped.append({'id':row['id'], 
                    'values':ast.literal_eval(row['values']), 
                    'metadata':meta})
    if len(prepped) >= 250:
        index.upsert(prepped)
        prepped = []

openai = OpenAI(api_key=OPENAI_API_KEY)

def get_embeddings(query, model="text-embedding-ada-002"):
    return openai.embeddings.create(input = query, model = model)

#Query & Its Embedding
query = "Can you please write a poem for me?"
emb = get_embeddings([query])

#Fetch similar vector from pinecone
matched_context = index.query(vector=emb.data[0].embedding, top_k = 3, include_metadata=True)
conext_text = [match['metadata']['text'] for match in matched_context['matches']]

prompt_start = 'Answer the question based on the conext below. \n \n' + 'Context: '

prompt_end = f'\n Question: {query} \n Answer: '

prompt = (prompt_start +
           '\n\n---n\n'.join(conext_text)
          + prompt_end)


result = openai.completions.create(
        model='gpt-3.5-turbo-instruct',
        prompt=prompt,
        temperature=0,
        max_tokens=650,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
)

print("="*100)
print(result.choices[0].text)