import json
import wikipediaapi
import requests
import os
import dotenv
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
from dataScraper import datacollecter

dotenv.load_dotenv()

DOCUMENT_PATH=os.getenv("TRAIN_DOCUMENT")
old_DOCUMENT_PATH="/home/walkiiiy/ChatTB/Bird_dev/dev_documentsV1"

class DocumentSimilarity:
    def __init__(self, model_name='paraphrase-MiniLM-L6-v2', collection_name='document_similarity'):
        # Initialize Chroma client and collection
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
            )

        # Initialize the SentenceTransformer model
        self.model = SentenceTransformer(model_name)

    def encode_text(self, text):
        """Encode the text into vector representation."""
        return self.model.encode([text])[0]  # Return as a numpy array

    def add_to_chroma_collection(self, texts, ids):
        """Add documents and their embeddings to Chroma collection."""
        embeddings = [self.encode_text(text) for text in texts]
        self.collection.add(
            documents=texts,
            metadatas=[None] * len(texts),  # No additional metadata
            ids=ids,
            embeddings=embeddings
        )

    def calculate_similarity(self, query_vector):
        """Calculate similarity between the query vector and documents in the collection."""
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=1,  # Only need the closest match
        )
        return results['distances'][0][0], results['documents'][0][0]  # Distance and document

    def get_distance(self,query,document):
        """Judge if the selected word and reason are related to the document."""
        # Prepare texts to encode
        document_text = document['summary'] + " " + document['text']

        # Add the vectors for selectedWord, reason, and document to the Chroma collection
        texts_to_index = [document_text]
        ids = ['document']  # Unique IDs for the texts, DO
        self.add_to_chroma_collection(texts_to_index, ids)

        # Query the vector for selectedWord and reason
        query_vector = self.encode_text(query)

        # Compute the similarity between selectedWord, reason, and document
        distance, res_document = self.calculate_similarity(query_vector)
        self.collection.delete('document')
        # Compare distances to the threshold
        # if selectedWord_distance > threshold and reason_distance > threshold:
        #     return "The selected word and reason are not related to the document."
        # else:
        #     return "The selected word and reason are related to the document."
        return distance,res_document

class WikiSearcher:
    def __init__(self):
        pass
    def wiki_search(self,query:str):# handel single query
        # MediaWiki API search endpoint
        search_url = 'https://en.wikipedia.org/w/api.php'
        user_agent = 'MyWikiSearchBot/1.0 (https://example.com/contact)'
        params = {
            'action': 'query',
            'list': 'search',
            'srsearch': query,
            'format': 'json',
            'utf8': 1,
            'srlimit':3,  # Limit to top 3 search results
        }
        response = requests.get(search_url, params=params, headers={'User-Agent': user_agent})
        data = response.json()

        search_results = data.get('query', {}).get('search', [])
        
        wiki = wikipediaapi.Wikipedia(
            language='en',
            user_agent=user_agent
        )
        if not search_results:
            return query,None,None
        

        page = wiki.page(search_results[0]['title'])    
        if page.exists():
            return page.title, page.summary, page.text
        else:
            return page.title, None, None


    def update_json_with_wiki_docs(self,json_file_path):
        print('processing',json_file_path)
        # Read the existing JSON file
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        
        # For each entry in the list, perform the wiki search and update
        for entry in data:
            if "document" in entry.keys():
                print(json_file_path," processed,skip")
                return
            word = entry["selectedWord"]
            title, summary, text = self.wiki_search(word)
            
            # Update the entry with the wiki document or null if not found
            entry["document"] = {
                "title": title,
                "summary": summary,
                "text": text if text else None
            }
        
        # Write the updated data back to the JSON file
        with open(json_file_path, 'w') as file:
            json.dump(data, file, indent=4)



# scrap pages
# json_file=[]
# collector=datacollecter()
# for file in os.listdir(old_DOCUMENT_PATH+'/Backgrounds'):
#     if file=="refined":
#         continue
#     json_file=os.path.join(old_DOCUMENT_PATH,'Backgrounds',file)
#     with open(json_file) as f:
#         documents=json.load(f)
#     for item in documents:
#         query=item["selectedWord"] #+" "+file
#         print(query+" is being collected")
#         scrap_res=collector.get_doc(query)
#         item["document"]=scrap_res
#     with open("/home/walkiiiy/ChatTB/dev_documents"+file,'w')as f:
#         json.dump(documents,f,indent=4)
#     print("result saved")
#     break
with open('/home/walkiiiy/ChatTB/dev_documentscodebase_community.json') as f:
    content=json.load(f)

with open("/home/walkiiiy/ChatTB/dev_documentscodebase_community.json","w") as f:
    json.dump(content,f,indent=4)


# search
# searcher=WikiSearcher()
# for file in os.listdir(DOCUMENT_PATH+'/Backgrounds'):
#     if file=="refined":
#         continue
#     json_file_path = os.path.join(DOCUMENT_PATH,'Backgrounds',file)
#     searcher.update_json_with_wiki_docs(json_file_path)




# compare
# comparer=DocumentSimilarity()
# for file in os.listdir(DOCUMENT_PATH+'/Backgrounds'):
#     if file=='refined':
#         continue
#     json_file_path = os.path.join(DOCUMENT_PATH,'Backgrounds',file)
    
#     refined_data=[]

#     with open(json_file_path) as f:
#         data=json.load(f)

#     for entry in data:
#         selectedWord = entry["selectedWord"]
#         reason=entry["reason"]
#         document=entry["document"]
#         if not document["summary"]:
#             continue
#         print(f"selectedWord:{selectedWord}\nreason:{reason}\nducument:{document["summary"][:100]}\n")
#         print('processing similarity....')

#         distance,res_document=comparer.get_distance(selectedWord+":"+reason,document)
        
#         entry["distance"]=distance        
#         if distance<=0.5:
#             refined_data.append(entry)
#         print(f"distance:{distance}\n------------------------\n")
        
#     with open(os.path.join(DOCUMENT_PATH,'Backgrounds','refined',file), 'w') as f:
#         print(os.path.join(DOCUMENT_PATH,'Backgrounds','refined',file))
#         json.dump(refined_data, f, indent=4)