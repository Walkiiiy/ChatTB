import json
import wikipediaapi
import requests
import os
import dotenv

dotenv.load_dotenv()
DOCUMENT_PATH=os.getenv("DEV_DOCUMENT")

def wiki_search_queryV2(query:str):# handel single query
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



def wiki_search_queryV1(query:str):  # handle single query
    user_agent = 'MyWikiSearchBot/1.0 (https://example.com/contact)'
    
    wiki = wikipediaapi.Wikipedia(
        language='en',
        user_agent=user_agent
    )
    
    page = wiki.page(query)
    if page.exists():
        return page.title, page.summary, page.text
    else:
        return page.title, None, None

def update_json_with_wiki_docs(json_file_path):
    print('processing',json_file_path)
    # Read the existing JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    # For each entry in the list, perform the wiki search and update
    for entry in data:
        word = entry["selectedWord"]
        title, summary, text = wiki_search_queryV2(word)
        
        # Update the entry with the wiki document or null if not found
        entry["document"] = {
            "title": title,
            "summary": summary,
            "text": text if text else None
        }
    
    # Write the updated data back to the JSON file
    with open(json_file_path, 'w') as file:
        json.dump(data, file, indent=4)

# Example usage
for file in os.listdir(DOCUMENT_PATH+'/DocsToSearch'):
    json_file_path = os.path.join(DOCUMENT_PATH,'DocsToSearch',file)
    update_json_with_wiki_docs(json_file_path)
