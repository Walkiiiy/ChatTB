from mcp.server import FastMCP
from dotenv import load_dotenv
import os
import wikipediaapi
# from LSH import lsh
import requests

load_dotenv()
GAODE_KEY = os.getenv("GAODE_KEY")
CITY_CODE_FILE = os.getenv("CITY_CODE_FILE")
GOOGLE_API = os.getenv("GOOGLE_API")
DOCUMENT_PATH=os.getenv("DEV_DOCUMENT")
app = FastMCP("weather-query")


@app.tool()    
def wiki_search(queries:list[list[str]],database_name):
    """
    Handle Wikipedia searches and return all possible Wikipedia search results.
    Parameters:
     - queries (List[List[str]]): The list contains every ambiguous word's every possible query that needs to look up.
     - database_name: the name of the database.   
    Returns:
     - List[List[str]]:The list contains every query word's possible explaination documents. 
    """
    visited=set()
    res=[]
    with open(os.path.join(DOCUMENT_PATH,f'{database_name}.txt'),'w') as f:
        f.write("db:"+database_name+'\n'+str(queries)+"\n\n\n\n\n----------------------\n")
        for possible_queries_one_word in queries:
            candidates=[]
            for possible_query in possible_queries_one_word:
                candidate=wiki_search_query(possible_query,visited)
                if candidate:
                    candidates+=candidate
                    for doc in candidate:
                        f.write(possible_query+"\n"+doc+"\n\n\n\n\n----------------------\n")
            res.append(candidates)
    return res

def wiki_search_query(query:str,visited:set):# handel single query
    # MediaWiki API search endpoint
    search_url = 'https://en.wikipedia.org/w/api.php'
    user_agent = 'MyWikiSearchBot/1.0 (https://example.com/contact)'
    params = {
        'action': 'query',
        'list': 'search',
        'srsearch': query,
        'format': 'json',
        'utf8': 1,
        'srlimit': 5,  # Limit to top 5 search results
    }
    response = requests.get(search_url, params=params, headers={'User-Agent': user_agent})
    data = response.json()

    # Extract search results
    search_results = data.get('query', {}).get('search', [])

    # If there are no search results, return a message
    # if not search_results:
    #     return "No pages found for your query."
    
    print(f'query:{query} found {len(search_results)} results.')
    # Now, use wikipediaapi to get the content of the best match
    wiki = wikipediaapi.Wikipedia(
        language='en',
        user_agent=user_agent
    )
    res=[]
    for match in search_results[:2]:
        if match['title'] not in visited:
            page = wiki.page(match['title'])    
            if page.exists():
                res.append(page.title+'\n'+page.summary)
            visited.add(match['title'])
    return res
        
if __name__ == "__main__":
    # query=[['Minecraft']]
    # res=wiki_search(query,'game')
    # print(res)
    app.run(transport='stdio')



# @app.tool()
# def search_and_fetch(query):
#     """
#     Main function to handle the search and fetch process.
#     """
#     search_results = google_search(query)
    
#     wiki_content = []
#     nonWiki_content = []
#     print(f"Total results found: {len(search_results)}")
    
#     for result in search_results:
#         try:
#             if 'wikipedia.org' in result.get('link', ''):
#                 wiki_url = result.get('link')
#                 page_title = wiki_url.split('/')[-1]
#                 wiki_content.append(wiki_search(page_title))
#             else:
#                 content_url = result.get('link')
#                 nonWiki_content.append(crawl_and_format_website(content_url))
#         except Exception as e:
#             print(f"Error processing the result {result.get('link')}: {str(e)}")
    
#     return wiki_content, nonWiki_content



# @app.tool()    
# def google_search(query, num_results=20):
#     """
#     Function to search Google Custom Search Engine API using httpx synchronously.

#     Parameters:
#     - query (str): The search query.
#     - num_results (int): The number of results to return.

#     Returns:
#     - list: A list of search result items.
#     """
#     url = "https://www.googleapis.com/customsearch/v1"
#     cse_id = 'e6f97f553d0704e79'  # Custom Search Engine ID
#     start_index = 1  # Start at the first result
    
#     all_results = []  # List to store all the results
    
#     # Fetch results in chunks of 10 (the maximum allowed by the API)
#     while len(all_results) < num_results:
#         params = {
#             'q': query,
#             'key': GOOGLE_API,
#             'cx': cse_id,
#             'num': 10,  # Maximum number of results per request
#             'start': start_index  # Index to start the next batch of results
#         }
        
#         with httpx.Client() as client:
#             response = client.get(url, params=params)
        
#         if response.status_code == 200:
#             search_results = response.json().get('items', [])
#             all_results.extend(search_results)  # Add new results to the list

#             if len(search_results) < 10:  # If less than 10 results were returned, stop
#                 break
            
#             # Increment the start index for the next batch of results
#             start_index += 10
#         else:
#             print(f"Error: {response.status_code}")
#             break
    
#     return all_results[:num_results]  # Return only the number of results requested



# @app.tool()
# def crawl_and_format_website(url):
#     """
#     Function to crawl a website and return a formatted, readable document.
#     """
#     headers = {
#         'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
#     }
    
#     try:
#         response = requests.get(url, headers=headers, timeout=10)
        
#         # Check for response status
#         if response.status_code == 200:
#             soup = BeautifulSoup(response.text, 'html.parser')
            
#             # Clean up unnecessary tags like scripts, styles, ads, etc.
#             for script_or_style in soup(['script', 'style', 'noscript']):
#                 script_or_style.decompose()  # Remove these elements
            
#             # Extract text only from certain relevant tags
#             content = []
#             seen_text = set()  # To track and avoid adding duplicate text
            
#             for tag in soup.find_all(['p', 'h1', 'h2', 'h3']):
#                 text = tag.get_text(strip=True)
#                 if text and text not in seen_text:  # Only add unique text
#                     content.append(text)
#                     seen_text.add(text)  # Track this text as seen
            
#             # If there are any divs with important content, include them
#             for div in soup.find_all('div', class_=re.compile('.*(content|main).*', re.IGNORECASE)):
#                 div_text = div.get_text(strip=True)
#                 if div_text and div_text not in seen_text:
#                     content.append(div_text)
#                     seen_text.add(div_text)
            
#             # Combine the extracted content into a single string with proper formatting
#             formatted_content = "\n\n".join(content)
#             return formatted_content
        
#         else:
#             return f"Error: Unable to access the website (Status code {response.status_code})"
    
#     except requests.exceptions.Timeout:
#         return f"Error: The request timed out. url:{url}"
#     except requests.exceptions.RequestException as e:
#         return f"Error: Request failed due to {str(e)}"
#     except Exception as e:
#         return f"Unexpected error: {str(e)}"
    