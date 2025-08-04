from selenium import webdriver
from selenium.webdriver.common.by import By
import time
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from urllib.parse import urljoin
import re
import random
import os
import httpx

GOOGLE_API = os.getenv("GOOGLE_API")

class datacollecter:
    def __init__(self):
        self.visited_urls = set()
        self.driver = webdriver.Chrome(service=Service("/home/walkiiiy/Apps/chromedriver-linux64/chromedriver"))
    def get_all_data(self,url):
        if url in self.visited_urls:
            return
        
        self.visited_urls.add(url)

        self.driver.get(url)

        time.sleep(random.uniform(2, 4))  # Adding a random sleep time to be less predictable

        # Extract title of the page
        page_title = self.driver.title

        relevant_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul', 'ol', 'li', 'a', 'span', 'strong', 'em']
        all_data = []

        for tag in relevant_tags:
            try:
                elements = self.driver.find_elements(By.TAG_NAME, tag)
                for element in elements:
                    # Check for stale elements and re-locate if necessary
                    
                    text = element.text.strip()

                    # Skip image links and non-HTML resources (like favicons)
                    if tag == 'img' or self.is_non_html_resource(url):
                        continue  # Skip the image or non-text content

                    # If the element is an image, get its alt text
                    if tag == 'img':
                        alt_text = element.get_attribute('alt')
                        if alt_text:
                            text = alt_text.strip()

                    # Append all the text (no filtering, capture everything)
                    all_data.append(text)
            except Exception as e:
                print(f"Error while extracting {tag}: {e}")
                continue  # Skip elements that cause issues

        res=''
        for idx, data in enumerate(all_data, start=1):
            if data:
                res+=data+'\n'
        return res

    def is_non_html_resource(self,url):
        """
        Check if the URL is a non-HTML resource like an image, favicon, or downloadable file.
        """
        # Skip non-HTML resources like images, favicons, and downloadable files
        if url.endswith(('.png', '.jpg', '.jpeg', '.gif', '.ico', '.svg', '.pdf')):
            return True  # Skip image links or downloadable files
        return False
        


    def bing_search(self,query, num_results=10):
        """
        Function to search using Bing Search API synchronously.

        Parameters:
        - query (str): The search query.
        - num_results (int): The number of results to return.

        Returns:
        - list: A list of search result items.
        """
        url = "https://api.bing.microsoft.com/v7.0/search"  # Bing Search API endpoint
        subscription_key = "YOUR_BING_API_KEY"  # Replace with your Bing Search API key
        
        headers = {
            "Ocp-Apim-Subscription-Key": subscription_key
        }
        
        all_results = []  # List to store all the results
        count = 0  # Variable to track how many results we've fetched

        # Fetch results in chunks of 50 (the maximum allowed by Bing API)
        while count < num_results:
            # Define parameters for the API request
            params = {
                "q": query,
                "count": 50,  # Maximum number of results per request
                "offset": count  # Start from where the last query ended
            }

            # Make the request to Bing API
            with httpx.Client(verify=False) as client:
                response = client.get(url, headers=headers, params=params)

            if response.status_code == 200:
                search_results = response.json().get("webPages", {}).get("value", [])
                all_results.extend(search_results)  # Add new results to the list

                # Update the count of fetched results
                count += len(search_results)

                if len(search_results) < 50:  # If less than 50 results are returned, stop
                    break
            else:
                print(f"Error: {response.status_code}")
                break
        
        # Return only the number of results requested
        return all_results[:num_results]





    def google_search(self,query,num_results=10):
        """
        Function to search Google Custom Search Engine API using httpx synchronously.

        Parameters:
        - query (str): The search query.
        - num_results (int): The number of results to return.

        Returns:
        - list: A list of search result items.
        """
        url = "https://www.googleapis.com/customsearch/v1"
        cse_id = 'e6f97f553d0704e79'  # Custom Search Engine ID
        start_index = 1  # Start at the first result
        
        all_results = []  # List to store all the results
        
        # Fetch results in chunks of 10 (the maximum allowed by the API)
        while len(all_results) < num_results:
            params = {
                'q': query,
                'key': GOOGLE_API,
                'cx': cse_id,
                'num': 10,  # Maximum number of results per request
                'start': start_index  # Index to start the next batch of results
            }
            while True:
                try:    
                    with httpx.Client(verify=False) as client:
                        response = client.get(url, params=params)
                        break
                except httpx.ConnectError as e:
                    print(e)

            if response.status_code == 200:
                search_results = response.json().get('items', [])
                all_results.extend(search_results)  # Add new results to the list

                if len(search_results) < 10:  # If less than 10 results were returned, stop
                    break
                
                # Increment the start index for the next batch of results
                start_index += 10
            else:
                print(f"Error: {response.status_code}")
                break
        
        return all_results  # Return only the number of results requested

    def get_doc(self,q:str):
        """
        getting a single query.returnning a list of dicts with scraped related pages.
        """
        print(q," is being searched")
        # try:  
        search_res=self.google_search(q)
        print(q,"seach compeleted")
        # except Exception as e:
        #     print("search error:",e)
        # print(search_res)
        q_res=[]

        for item in search_res:
            print(item['link']," is being scraped")
            while True:
                try:
                    item_res=self.get_all_data(item['link'])
                    break
                except Exception as e:
                    print(f"Error: {e}")
            print(item["link"],"scraping compeleted")
            q_res.append({
                "title":item['title'],
                "link":item['link'],
                "content":item_res
            })
        return q_res

    def __del__(self):
        self.driver.quit()
