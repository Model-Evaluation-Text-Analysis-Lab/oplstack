import wikipedia
import json


# Set the Wikipedia library to use the English version of Wikipedia
wikipedia.set_lang("en")

years = ["2022", "2023"]
page_data = []

for year in years:
    print("YEAR -------------", year)
    # Fetch the first 500 search results
    search_results = wikipedia.search(year, results=50000)
    # Check if there are any search results
    if not search_results:
        continue
    # Fetch content for all search results
    i = 0
    for page_title in search_results:
        i += 1
        print(i)
        try:
            # Fetch the page object for the current title
            page = wikipedia.page(page_title)
            print(f'Succeeded in getting {page_title}')
            # Fetch the first 500 words of the page content
            content = page.content
            # Save the content with the id, title, and content
            page_data.append({
                "id": i,
                "title": page_title,
                "content": content
            })

        except Exception as e:
            i-=1
            print(f'Error fetching content for {page_title}: {e}')

# Save the JSON data to a file
with open('/Users/undi69/Desktop/Jarvis/OPLStack/2022_page_summaries2.json', 'w') as f:
    json.dump(page_data, f)
