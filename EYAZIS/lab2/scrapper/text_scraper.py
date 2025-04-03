import os
import re
from newspaper import Article

# Define the directory to save articles
output_directory = "articles"
os.makedirs(output_directory, exist_ok=True)  # Create the directory if it doesn't exist

def sanitize_filename(title):
    # Remove invalid characters from the title
    return re.sub(r'[<>:"/\\|?*]', '_', title)

# Open the file containing the URLs
with open("extracted_links.txt", "r") as url_file:
    urls = url_file.readlines()

for url in urls:
    url = url.strip()  # Remove any leading/trailing whitespace
    if url:  # Check if the line is not empty
        try:
            article = Article(url)
            article.download()
            article.parse()  # Parse the article to extract text

            # Sanitize the filename based on the article title
            sanitized_title = sanitize_filename(article.title)
            filename = f"{sanitized_title}.txt"
            filepath = os.path.join(output_directory, filename)

            # Write metadata and article text to the file
            with open(filepath, "w", encoding='utf-8') as output_file:
                output_file.write(f"Title: {article.title}\n")
                output_file.write(f"Authors: {', '.join(article.authors)}\n")
                output_file.write(f"Published Date: {article.publish_date}\n")
                output_file.write(f"Source URL: {url}\n\n")
                output_file.write(article.text)

            print(f"Saved article: {filename}")
        except Exception as e:
            print(f"Failed to process {url}: {e}")