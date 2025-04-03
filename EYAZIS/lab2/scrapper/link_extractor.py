import os
import re
from bs4 import BeautifulSoup

# Directory containing HTML files
directory = "scrapper//pages"  # Change this to your actual directory path

links_set = set()

# Base URL to prepend
base_url = "https://www.nytimes.com"

# List all HTML files in the directory
html_files = [f for f in os.listdir(directory) if f.endswith(".html")]

# Loop through each HTML file
for file_name in html_files:
    file_path = os.path.join(directory, file_name)

    # Open and read the content of the HTML file
    with open(file_path, "r", encoding="utf-8") as file:
        html_content = file.read()

    # Parse the HTML content
    soup = BeautifulSoup(html_content, "html.parser")

    # Extract all <a> tags
    links = []

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        # Use regex to filter relevant links
        if re.search(r"arts/music/(?!\?page=)\S+", href):
            # Prepend base URL to relative links
            full_link = href if href.startswith("http") else base_url + href
            links.append(full_link)
            links_set.add(full_link)

    # Debug prints
    print(f"Processed file: {file_name}")
    print(f"Links extracted from {file_name}: {len(links)}")
    print(f"Current unique links count: {len(links_set)}\n")

# Write all unique links to a text file
with open("scrapper/extracted_links.txt", "w", encoding="utf-8") as file:
    for link in links_set:
        file.write(link + "\n")