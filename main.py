import os
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
from crewai import Agent, Crew, Task
from dotenv import load_dotenv

# Load API Key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# --- Agent 1: News Fetcher ---
class NewsFetcher:
    def fetch_news(self, url):
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Extract article title (heading)
            title_tag = soup.find("h1") or soup.find("title")
            article_title = title_tag.get_text().strip() if title_tag else "Untitled Article"
            
            # Extract article body text
            paragraphs = soup.find_all("p")
            article_text = " ".join([p.get_text() for p in paragraphs])
            
            return article_title, article_text if article_text else "No content found."
        except Exception as e:
            return "Error fetching article", f"Error: {str(e)}"

fetcher_agent = Agent(
    role="News Fetcher",
    backstory="Scrapes and extracts article text and heading.",
    goal="Fetch article heading and text.",
)

# --- Agent 2: Categorizer ---
def categorize_news(article_text):
    model = genai.GenerativeModel("gemini-pro")
    prompt = f"Categorize this news article: '{article_text[:1000]}' into categories like Technology, Politics, Sports, etc."
    response = model.generate_content(prompt)
    return response.text.strip() if response else "Unknown Category"

categorizer_agent = Agent(
    role="Categorizer",
    backstory="Classifies news articles.",
    goal="Categorize articles.",
)

# --- Agent 3: Summarizer ---
def summarize_article(article_title, article_text):
    model = genai.GenerativeModel("gemini-pro")
    prompt = (f"Here is a news article titled '{article_title}'.\n\n"
              f"Summarize this article in a detailed way with a minimum of 200 words:\n\n{article_text[:3000]}")
    response = model.generate_content(prompt)
    return response.text.strip() if response else "Summary not available."

summarizer_agent = Agent(
    role="Summarizer",
    backstory="Generates detailed article summaries of at least 200 words.",
    goal="Summarize articles in a detailed way.",
)

# --- Crew Setup ---
crew = Crew(
    agents=[fetcher_agent, categorizer_agent, summarizer_agent]
)

# --- CLI Interaction ---
def main():
    url = input("Enter the news article URL: ")
    
    fetcher = NewsFetcher()
    article_title, article_text = fetcher.fetch_news(url)
    
    if "Error" in article_text:
        print(article_text)
        return

    category = categorize_news(article_text)
    summary = summarize_article(article_title, article_text)

    print(f"Title: {article_title}")
    print(f"Category: {category}")
    print(f"Summary:\n{summary}")

if __name__ == "__main__":
    main()
