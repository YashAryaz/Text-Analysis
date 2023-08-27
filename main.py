from bs4 import BeautifulSoup
import requests
import os
import pandas as pd
import nltk
from textblob import TextBlob
from textstat import syllable_count
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
nltk.download('punkt')
nltk.download('stopwords')

class ArticleScraper:
    def __init__(self,path):
        self.input_path = path+"/Input.xlsx"
        self.output_folder = 'extracted_articles'
        os.makedirs(self.output_folder, exist_ok=True)
    
    def scrape_articles(self):
        input_df = pd.read_excel(self.input_path)
        urls = input_df['URL'].tolist()
        
        for idx, url in enumerate(urls):
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                article_title = soup.find('title').get_text()
                article_text = ""
                
                article_content = soup.find('div', class_='td-post-content tagdiv-type')
                if article_content:
                    paragraphs = article_content.find_all('p')
                    for p in paragraphs:
                        article_text += p.get_text() + '\n'
                
                url_id = f'URL_{idx + 1}'
                filename = os.path.join(self.output_folder, f'{url_id}.txt')
                with open(filename, 'w', encoding='utf-8') as file:
                    file.write(article_title + '\n')
                    file.write(article_text)
                
                print(f'Saved {filename}')
            else:
                print(f'Failed to retrieve content from {url}')

class ArticleAnalyzer:
    def __init__(self, path):
        input_df = pd.read_excel(path+"/Input.xlsx")
        self.input_df = input_df
        self.output_df = pd.DataFrame(columns=input_df.columns.tolist() + [
            'POSITIVE SCORE', 'NEGATIVE SCORE', 'POLARITY SCORE', 'SUBJECTIVITY SCORE',
            'AVG SENTENCE LENGTH', 'PERCENTAGE OF COMPLEX WORDS', 'FOG INDEX',
            'AVG NUMBER OF WORDS PER SENTENCE', 'COMPLEX WORD COUNT', 'WORD COUNT',
            'SYLLABLE PER WORD', 'PERSONAL PRONOUNS', 'AVG WORD LENGTH'
        ])

        # Load positive and negative words
        with open(f'{path}/MasterDictionary/positive-words.txt', 'r') as positive_file:
            self.positive_words = positive_file.read().splitlines()

        with open(f'{path}/MasterDictionary/negative-words.txt', 'r', encoding='ISO-8859-1', errors='replace') as negative_file:
            self.negative_words = negative_file.read().splitlines()

        # Load stopwords from various files
        self.stopwords_set = set()
        stopwords_files = [
            'StopWords_Auditor.txt', 'StopWords_Currencies.txt',
            'StopWords_DatesandNumbers.txt', 'StopWords_GenericLong.txt',
            'StopWords_Geographic.txt', 'StopWords_Names.txt'
        ]

        for file in stopwords_files:
            with open(f'{path}/StopWords/'+file, 'r',encoding='ISO-8859-1', errors='replace') as stopwords_file:
                self.stopwords_set.update(stopwords_file.read().splitlines())

    def analyze_articles(self):
        for idx, row in self.input_df.iterrows():
            url_id = f'URL_{idx + 1}'
            filename = os.path.join('extracted_articles', f'{url_id}.txt')

            try:
                with open(filename, 'r', encoding='utf-8') as file:
                    article = file.read()

                    blob = TextBlob(article)
                    sentences = sent_tokenize(article)
                    words = word_tokenize(article)

                    positive_score = sum(1 for word in words if word in self.positive_words)
                    negative_score = sum(1 for word in words if word in self.negative_words)
                    polarity_score = blob.sentiment.polarity
                    subjectivity_score = blob.sentiment.subjectivity

                    avg_sentence_length = sum(len(sent.split()) for sent in sentences) / len(sentences)

                    words_without_stopwords = [word for word in words if word.lower() not in self.stopwords_set]
                    complex_words = [word for word in words_without_stopwords if syllable_count(word) > 2]

                    complex_word_percentage = len(complex_words) / len(words_without_stopwords) * 100

                    fog_index = 0.4 * (avg_sentence_length + complex_word_percentage)

                    avg_words_per_sentence = len(words) / len(sentences)
                    complex_word_count = len(complex_words)
                    word_count = len(words)

                    syllable_per_word = sum(syllable_count(word) for word in words) / len(words)

                    personal_pronouns = sum(1 for word in words_without_stopwords if word.lower() in ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours'])

                    avg_word_length = sum(len(word) for word in words) / len(words)

                    self.output_df.loc[idx] = row.tolist() + [
                        positive_score, negative_score, polarity_score, subjectivity_score,
                        avg_sentence_length, complex_word_percentage, fog_index,
                        avg_words_per_sentence, complex_word_count, word_count,
                        syllable_per_word, personal_pronouns, avg_word_length
                    ]
            except FileNotFoundError:
                print(f'Article not found for URL: {url_id}')

        # Save the output data to a new Excel file
        self.output_df.to_excel('output_analysis.xlsx', index=False)
        print("Execution Completed..")

if __name__=="__main__":
    input_path = input("Enter the path to the input Excel file (including filename): ")

    # Create an instance of the ArticleScraper class and scrape articles
    scraper = ArticleScraper(input_path)
    scraper.scrape_articles()
    # Create an instance of ArticleAnalyzer class and analyze articles
    analyzer = ArticleAnalyzer(input_path)
    analyzer.analyze_articles()