# Import necessary libraries
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download the VADER lexicon if not already present
nltk.download('vader_lexicon')

# Initialize the sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Function to analyze sentiment
def analyze_sentiment(text):
    sentiment_score = sia.polarity_scores(text)
    return sentiment_score

# Example usage
if __name__ == "__main__":
    # Sample text input
    text_input = input("Enter a sentence for sentiment analysis: ")
    
    # Analyze the sentiment
    sentiment_result = analyze_sentiment(text_input)
    
    # Print the results
    print("\nSentiment Analysis Result:")
    print(f"Positive: {sentiment_result['pos']}")
    print(f"Negative: {sentiment_result['neg']}")
    print(f"Neutral: {sentiment_result['neu']}")
    print(f"Compound Score: {sentiment_result['compound']}")

    # Interpret the compound score
    if sentiment_result['compound'] >= 0.05:
        print("Overall Sentiment: Positive")
    elif sentiment_result['compound'] <= -0.05:
        print("Overall Sentiment: Negative")
    else:
        print("Overall Sentiment: Neutral")
