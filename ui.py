import pandas as pd
import string
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Your transcript goes here
transcript = """      "I'm thoroughly satisfied with my experience with your company. Well done!"
 "I'm disappointed with the poor quality of your product."
    
 "My experience with your company was neither exceptional nor disappointing."   
    
    """

sentences = [' '.join(sent.split()).strip() for sent in transcript.replace('\n', '').split('. ')]

# convert to dataframe
df = pd.DataFrame(sentences, columns=['content'])

def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def clean_text(text, digits=False, stop_words=False, lemmatize=False, only_noun=False):
    text = str(text).lower()
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    
    if digits:
        text = [word for word in text if not any(c.isdigit() for c in word)]
        
    if stop_words:
        stop = stopwords.words('english')
        text = [x for x in text if x not in stop]
    
    text = [t for t in text if len(t) > 0]
    
    if lemmatize:
        pos_tags = pos_tag(text)
        text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
        
    if only_noun:
        is_noun = lambda pos: pos[:2] == 'NN'
        text = [word for (word, pos) in pos_tag(text) if is_noun(pos)]
    
    text = [t for t in text if len(t) > 1]
    text = " ".join(text)
    
    return text

df['content_clean'] = df['content'].apply(lambda x: clean_text(x, digits=True, stop_words=True, lemmatize=True))

sid = SentimentIntensityAnalyzer()

df['sentiment']= df['content_clean'].apply(lambda x: sid.polarity_scores(x))
df = pd.concat([df.drop(['sentiment'], axis=1), df['sentiment'].apply(pd.Series)], axis=1)
df = df.rename(columns={'neu': 'neutral', 'neg': 'negative', 'pos': 'positive'})

df['confidence'] = df[["negative", "neutral", "positive"]].max(axis=1)
df['sentiment'] = df[["negative", "neutral", "positive"]].idxmax(axis=1)

grouped = pd.DataFrame(df['sentiment'].value_counts()).reset_index()
grouped.columns = ['sentiment', 'count']

sentiment_ratio = df['sentiment'].value_counts(normalize=True).to_dict()
for key in ['negative', 'neutral', 'positive']:
    if key not in sentiment_ratio:
        sentiment_ratio[key] = 0.0

sentiment_score = (sentiment_ratio['neutral'] + sentiment_ratio['positive']) - sentiment_ratio['negative']

# Attractive front page

st.title("Sentiment Analysis on Earnings Call Transcript")
st.image("https://news.itmo.ru/images/news/big/p9806.jpg", use_column_width=True)
st.markdown(
    """
    ## Welcome to the Sentiment Analysis on Earnings Call Transcript Dashboard! 
    
    Explore sentiment distribution, sentiment ratio, and more.
    """
)

# Navigation links
st.sidebar.title("EXPLORE")
selected_page = st.sidebar.radio("", ["Home", "Sentiment Ratios", "Sentiment Score", "Sentiments Distribution", "Sentiment Breakdown"])

if selected_page == "Sentiment Ratios":
    # Display sentiment ratio
    st.subheader("Sentiment Ratio")
    st.write(sentiment_ratio)

elif selected_page == "Sentiment Score":
    # Display sentiment score
    st.subheader("Sentiment Score")
    st.plotly_chart(go.Figure(go.Indicator(
        mode="number+delta",
        value=sentiment_score,
        delta={"reference": 0.5},
        title={"text": "Sentiment Score"},
    )))

elif selected_page == "Sentiments Distribution":
    # Display sentiment pie chart
    st.subheader("Sentiments Distribution")
    fig_pie = px.pie(grouped, values='count', names='sentiment', title='Sentiments')
    st.plotly_chart(fig_pie)

elif selected_page == "Sentiment Breakdown":
    # Display all sentence locations
    st.subheader("Sentiment Breakdown")
    fig_scatter = px.scatter(df, y='sentiment', color='sentiment',
                             size='confidence', hover_data=['content'],
                             color_discrete_map={"negative": "firebrick", "neutral": "navajowhite", "positive": "darkgreen"})

    fig_scatter.update_layout(width=800, height=300)
    st.plotly_chart(fig_scatter)


