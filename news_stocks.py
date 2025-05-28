import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")

from datetime import datetime
import yfinance as yf

# Get 2+ years of price data
start = datetime(2022, 1, 1)
end = datetime.now()

df = yf.download("AAPL", start=start, end=end)
df.reset_index(inplace=True)
df['Date_'] = pd.to_datetime(df['Date'])
df['Return'] = df['Close'].pct_change().shift(-1)
df['Label'] = df['Return'].apply(lambda x: 1 if x > 0 else 0)

df.info

# Plot Close Price
plt.figure(figsize=(12, 5))
plt.plot(df['Date'], df['Close'], label='Close Price')
plt.title("AAPL Close Price")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()

# Plot Volume
plt.figure(figsize=(12, 5))
plt.plot(df['Date'], df['Volume'], label='Volume', color='purple')
plt.title("AAPL Trading Volume")
plt.xlabel("Date")
plt.ylabel("Volume")
plt.legend()
plt.show()

# Add moving averages
ma_days = [10, 20, 50]
for ma in ma_days:
    df[f"MA_{ma}"] = df['Close'].rolling(window=ma).mean()

# Plot using matplotlib
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Close'], label='Close', color='black')

for ma in ma_days:
    plt.plot(df['Date'], df[f"MA_{ma}"], label=f"{ma}-Day MA")

plt.title("AAPL Adjusted Close Price with Moving Averages")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

from newspaper import Article
from newspaper import build
from datetime import datetime

# Build the news source (this fetches article links)
cnbc_paper = build('https://www.cnbc.com', memoize_articles=False)

# Filter articles that likely mention Apple
apple_news = []
for article in cnbc_paper.articles:
    if 'apple' in article.url.lower():
        try:
            article.download()
            article.parse()
            apple_news.append({
                'date': article.publish_date.date() if article.publish_date else datetime.today().date(),
                'title': article.title,
                'text': article.text,
                'url': article.url
            })
        except:
            continue

# Create a DataFrame
import pandas as pd
news_df = pd.DataFrame(apple_news)
news_df['date'] = pd.to_datetime(news_df['date']).dt.date
news_df = news_df[['date', 'title', 'url']]
news_df.head()
import feedparser
from datetime import datetime
import pandas as pd

rss = "https://news.google.com/rss/search?q=apple&hl=en-IN&gl=IN&ceid=IN:en"
rss = feedparser.parse(rss)

google_headlines = []
for entry in rss.entries:
    date = datetime(*entry.published_parsed[:6]).date()
    google_headlines.append({
        'title': entry.title,
        'date': date,
        'url': entry.link
     })
google_df = pd.DataFrame(google_headlines)
google_df.head()
google_df

import feedparser
from datetime import datetime
import pandas as pd

# Define negative keywords to filter headlines
negative_keywords = [
    'miss', 'lawsuit', 'investigation', 'cut', 'fall', 'drops',
    'regulation', 'privacy', 'decline', 'recall', 'antitrust',
    'delay', 'backlash', 'issue', 'disappoint', 'slowdown', 'ban',
    'resignation', 'probe', 'fines', 'warning', 'concerns', 'downturn', 'sues', 'lawsuits',
    'troubles', 'sued', 'slips', 'decline', 'dip', 'controversy', 'plunges', 'loss', 'antitrust'
    'fined', 'delays', 'privacy concerns', 'backlash', 'layoffs'
]

# Google News RSS feed for Apple
rss_url = "https://news.google.com/rss/search?q=Apple+stock&hl=en-IN&gl=IN&ceid=IN:en"
feed = feedparser.parse(rss_url)

# Filter and collect negative news
negative_news = []
for entry in feed.entries:
    title = entry.title
    pub_date = datetime(*entry.published_parsed[:6]).date()
    url = entry.link

    # Simple filter for negative sentiment
    if any(word in title.lower() for word in negative_keywords):
        negative_news.append({'date': pub_date, 'title': title, 'url': url, 'Label': 0})

# Convert to DataFrame
negative_news_df = pd.DataFrame(negative_news)
print(f"Scraped {len(negative_news_df)} negative headlines.")
negative_news_df.info

import feedparser
from datetime import datetime
import pandas as pd

# Define positive keywords to filter headlines
positive_keywords = [
    'rise', 'growth', 'surge', 'gain', 'profit', 'beats expectations',
    'record', 'rebound', 'rallies', 'boost', 'soars', 'increase',
    'recovery', 'jumps', 'upward', 'positive outlook', 'strong', 'expands',
    'acquisition', 'improves', 'upgrade', 'outperforms', 'accelerates',
    'rise', 'rises', 'rising',
    'gain', 'gains', 'gaining',
    'surge', 'surges', 'surging',
    'soar', 'soars', 'soaring',
    'jump', 'jumps', 'jumping',
    'rally', 'rallies', 'rallying',
    'boost', 'boosts', 'boosted',
    'growth', 'expansion', 'expands', 'expanding',
    'profit', 'profits', 'profitable', 'net income',
    'beats expectations', 'outperforms', 'beats estimates',
    'record high', 'record profits', 'record growth',
    'strong quarter', 'strong earnings', 'strong performance',
    'positive outlook', 'upward trend', 'bullish',
    'recovery', 'rebound', 'recovers', 'rebounding',
    'accelerates', 'acceleration',
    'success', 'successful', 'wins', 'winner',
    'upgrade', 'upgraded',
    'launch', 'launches', 'released', 'introduces',
    'acquisition', 'merger', 'partner', 'partnership',
    'investment', 'investments', 'investors optimistic',
    'new high', 'stock climbs', 'demand increases',
    'innovation', 'innovative', 'breakthrough',
    'expands market', 'opens new market',
    'strategic deal', 'milestone', 'momentum',
    'buy rating', 'overweight rating', 'price target increased'
]

# Google News RSS feed for Apple
rss_url = "https://news.google.com/rss/search?q=Apple+stock&hl=en-IN&gl=IN&ceid=IN:en"
feed = feedparser.parse(rss_url)

# Filter and collect positive news
positive_news = []
for entry in feed.entries:
    title = entry.title
    pub_date = datetime(*entry.published_parsed[:6]).date()
    url = entry.link

    # Simple filter for positive sentiment
    if any(word in title.lower() for word in positive_keywords):
        positive_news.append({'date': pub_date, 'title': title, 'url': url, 'Label': 1})

# Convert to DataFrame
positive_news_df = pd.DataFrame(positive_news)
print(f"Scraped {len(positive_news_df)} positive headlines.")
positive_news_df.info()

import pandas as pd
from datetime import datetime

# Synthetic positive headlines
positive_headlines = [
    "Apple stock jumps 3% after strong iPhone sales in Q2",
    "Apple announces record-breaking quarterly revenue",
    "iPhone 15 sales surge past analyst expectations",
    "Apple beats Wall Street estimates in earnings report",
    "Tim Cook praises team for strongest quarter in company history",
    "Apple Watch demand hits new all-time high",
    "Apple stock climbs as new MacBooks receive critical acclaim",
    "Analysts upgrade Apple to “Strong Buy” after bullish forecast",
    "Apple gains momentum in India with record market share",
    "iPad sales surge amid global tablet demand",
    "Apple stock rallies on strong services revenue growth",
    "Apple announces $90B stock buyback program",
    "Apple TV+ wins multiple Emmy awards, boosts brand sentiment",
    "Apple’s chip strategy leads to 25% margin improvement",
    "Apple reaches all-time high in market cap",
    "AI-powered Siri update drives positive investor sentiment",
    "Apple Pay expands to 15 new countries",
    "Apple secures major deal with Disney+ on content bundling",
    "iOS update receives praise for speed and stability",
    "Apple leads S&P 500 gains in tech-driven rally",
    "Apple’s carbon-neutral initiative draws praise from investors",
    "Apple Music subscriptions hit 100 million milestone",
    "Apple’s Vision Pro headset receives preorders surge",
    "Apple expands R&D in machine learning, boosts stock outlook",
    "Morgan Stanley calls Apple “top pick” for next quarter",
    "iPhone trade-in program drives upgrade cycle",
    "Apple Car rumors push stock to monthly high",
    "Analysts raise Apple’s price target to $220",
    "Apple signs exclusive supply deal with TSMC for 3nm chips",
    "Apple receives government grant for clean energy innovation",
    "Apple’s education initiative gains traction in Europe",
    "Apple sees 40% growth in China sales",
    "Goldman Sachs upgrades Apple to “Buy” rating",
    "iPhone loyalty remains highest among smartphone brands",
    "Apple Watch saves lives, boosts product image",
    "Apple’s App Store hits $100B in developer payouts",
    "Apple expands manufacturing base in Vietnam",
    "Apple beats Q1 expectations with strong hardware performance",
    "Apple posts record holiday sales",
    "Apple ecosystem cited as key strength in market survey",
    "Apple partners with health providers to advance digital health",
    "Mac shipments rise 15% year over year",
    "Apple opens flagship retail store in Singapore",
    "Investors optimistic after Apple’s AI roadmap presentation",
    "Apple introduces satellite connectivity, shares pop",
    "Apple stock soars after positive Fed commentary",
    "Services segment becomes Apple’s second-largest revenue stream",
    "Apple leads innovation index for fifth year in a row",
    "Apple Card user satisfaction at record levels",
    "Apple’s ARKit 5 earns developer praise",
    "New iMac gets glowing reviews from tech outlets",
    "Apple’s stock boosted by better-than-expected earnings call",
    "Apple earns top spot in JD Power satisfaction survey",
    "Apple retail expansion continues with five new global stores",
    "Apple integrates ChatGPT-like assistant in iOS 18",
    "Apple outpaces Samsung in global smartphone revenue",
    "Apple raises dividend amid strong cash flow",
    "Apple to launch hardware subscription service",
    "MacBook Pro wins design award, sparks consumer buzz",
    "Apple ranks #1 in Fortune’s World’s Most Admired Companies",
    "Apple’s iCloud upgrade draws positive user reviews",
    "Apple secures new patent on revolutionary display tech",
    "Apple Care+ revenue grows 30% YoY",
    "Apple stock climbs on positive sentiment from retail investors",
    "Apple AirPods market share hits new peak",
    "Apple invests $1B in green manufacturing tech",
    "Apple’s AI-powered fitness app sees strong early adoption",
    "Apple Watch Ultra receives best-in-class battery praise",
    "iOS ecosystem praised for best-in-class security",
    "Apple launches new subscription bundle with price advantage",
    "Apple boosts shareholder returns through aggressive buybacks",
    "Apple signs strategic deal with Salesforce on enterprise apps",
    "Apple leads Dow Jones rally after strong guidance",
    "Apple to host record-setting WWDC with major product reveals",
    "Apple stock finishes week up 4% on investor optimism"
]

# Generate DataFrame
positive = pd.DataFrame({
    'date': [datetime.now().date()] * len(positive_headlines),
    'title': positive_headlines,
    'url': ['https://www.example.com/apple-positive-news'] * len(positive_headlines),
    'Label': [1] * len(positive_headlines)  # 1 for positive
})

positive.head

# Combine all news sources
combined_news = pd.concat([news_df, google_df, negative_news_df, positive_news_df, positive], ignore_index=True)

# Standardize date column
combined_news['date'] = pd.to_datetime(combined_news['date'])
combined_news.drop_duplicates(subset='title', inplace=True)
print(df.columns)

# Flatten MultiIndex columns
df.columns = ['_'.join(col).strip('_') for col in df.columns.values]

# Check new column names
print(df.columns)

combined_news['date'].isnull().sum()
print("🧹 After cleaning, headlines left:", len(combined_news))
combined_news = combined_news.dropna(subset=['date'])
combined_news = combined_news.dropna(subset=['date'])

df = df.loc[:, ~df.columns.duplicated()]

combined_news = combined_news.sort_values('date')
df = df.sort_values('Date')  # or 'Date_' if you're using that column

combined_news = combined_news.sort_values('date')
df = df.sort_values('Date')  # or 'Date_' if you're using that column

merged = pd.merge_asof(
    combined_news,
    df,
    left_on='date',
    right_on='Date',
    direction='backward'
)
merged = merged.copy()
merged['Label'] = merged['Label_x'].fillna(merged['Label_y'])
merged = merged.drop(columns=['Label_x', 'Label_y'])  # Clean up
print(merged['Label'].value_counts())
merged.info
def create_label(ret, threshold=0.01):
    if ret > threshold:
        return 1  # Price went up
    elif ret < -threshold:
        return 0  # Price went down
    else:
        return -1  # Neutral movement (optional)

merged['Label'] = merged['Return'].apply(create_label)
filtered_df = merged.dropna(subset=['title', 'Label'])  # Ensure both columns are present
filtered_df = filtered_df[filtered_df['Label'].isin([0, 1])]  # Optional: remove NaNs or unexpected labels
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=300, stop_words='english')
X_tfidf = vectorizer.fit_transform(merged['title'])
y = merged['Label'].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, stratify=y, random_state=42
)
import numpy as np

label_mapping = {-1: 0, 0: 1, 1: 2}
map_func = np.vectorize(lambda x: label_mapping[x])
y_train = map_func(y_train)
y_test = map_func(y_test)
import pandas as pd

label_mapping = {-1: 0, 0: 1, 1: 2}
y_train = pd.Series(y_train).map(label_mapping).values
y_test = pd.Series(y_test).map(label_mapping).values

print("NaNs in y_train:", np.isnan(y_train).sum())
print("NaNs in y_test:", np.isnan(y_test).sum())

# Assuming df is your merged DataFrame
df = df.dropna(subset=['Label'])  # Drop rows where label is missing

# 1. Remove rows with missing or invalid data
df_clean = merged.dropna(subset=['title', 'Label'])

# 2. Vectorize titles
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
X = vectorizer.fit_transform(df_clean['title'])

# 3. Prepare labels — and if needed, remap -1/0/1 to 0/1/2
label_mapping = {-1: 0, 0: 1, 1: 2}
y = df_clean['Label'].map(label_mapping).astype(int)

print("✅ X:", X.shape)
print("✅ y:", y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

from xgboost import XGBClassifier

model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("✅ Classification Report:\n", classification_report(y_test, y_pred))
import joblib

joblib.dump(model, "xgb_news_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
print("✅ Model and vectorizer saved.")

def predict_news_impact(text, vectorizer, model):
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0][pred]
    label = {0: "📉 DOWN", 1: "📊 NEUTRAL", 2: "📈 UP"}.get(pred, "Unknown")
    return label, round(proba * 100, 2)

# Example
headline = "Apple's quarterly sales going higher than ever before"
label, confidence = predict_news_impact(headline, vectorizer, model)
print(f"📰 Headline: {headline}\n🔮 Prediction: {label} (Confidence: {confidence}%)")
