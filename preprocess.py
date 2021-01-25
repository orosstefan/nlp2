import nltk
import pandas as pd

# Change to nltk data path and uncomment download
nltk.data.path.append('/Users/orosstefan/Dev/nltk_data')
# nltk.download()


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
porter = PorterStemmer()
dataframe = pd.read_csv('dataset/Reviews.csv')


def preprocess_labels(reviews):
    # Tokenize Lyrics
    tokens = word_tokenize(reviews)
    table = str.maketrans('', '', string.punctuation)

    labels_list = []
    for token in tokens:
        # Convert to lower case
        token = token.lower()

        # Filter punctuation
        stripped = token.translate(table)

        labels_list.append(stripped)
    labels_list = list(filter(None, labels_list))
    return labels_list
def preprocess_inputs(reviews):

    # Tokenize Lyrics
    tokens = word_tokenize(reviews)
    table = str.maketrans('', '', string.punctuation)
    stop_words = stopwords.words('english')

    inputs_list = []
    for token in tokens:
        # Convert to lower case
        token = token.lower()

        # Filter punctuation
        stripped = token.translate(table)

        # Remove stopwords
        if stripped in stop_words:
            continue

        # Stemming
        stemmed = porter.stem(stripped)
        inputs_list.append(stemmed)
    inputs_list = list(filter(None, inputs_list))
    return inputs_list

def preprocess_df(df, filename):
    print(len(df))
    # Drop NaN
    df = df.dropna(subset=['Text'])
    df = df.dropna(subset=['Summary'])
    print(len(df))
    df = df.drop(['Id', 'ProductId', 'UserId', 'ProfileName', 'HelpfulnessNumerator', 'HelpfulnessDenominator', 'Score', 'Time'], axis=1)

    text = df[['Text']]
    summary = df[['Summary']]

    # Clean lyrics stemmed
    df['Text'] = text['Text'].apply(lambda x: preprocess_inputs(x)).to_frame()
    df['Summary'] = summary['Summary'].apply(lambda x: preprocess_labels(x)).to_frame()
    df.to_csv(filename + '.csv', header=True, index=False)


preprocess_df(dataframe, 'dataset/preprocessed')
