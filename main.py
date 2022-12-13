from nltk.util import ngrams
import nltk
from unidecode import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
from nltk.corpus import stopwords
import re
import streamlit as st

import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
import time
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')


with st.spinner('Please wait...'):
    time.sleep(15)
resto = pd.read_csv('../dataset/sysrec_1009.csv')
option = st.sidebar.selectbox(
    'Silakan pilih:',
    ('Home', 'Dataset', 'Pre-Processing & EDA', 'Modeling', 'Recommendation')
)
# Vocab untuk stopwords
stops = set(nltk.corpus.stopwords.words("english"))

# Format html
html_tag = re.compile(r'<.*?>')
http_link = re.compile(r'https://\S+')
www_link = re.compile(r'www\.\S+')

# Menghilangkan akun user
user_name = re.compile(r'\@[a-z0-9]+')

# Tanda baca yang tidak diperlukan
punctuation = re.compile(r'[^\w\s\n]')

# Function untuk memproses cleaning teks data


def data_cleaning(text, stopwords=False):
    # unicode text data
    text = unidecode(str(text))

    # lower casting
    text = text.lower()

    # menghilangkan html tag
    text = re.sub(html_tag, r'', text)

    # menghilangkan url
    text = re.sub(http_link, r'', text)
    text = re.sub(www_link, r'', text)

    # menghilangkan user name
    text = re.sub(user_name, r'', text)

    # menghilangkan tanda baca
    text = re.sub(punctuation, r'', text)

    # Tokenize
    text = text.split()

    # remove stopword
    if stopwords:
        text = [w for w in text if not w in stops]
    return text


if option == 'Home' or option == '':
    st.write("""# Welcome""")  # menampilkan halaman utama
    st.write("""
    ## PROJECT - Senior Data Scientist [Recommendation System]
    ### Study Case Restaurant
    ---
    #### Aziz Hendra Atmaja
    ---
    """)
elif option == 'Dataset':
    st.write("""## Read Dataset""")  # menampilkan judul halaman dataframe

    st.write('### Import Package')

    import_pack = '''
    import numpy as np
    import pandas as pd
    import seaborn as sb
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    import warnings
    warnings.filterwarnings('always')
    warnings.filterwarnings('ignore')
    from nltk.corpus import stopwords
    from sklearn.metrics.pairwise import linear_kernel
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from unidecode import unidecode

    import nltk
    from nltk.util import ngrams
    '''

    st.code(import_pack, language="python")

    read_dataset = '''
    resto = pd.read_csv('../dataset/sysrec_1009.csv')
    resto.head()'''
    st.write('### Read Dataset')
    st.code(read_dataset, language="python")
    # st.success('Done!')

    st.write(resto.head())

elif option == 'Pre-Processing & EDA':
    # cek prosentase data kosong
    resto_null = round(100*(resto.isnull().sum())/len(resto), 2)
    st.write('Prosentase data kosong')
    resto_null

    # hapus kolom menu_favorit karena data kosong lebih dari 50%
    resto = resto.drop(['menu_favorit'], axis=1)

    # cek data duplikat
    resto.duplicated().sum()

    # hapus data duplikat
    resto.drop_duplicates(inplace=True)

    # hapus NaN values dari dataset
    resto.isnull().sum()
    resto.dropna(how='any', inplace=True)

    # cek data rating
    st.write('cek Rating')
    resto['rating'].unique()

    # Menghapus karakter '/5' dari kolom rating
    resto = resto.loc[resto.rating != 'NEW']
    resto = resto.loc[resto.rating != '-'].reset_index(drop=True)
    def hapus_slash(x): return x.replace('/5', '') if type(x) == np.str else x
    resto.rating = resto.rating.apply(hapus_slash).str.strip().astype('float')
    st.write('Head Rating')
    resto['rating'].head()

    st.write('view after clean /5')
    st.write(resto.head())

    # Menerapkan cleaning pada kolom review
    resto['review'] = resto['review'].apply(
        lambda x: data_cleaning(x, stopwords=True))
    st.write('view after cleaning')
    resto.head()

    # melihat sebaran data dari rating resto

    plt.figure(figsize=(8, 6))
    sns.countplot(x=resto['rating'], data=resto)
    plt.tight_layout()

elif option == 'Modeling':
    st.write("""## Draw Modeling""")  # menampilkan judul halaman

    # membuat variabel chart data yang berisi data dari dataframe
    # data berupa angka acak yang di-generate menggunakan numpy
    # data terdiri dari 2 kolom dan 20 baris
    # chart_data = pd.DataFrame(
    #    np.random.randn(20,2),
    #    columns=['a','b']
    # )
    # menampilkan data dalam bentuk chart
    # st.line_chart(chart_data)
    # data dalam bentuk tabel
    # chart_data


elif option == 'Recommendation':
    st.write("""## Recommendation System""")  # menampilkan judul halaman
