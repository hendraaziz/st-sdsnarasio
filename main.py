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


option = st.sidebar.selectbox(
    'Silakan pilih:',
    ('Home', 'Dataset', 'Pre-Processing & EDA', 'Modeling', 'Recommendation')
)

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
    with st.spinner('Wait for it...'):
        time.sleep(11)
    st.success('Done!')

    resto = pd.read_csv('../dataset/sysrec_1009.csv')
    st.write(resto.head())
elif option == 'Pre-Processing & EDA':
    # cek prosentase data kosong
    resto_null = round(100*(resto.isnull().sum())/len(resto), 2)
    resto_null
    st.write(resto_null)
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
