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
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')


option = st.sidebar.selectbox(
    'Silakan pilih:',
    ('Home', 'Dataset', 'Modeling', 'Recommendation')
)

if option == 'Home' or option == '':
    st.write("""# Halaman Utama""")  # menampilkan halaman utama
    st.write("""
    ## PROJECT - Senior Data Scientist [Recommendation System]
    ### Study Case Restaurant
    ---
    #### Aziz Hendra Atmaja
    ---
    """)
elif option == 'Dataset':
    st.write("""## Read Dataset""")  # menampilkan judul halaman dataframe

    # membuat dataframe dengan pandas yang terdiri dari 2 kolom dan 4 baris data
    # df = pd.DataFrame({
    #    'Column 1':[1,2,3,4],
    #    'Column 2':[10,12,14,16]
    # })
    # df #menampilkan dataframe
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
