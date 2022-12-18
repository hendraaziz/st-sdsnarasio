import streamlit as st
import numpy as np
import pandas as pd
import pickle
import time

with st.spinner('Please wait, Data loading...'):
    time.sleep(10)
resto = pickle.load(
    open('/home/conda/st-sds/data/resto_dataset', 'rb'))

cosine_similarities = pickle.load(
    open('/home/conda/st-sds/data/cosine_resto_model', 'rb'))


#
resto_df = pd.DataFrame(resto)

st.write("""# Welcome""")  # menampilkan halaman utama
st.write("""
## PROJECT - Senior Data Scientist [Recommendation System]
### Study Case Restaurant
---
#### Aziz Hendra Atmaja
---
""")

st.write("""## Read Dataset""")  # menampilkan judul halaman dataframe

st.write('### Import Package')

import_pack = '''
import streamlit as st
import numpy as np
import pandas as pd
import pickle

resto = pickle.load(open('resto_dataset','rb'))

cosine_similarities = pickle.load(open('cosine_resto_model','rb'))

resto_df = pd.DataFrame(resto)

    '''

st.code(import_pack, language="python")

st.write('### Load Dataset & Model')
read_dataset = '''
resto = pickle.load(open('resto_dataset','rb'))

cosine_similarities = pickle.load(open('cosine_resto_model','rb'))

resto_df = pd.DataFrame(resto)
'''
# st.write('### Read Dataset')
st.code(read_dataset, language="python")
# st.success('Done!')

st.write(resto_df.head())

# set index pada dataframe sample_resto
resto_df.set_index('nama', inplace=True)

# mengubah menjadi data series
seri = pd.Series(resto_df.index)

# Fungsi Rekomendasi


def rekomendasi_review(name, cosine_similarities=cosine_similarities):

    # Create a list to put top restaurants
    rekomend_byresto = []

    # Find the index of the hotel entered
    idx = seri[seri == name].index[0]

    # Find the restaurants with a similar cosine-sim value and order them from bigges number
    score_series = pd.Series(
        cosine_similarities[idx]).sort_values(ascending=False)

    # Extract top 30 restaurant indexes with a similar cosine-sim value
    top30_indexes = list(score_series.iloc[0:31].index)

    # Names of the top 30 restaurants
    for each in top30_indexes:
        rekomend_byresto.append(list(resto_df.index)[each])

    # Creating the new data set to show similar restaurants
    rekom_resto = pd.DataFrame(columns=['jenis_makanan', 'rating', 'kota'])

    # Create the top 30 similar restaurants with some of their columns
    for each in rekomend_byresto:
        rekom_resto = rekom_resto.append(pd.DataFrame(
            resto_df[['jenis_makanan', 'rating', 'kota']][resto_df.index == each].sample()))

    # Drop the same named restaurants and sort only the top 10 by the highest rating
    rekom_resto = rekom_resto.drop_duplicates(
        subset=['jenis_makanan', 'rating', 'kota'], keep=False)
    rekom_resto = rekom_resto.sort_values(
        by='rating', ascending=False).head(10)

    print('%s RESTO REKOMENDASI MEMILIKI KEMIRIPAN REVIEW DENGAN %s : ' %
          (str(len(rekom_resto)), name))

    return rekom_resto


st.write('### Restaurant Recommender System')

resto_list = resto['nama'].values
selected_resto = st.selectbox(
    "Type or select a resto name from the dropdown",
    resto_list
)

if st.button('Recommend'):
    rekomend = rekomendasi_review(selected_resto)
    st.write(rekomend)
