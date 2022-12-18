import streamlit as st
import numpy as np
import pandas as pd
import pickle
import time

with st.spinner('Please wait, Data loading...'):
    time.sleep(22)
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

st.write('### Restaurant Recommender System')

resto_list = resto_df['title'].values
selected_resto = st.selectbox(
    "Type or select a resto name from the dropdown",
    resto_list
)
