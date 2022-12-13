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


st.write("""
## PROJECT - Senior Data Scientist [Recommendation System]
### Study Case Restaurant
---
#### Aziz Hendra Atmaja
---
""")

option = st.sidebar.selectbox(
    'Silakan pilih:',
    ('Home', 'Dataframe', 'Modeling', 'Recommendation')
)
