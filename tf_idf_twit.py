import streamlit as st
import numpy as np
import joblib

def cosine_similarity(query, pool, k=10):
    return np.argsort(-pool.dot(query.T).toarray().squeeze(-1))[0:k]

x_loaded = np.load('x.npy',allow_pickle=True)
embeddings_t = x_loaded.item()

y_loaded = np.load('y.npy',allow_pickle=True)
lables_t = y_loaded

x_loaded = np.load('x_mc.npy',allow_pickle=True)
embeddings_mc = x_loaded.item()

y_loaded = np.load('y_mc.npy',allow_pickle=True)
lables_mc = y_loaded


MC_20 = ['❤','😂', '👍', '🙏', '🙌', '😘', '😍', '😊', '🔥', '👏', '👌', '💪', '👊', '😉', '🎉', '😎',"😁", "💯", "😜", "👀"]
inv_voc = {}
for i,m in enumerate(MC_20):
  inv_voc[i] = m

#Interface
st.markdown('## Emoji Prediction by two differently trained TF-IDF algorithms')
inn = st.text_input('Enter some text')
#Predict button
if st.button('Predict'):
    model_t = joblib.load('tf-idf_twit_model.pkl')
    model_mc = joblib.load('tf-idf_mc_model.pkl')
    qry_t = model_t.transform([inn])
    qry_mc = model_mc.transform([inn])
    index_t = cosine_similarity(qry_t,embeddings_t)[0]
    index_mc = cosine_similarity(qry_mc,embeddings_mc)[0]
    
    st.markdown(f'### Prediction for TF-IDF trained on Twitter dataset is: {inv_voc[lables_t[index_t]]}')
    st.markdown(f'### Prediction for TF-IDF trained on MC_20 dataset is: {inv_voc[lables_mc[index_mc]]}')