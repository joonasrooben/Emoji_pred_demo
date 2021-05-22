import streamlit as st
import numpy as np
import joblib

def cosine_similarity(query, pool, k=10):
    return np.argsort(-pool.dot(query.T).toarray().squeeze(-1))[0:k]

x_loaded = np.load('x.npy',allow_pickle=True)
embeddings = x_loaded.item()

y_loaded = np.load('y.npy',allow_pickle=True)
lables = y_loaded


MC_20 = ['❤','😂', '👍', '🙏', '🙌', '😘', '😍', '😊', '🔥', '👏', '👌', '💪', '👊', '😉', '🎉', '😎',"😁", "💯", "😜", "👀"]
inv_voc = {}
for i,m in enumerate(MC_20):
  inv_voc[i] = m

#Interface
st.markdown('## Emoji Prediction')
inn = st.text_input('Enter some text')
#Predict button
if st.button('Predict'):
    model = joblib.load('tf-idf_twit_model.pkl')
    qry = model.transform([inn])
    index = cosine_similarity(qry,embeddings)[0]
    
    st.markdown(f'### Prediction is {inv[lables[index]]}')
