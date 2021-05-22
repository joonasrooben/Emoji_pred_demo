import streamlit as st
import numpy as np
import torch
from transformers import DistilBertForSequenceClassification, AdamW, DistilBertConfig
from transformers import DistilBertTokenizer
import joblib

def predict_emoji(model, sentence, voc):
    model.eval()
    text = tokenizer.encode(sentence, add_special_tokens = True, max_length=128, padding= "max_length", truncation = True, return_tensors = "pt")
    attention_mask = text > 0
    attention_mask = attention_mask.squeeze().to(device)
    text = text.to(device)
    bind = (text, attention_mask)
    with torch.no_grad():
      outputs = model(bind[0], bind[1])[0]
    prediction, indices = torch.topk(torch.sigmoid(outputs), 1)
    indices = indices.detach().cpu().numpy()
    mojis = [voc[pred]for pred in indices[0]]
    return mojis
    
voc = {'❤': 4, '🎉': 11, '👀': 16, '👊': 1, '👌': 13, '👍': 15, '👏': 6, '💪': 0, '💯': 12, '🔥': 7, '😁': 9, '😂': 19, '😉': 10, '😊': 17, '😍': 18, '😎': 8, '😘': 14, '😜': 5, '🙌': 2, '🙏': 3}
inv_voc = {}
for m in voc:
  inv_voc[voc[m]] = m
  
output_dir = './bert_twitter'
model = DistilBertForSequenceClassification.from_pretrained(output_dir)
tokenizer = DistilBertTokenizer.from_pretrained(output_dir)

#Interface
st.markdown('## Emoji Prediction')
inn = st.text_input('Enter some text')

#Predict button
if st.button('Predict'):
    
    # Copy the model to the GPU.
    model.to(device)
    st.markdown(f'### Prediction is {predict_emoji(model, inn, inv_voc)}')