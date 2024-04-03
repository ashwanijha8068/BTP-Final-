import streamlit as st
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import tempfile
import os

# Load tokenizer and models
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model_1 = BertForSequenceClassification.from_pretrained("Model_1")
model_2 = BertForSequenceClassification.from_pretrained("Model_2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_1.to(device)
model_2.to(device)


def predict(file, model_id):
    df = pd.read_csv(file)
    if model_id == 1:
        model = model_1
        column_name = 'clean_tweet'
    elif model_id == 2:
        model = model_2
        column_name = 'text'
    else:
        return "Invalid model ID"

    # Tokenize the input text and convert to tensors
    encoding = tokenizer(df[column_name].tolist(), padding=True, truncation=True, return_tensors='pt')
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # Make predictions
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=1).cpu().numpy()

    if model_id==1:
        annotations = ['irrelevant' if p == 0 else 'relevant' for p in predictions]
    else:
        annotations = ['business' if p == 0 else 'threat' for p in predictions]



    # Add annotations to the original DataFrame
    df['relevant_or_irrelevant'] = annotations

    return df


st.title('Model Deployment API')
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
model_id = st.selectbox("Select Model ID", [1, 2])

if uploaded_file is not None:
    df = predict(uploaded_file, model_id)

    # Display only 'clean_tweet' and 'relevant_or_irrelevant' columns
    st.write(df[['clean_tweet', 'relevant_or_irrelevant']])

    # Filter out only the relevant tweets for the download
    relevant_tweets = df[df['relevant_or_irrelevant'] == 'relevant']

    # Rename 'clean_tweet' column to 'text' for the downloaded file
    relevant_tweets.rename(columns={'clean_tweet': 'text'}, inplace=True)

    # Button to download the DataFrame as a CSV file
    csv = relevant_tweets[['text', 'relevant_or_irrelevant']].to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download Relevant Tweets",
        csv,
        "relevant_tweets.csv",
        "text/csv",
        key='download-csv'
    )
    if st.button("Feed this dataset to Model 2 for Multi-Classification"):
        # Save the CSV data to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp:
            temp.write(csv)
            temp_path = temp.name

        # Pass the path of the temporary file to the predict function
        df_model_2 = predict(temp_path, 2)
        df_model_2.rename(columns={'relevant_or_irrelevant': 'type'}, inplace=True)
        st.write(df_model_2[['text', 'type']])
        # Button to download the DataFrame as a CSV file
        csv_model_2 = df_model_2[['text', 'type']].to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Multi-Classified Tweets",
            csv_model_2,
            "multi_classified_tweets.csv",
            "text/csv",
            key='download-csv-model-2'
        )

        # Clean up the temporary file
        os.unlink(temp_path)