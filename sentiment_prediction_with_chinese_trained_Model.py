#sentimentanalysiswithCHtrainedModel, chinese version 

import pandas as pd  
import tensorflow as tf  
from transformers import BertTokenizer, TFBertForSequenceClassification  
import numpy as np  
from google.colab import drive  

# Mount Google Drive  
drive.mount('/content/drive')  

# Load data from an Excel file located in Google Drive with German text  
data = pd.read_excel('/content/drive/My Drive/sentimentproject/combined2223CH_output.xlsx')  
new_texts = data['text'].values  # Get texts for prediction  

# Define label mappings  
label_to_index = {  
    'worried': 0,  
    'critical': 1,  
    'neutral': 2,  
    'hopeful': 3,  
    'happiness': 4  
}  

# Create an index to label mapping  
index_to_label = {index: label for label, index in label_to_index.items()}  

# Load a pre-trained BertTokenizer (using a general pre-trained model for German text)  
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')  

# Load the trained sentiment analysis model  
loaded_model = TFBertForSequenceClassification.from_pretrained('/content/drive/My Drive/sentimentproject/savedCH_model')  

# Function to predict classes for new texts  
def predict_texts(texts):  
    # Tokenize the input texts  
    tokenized_texts = tokenizer(texts.tolist(), padding=True, truncation=True, return_tensors='tf', max_length=512)  
    predictions = loaded_model.predict([tokenized_texts['input_ids'], tokenized_texts['attention_mask']]).logits  
    
    predicted_labels = np.argmax(predictions, axis=1)  # Get predicted class indices  
    probabilities = tf.nn.softmax(predictions, axis=1).numpy()  # Get predicted probabilities  

    results = []  
    for i, text in enumerate(texts):  
        predicted_label = index_to_label[predicted_labels[i]]  
        probability = probabilities[i, predicted_labels[i]]  
        
        # Create a one-hot encoded vector  
        one_hot = np.zeros(len(label_to_index))  
        one_hot[predicted_labels[i]] = 1  
        
        results.append((text, predicted_label, probability, one_hot))  
    return results  

# Make predictions on new texts  
predicted_results = predict_texts(new_texts)  

# Prepare data for saving  
final_results = []  
for res in predicted_results:  
    text, label, prob, one_hot = res  
    print(f'Text: "{text}" | Predicted Label: {label} | Probability: {prob:.4f} | One-Hot Encoding: {one_hot}')  
    
    # Append results for saving  
    final_results.append({  
        'Text': text,  
        'Predicted Label': label,  
        'Probability': prob,  
        'One-Hot Encoding': one_hot  
    })  

# Convert results to DataFrame for saving  
results_df = pd.DataFrame(final_results)  

# Save to Excel (We will flatten the one-hot encoding for better visualization)  
results_df[['Text', 'Predicted Label', 'Probability']] = results_df[['Text', 'Predicted Label', 'Probability']]  
results_df[['Worried', 'Critical', 'Neutral', 'Hopeful', 'Happiness']] = pd.DataFrame(results_df['One-Hot Encoding'].tolist(), index=results_df.index)  

# Drop the 'One-Hot Encoding' list column  
results_df.drop('One-Hot Encoding', axis=1, inplace=True)  

# Save results to an Excel file  
results_df.to_excel('/content/drive/My Drive/sentimentproject/predicted2223CH_results_one_hot.xlsx', index=False)