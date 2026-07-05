from google.colab import drive
import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.model_selection import train_test_split

# Mount Google Drive
drive.mount('/content/drive')

# Load data from a single Excel file located in Google Drive
data = pd.read_excel('/content/drive/My Drive/combined_excel_file.xlsx', engine='openpyxl')

# Assuming 'data' is a pandas DataFrame with 'text' and 'label' columns preprocessed
labels = data['label'].tolist()

# Create label to index mapping
label_to_index = {
    'worried': 0, 'critical': 1, 'neutral': 2, 'hopeful': 3, 
    'happiness': 4
}

# Convert labels to indices, filtering out invalid labels
valid_indices = [i for i, label in enumerate(labels) if label in label_to_index]
indexed_labels = [label_to_index[labels[i]] for i in valid_indices]

# Filter texts to only include those with valid corresponding labels
filtered_texts = [data['text'].iloc[i] for i in valid_indices]

# Tokenize the filtered texts using a multilingual tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
tokenized_texts = tokenizer(filtered_texts, padding=True, truncation=True, return_tensors='tf', max_length=512)

# Extract input_ids and attention_masks for train_test_split
input_ids = tokenized_texts['input_ids'].numpy()
attention_masks = tokenized_texts['attention_mask'].numpy()
label_array = np.array(indexed_labels)
print (label_array)
print(input_ids)
print(attention_masks)

# Split the data into training and testing sets
train_inputs, test_inputs, train_labels, test_labels, train_masks, test_masks = train_test_split(
    input_ids, label_array, attention_masks, test_size=0.2, random_state=42
)

# Load pretrained BERT model with the specified number of labels
model = TFBertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=len(label_to_index))

# Prepare for training
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# Train the model
model.fit([train_inputs, train_masks], train_labels, validation_data=([test_inputs, test_masks], test_labels), epochs=6, batch_size=4)

# Save the trained model to Google Drive
model_save_path = '/content/drive/My Drive/saved_bert_model'
model.save_pretrained(model_save_path)

# Function to predict new texts
def predict_texts(texts):
    new_tokenized_texts = tokenizer(texts, padding=True, truncation=True, return_tensors='tf', max_length=512)
    predictions = model.predict([new_tokenized_texts['input_ids'], new_tokenized_texts['attention_mask']])
    predicted_labels = np.argmax(predictions.logits, axis=1)
    probabilities = tf.nn.softmax(predictions.logits, axis=1).numpy()

    index_to_label = {v: k for k, v in label_to_index.items()}
    results = []
    for i, text in enumerate(texts):
        predicted_label = index_to_label[predicted_labels[i]]
        probability = probabilities[i, predicted_labels[i]]
        results.append((text, predicted_label, probability))
    return results

# Example usage
new_texts = ["我讨厌这个产品！", "Dieses Produkt ist großartig!"]
predictions = predict_texts(new_texts)
for text, label, prob in predictions:
    print(f"Text: {text}, Predicted Sentiment: {label}, Probability: {prob:.4f}")