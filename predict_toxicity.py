# Import necessary libraries
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import TextVectorization
import gradio as gr
import pandas as pd

# Load your pre-trained model
model = tf.keras.models.load_model('toxicity.h5')

# Initialize the vectorizer and adapt it with the same parameters and vocabulary used for training
MAX_FEATURES = 200000
vectorizer = TextVectorization(max_tokens=MAX_FEATURES,
                               output_sequence_length=1800,
                               output_mode='int')

# Load and limit the vocabulary
with open('vocab.txt', 'r', encoding='utf-8') as f:
    vocab = f.readlines()

# Remove duplicates, sort, and limit to 200,000
unique_vocab = sorted(set(word.strip() for word in vocab))[:200000]

# Save the top 200,000 words back to vocab.txt
with open('vocab.txt', 'w', encoding='utf-8') as f:
    for word in unique_vocab:
        f.write(word + '\n')

print("Vocabulary limited to 200,000 words and saved to vocab.txt.")



vectorizer.set_vocabulary(vocab)

# Define prediction function
def score_comment(comment):
    vectorized_comment = vectorizer([comment])
    results = model.predict(vectorized_comment)

    text = ''
    # Load the original labels (assuming they are saved with the columns)
    df = pd.read_csv(r'C:\Users\KHAIRI\Downloads\comments\train.csv')
    for idx, col in enumerate(df.columns[2:]):
        text += '{}: {}\n'.format(col, results[0][idx] > 0.5)

    return text

# Set up Gradio interface for interactive predictions
interface = gr.Interface(fn=score_comment,
                         inputs=gr.Textbox(lines=2, placeholder='Enter a comment'),
                         outputs='text')

# Launch the interface
interface.launch()
