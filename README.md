# Sentiment-Analysis-of-IMDB-Movie-Reviews
This repository contains two Python implementations for binary sentiment analysis of the IMDB movie reviews dataset. The goal is to classify reviews as either positive or negative. Each implementation focuses on a different approach to text representation and model design.

# **Overview**
## **Model Versions:**

**Version 1:** A basic LSTM-based model with trainable embeddings initialized randomly.

**Version 2:** An LSTM-based model using pre-trained GloVe embeddings for better word representation.
Both versions showcase:

## **Preprocessing of the IMDB dataset**
Definition of model architectures with detailed customization
Training with techniques to optimize performance (callbacks, dropout, etc.)
Evaluation on the test set
Visualization of training and validation metrics
Example predictions on test samples

## **Dataset**
The IMDB dataset consists of 50,000 movie reviews, evenly split into:

25,000 training samples

25,000 testing samples

Only the top 10,000 most frequently used words in the dataset are retained for tokenization. The reviews are encoded as sequences of integers, where each integer represents a word's index in a dictionary. Input sequences are padded to a uniform length of 500 words for consistency across the models.

## **Code Walkthrough**
### **1.Data Preparation**

**Loading the IMDB Dataset:** The dataset is loaded using keras.datasets.imdb. Only the most frequent max_features=10,000 words are used.

**Padding Sequences:** The sequences are padded to a fixed length (maxlen=500) using pad_sequences, ensuring uniform input size.
### **2.Model Architectures**

#### **Version 1: Basic LSTM Model**

This model uses a randomly initialized embedding layer that is trainable during the learning process. Key components include:

**Embedding Layer:** Converts integer-encoded input sequences into dense vectors of size embedding_dim=32.

**LSTM Layer:** Extracts sequential patterns from the text with lstm_units=32.

**Dropout Layers:** Applied after the embedding and LSTM layers to reduce overfitting.

**Output Layer:** A Dense layer with 1 neuron and a sigmoid activation for binary classification.


#### **Version 2: LSTM Model with Pre-trained GloVe Embeddings**

This version incorporates GloVe word embeddings for better initialization of the embedding layer. Key enhancements include:

**Pre-trained Embeddings:** GloVe embeddings (glove.6B.100d.txt) are loaded and mapped to the IMDB vocabulary.

**Fixed Embeddings:** The embedding layer uses the pre-trained GloVe weights and is set as trainable=False to preserve their semantic structure.

**LSTM and Dropout Layers:** Similar to Version 1, with lstm_units=32.

**Output Layer:** A Dense layer with 1 neuron and sigmoid activation for binary classification.

### **3. Training**
Both models are trained using the following:

**Loss Function:** binary_crossentropy for binary classification tasks.

**Optimizer:** rmsprop for efficient weight updates.

**Callbacks:**

ModelCheckpoint to save the model with the best validation accuracy.

ReduceLROnPlateau to reduce the learning rate when validation loss plateaus.

### **4. Visualization**

During training, the models track:

Training accuracy and validation accuracy across epochs.

Training loss and validation loss across epochs.

These metrics are visualized using matplotlib to identify overfitting or underfitting.

### **5. Evaluation**

The trained models are evaluated on the test set to calculate loss and accuracy.

Example test samples are passed through the model to predict sentiments.

### **6. GloVe Embeddings**
For Version 2, GloVe embeddings must be downloaded from the GloVe Official Website. These embeddings map words to a 100-dimensional vector space, enhancing the model's understanding of semantic relationships.
