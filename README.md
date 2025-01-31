# Comment Classifier Using LSTM

This repository contains a **Long Short-Term Memory (LSTM)** model that classifies text comments according to multiple toxicity labels:

- `toxic`
- `severe_toxic`
- `obscene`
- `threat`
- `insult`
- `identity_hate`

## Overview

The goal is to demonstrate how to build and train an LSTM model to detect and classify toxic comments. The model takes raw text, tokenizes and pads it, then outputs predictions on the six labels above.

Key features:

- **Tokenization**: Convert text into integer sequences  
- **Padding**: Ensure each sequence has a fixed length (250)  
- **Embedding Layer**: Transform word indices into embeddings (size: 128)  
- **LSTM Layer**: Extract sequential features (size: 60)  
- **Global Max Pooling**: Reduce dimensionality  
- **Dropout**: Mitigate overfitting  
- **Dense Layers**: Classify into 6 binary labels using a sigmoid activation  

## Installation & Requirements

1. **Python 3.7+** recommended  
2. Required libraries (install via `pip`):
   ```bash
   pip install numpy pandas matplotlib tensorflow
   ```

## Data
   The data was obtained from [jigsaw toxic classification challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge).

## Model Summary

Model: "model_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_3 (InputLayer)        [(None, 250)]             0         
                                                                 
 embedding_4 (Embedding)     (None, 250, 128)          2560000   
                                                                 
 lstm_layer (LSTM)           (None, 250, 60)           45360     
                                                                 
 global_max_pooling1d_2 (Glo (None, 60)                0         
balMaxPooling1D)                                                
                                                                 
 dropout_3 (Dropout)         (None, 60)                0         
                                                                 
 dense_2 (Dense)             (None, 50)                3050      
                                                                 
 dropout_5 (Dropout)         (None, 50)                0         
                                                                 
 dense_4 (Dense)             (None, 6)                 306       
                                                                 
=================================================================
Total params: 2,611,266
Trainable params: 2,611,266
Non-trainable params: 0

    
