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

   The data was obtained from [jigsaw toxic classification challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge)

    
