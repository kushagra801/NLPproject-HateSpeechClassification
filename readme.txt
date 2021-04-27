• Sentiment classifier from scratch using Long Short Term Memory (LSTM) and word embeddings obtained by implementing
SkipGram algorithm to identify hate speech.
• Furthermore the accuracy was improved by implementing n-gram model using Convolutional Neural Network (CNN) and implementing
negative sampling to get the word embeddings. The model was able to achieve a Test Accuracy of 84%. The model
created could be trained on data set for any language. For our project HASOC Hindi data set and Bengali tweets data set were
used.

NNTI_final_project_task_1.ipynb = Original word embeddings using skipgram.

SGNS.ipynb = Word embeddings from Skipgram with Negative Sampling (task 3)

LSTM.py = original network for Task 2 for sentiment analysis.

DataPreprocess.py = Data cleaning and converting pandas data frame to torch text data frames.

TrainEvalLoops.py = functions to train and evaluate CNN model

model.py = CNN network for sentiment analysis improving on the result from LSTM

main.py = reading file, calling network, outputting results

Data Folder = contains original csv and tsv files for Hindi and Bengali. Model weights [word embeddings] for both Hindi and Bengali, from basic skiagram and for negative sampling. Contains print statements from running our main.py various ways. Stopwords for both Hindi and Bengali.


