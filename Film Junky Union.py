#!/usr/bin/env python
# coding: utf-8

# ##  Film Junky Union (Machine Learning for Texts) 

# # Project Statement

# The Film Junky Union, a new edgy community for classic movie enthusiasts, is developing a system for filtering and categorizing movie reviews. The goal is to train a model to automatically detect negative reviews. You'll be using a dataset of IMBD movie reviews with polarity labelling to build a model for classifying positive and negative reviews. It will need to have an F1 score of at least 0.85.
# 
# The following will be shown in the project:
# $1. Load the data.
# $2. Preprocess the data.
# $3. Conduct an EDA with a conclusion on the class imbalance.
# $4. Preprocess the data for modeling.
# $5. Train at least three different models for the given train dataset.
# $6. Test the models for the given test dataset.
# $7. Compose a few reviews and classify them with all the models.
# $8. Check for differences between the testing results of models in the above two points. Try to explain them.
# $9. Present  findings.
# 
# The data is stored in the imdb_reviews.tsv file.
# Here's the description of the selected fields:
# 
# review: the review text
# pos: the target, '0' for negative and '1' for positive
# ds_part: 'train'/'test' for the train/test part of dataset, correspondingly

# ## Initialization

# In[1]:


import math

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from tqdm.auto import tqdm


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'png'")
# the next line provides graphs of better quality on HiDPI screens
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

plt.style.use('seaborn')


# In[3]:


# this is to use progress_apply
tqdm.pandas()


# ## Load Data

# In[4]:


df_reviews = pd.read_csv('/datasets/imdb_reviews.tsv', sep='\t', dtype={'votes': 'Int64'})


# In[5]:


#display info to see datatypes
df_reviews.info()


# In[6]:


#search for duplicated reviews
df_reviews.duplicated().sum()


# In[7]:


#look for empty reviews
df_reviews.isna().sum()


# In[8]:


empty=df_reviews.isna()
empty.head(2)


# In[9]:


df_reviews=df_reviews.dropna()
df_reviews.isna().sum()


# ## EDA

# Check the number of movies and reviews over years.

# In[10]:


fig, axs = plt.subplots(2, 1, figsize=(16, 8))

ax = axs[0]

dft1 = df_reviews[['tconst', 'start_year']].drop_duplicates() \
    ['start_year'].value_counts().sort_index()
dft1 = dft1.reindex(index=np.arange(dft1.index.min(), max(dft1.index.max(), 2021))).fillna(0)
dft1.plot(kind='bar', ax=ax)
ax.set_title('Number of Movies Over Years')

ax = axs[1]

dft2 = df_reviews.groupby(['start_year', 'pos'])['pos'].count().unstack()
dft2 = dft2.reindex(index=np.arange(dft2.index.min(), max(dft2.index.max(), 2021))).fillna(0)

dft2.plot(kind='bar', stacked=True, label='#reviews (neg, pos)', ax=ax)

dft2 = df_reviews['start_year'].value_counts().sort_index()
dft2 = dft2.reindex(index=np.arange(dft2.index.min(), max(dft2.index.max(), 2021))).fillna(0)
dft3 = (dft2/dft1).fillna(0)
axt = ax.twinx()
dft3.reset_index(drop=True).rolling(5).mean().plot(color='orange', label='reviews per movie (avg over 5 years)', ax=axt)

lines, labels = axt.get_legend_handles_labels()
ax.legend(lines, labels, loc='upper left')

ax.set_title('Number of Reviews Over Years')

fig.tight_layout()


# Let's check the distribution of number of reviews per movie with the exact counting and KDE (just to learn how it may differ from the exact counting)

# In[11]:


fig, axs = plt.subplots(1, 2, figsize=(16, 5))

ax = axs[0]
dft = df_reviews.groupby('tconst')['review'].count() \
    .value_counts() \
    .sort_index()
dft.plot.bar(ax=ax)
ax.set_title('Bar Plot of #Reviews Per Movie')

ax = axs[1]
dft = df_reviews.groupby('tconst')['review'].count()
sns.kdeplot(dft, ax=ax)
ax.set_title('KDE Plot of #Reviews Per Movie')

fig.tight_layout()


# A KDE plot shows the probability density of a continuous variable by creating a smooth curve over the data points.  This does not assume a specific distribution shape for the data. Both charts above showcase that movies earlier in the dataframe have a larger amount of reviews.  The first 5 movies are shown on the KDE plot as he mot frequently reviewed.  Another peak occurs at movie 30.  The curve skews to the left.   

# In[12]:


df_reviews['pos'].value_counts()


# The reviews in the dataframe are quite closely split between being positive and negative. 

# In[13]:


fig, axs = plt.subplots(1, 2, figsize=(12, 4))

ax = axs[0]
dft = df_reviews.query('ds_part == "train"')['rating'].value_counts().sort_index()
dft = dft.reindex(index=np.arange(min(dft.index.min(), 1), max(dft.index.max(), 11))).fillna(0)
dft.plot.bar(ax=ax)
ax.set_ylim([0, 5000])
ax.set_title('The train set: distribution of ratings')

ax = axs[1]
dft = df_reviews.query('ds_part == "test"')['rating'].value_counts().sort_index()
dft = dft.reindex(index=np.arange(min(dft.index.min(), 1), max(dft.index.max(), 11))).fillna(0)
dft.plot.bar(ax=ax)
ax.set_ylim([0, 5000])
ax.set_title('The test set: distribution of ratings')

fig.tight_layout()


# The train and test set look identical in their distribution of ratings.  This means that once the models have been trained it should be accurate being run on the test set. 

# Distribution of negative and positive reviews over the years for two parts of the dataset

# In[14]:


fig, axs = plt.subplots(2, 2, figsize=(16, 8), gridspec_kw=dict(width_ratios=(2, 1), height_ratios=(1, 1)))

ax = axs[0][0]

dft = df_reviews.query('ds_part == "train"').groupby(['start_year', 'pos'])['pos'].count().unstack()
dft.index = dft.index.astype('int')
dft = dft.reindex(index=np.arange(dft.index.min(), max(dft.index.max(), 2020))).fillna(0)
dft.plot(kind='bar', stacked=True, ax=ax)
ax.set_title('The train set: number of reviews of different polarities per year')

ax = axs[0][1]

dft = df_reviews.query('ds_part == "train"').groupby(['tconst', 'pos'])['pos'].count().unstack()
sns.kdeplot(dft[0], color='blue', label='negative', kernel='epa', ax=ax)
sns.kdeplot(dft[1], color='green', label='positive', kernel='epa', ax=ax)
ax.legend()
ax.set_title('The train set: distribution of different polarities per movie')

ax = axs[1][0]

dft = df_reviews.query('ds_part == "test"').groupby(['start_year', 'pos'])['pos'].count().unstack()
dft.index = dft.index.astype('int')
dft = dft.reindex(index=np.arange(dft.index.min(), max(dft.index.max(), 2020))).fillna(0)
dft.plot(kind='bar', stacked=True, ax=ax)
ax.set_title('The test set: number of reviews of different polarities per year')

ax = axs[1][1]

dft = df_reviews.query('ds_part == "test"').groupby(['tconst', 'pos'])['pos'].count().unstack()
sns.kdeplot(dft[0], color='blue', label='negative', kernel='epa', ax=ax)
sns.kdeplot(dft[1], color='green', label='positive', kernel='epa', ax=ax)
ax.legend()
ax.set_title('The test set: distribution of different polarities per movie')

fig.tight_layout()


# The test set seems to have a smoother curve of positive reviews compared to the train set in the first 10 movies.  The test set also seems to have more fluctuations in the negative reviews compared to the train set. 

# ## Evaluation Procedure

# Composing an evaluation routine which can be used for all models in this project

# In[15]:


import sklearn.metrics as metrics

def evaluate_model(model, train_features, train_target, test_features, test_target):
    
    eval_stats = {}
    
    fig, axs = plt.subplots(1, 3, figsize=(20, 6)) 
    
    for type, features, target in (('train', train_features, train_target), ('test', test_features, test_target)):
        
        eval_stats[type] = {}
    
        pred_target = model.predict(features)
        pred_proba = model.predict_proba(features)[:, 1]
        
        # F1
        f1_thresholds = np.arange(0, 1.01, 0.05)
        f1_scores = [metrics.f1_score(target, pred_proba>=threshold) for threshold in f1_thresholds]
        
        # ROC
        fpr, tpr, roc_thresholds = metrics.roc_curve(target, pred_proba)
        roc_auc = metrics.roc_auc_score(target, pred_proba)    
        eval_stats[type]['ROC AUC'] = roc_auc

        # PRC
        precision, recall, pr_thresholds = metrics.precision_recall_curve(target, pred_proba)
        aps = metrics.average_precision_score(target, pred_proba)
        eval_stats[type]['APS'] = aps
        
        if type == 'train':
            color = 'blue'
        else:
            color = 'green'

        # F1 Score
        ax = axs[0]
        max_f1_score_idx = np.argmax(f1_scores)
        ax.plot(f1_thresholds, f1_scores, color=color, label=f'{type}, max={f1_scores[max_f1_score_idx]:.2f} @ {f1_thresholds[max_f1_score_idx]:.2f}')
        # setting crosses for some thresholds
        for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
            closest_value_idx = np.argmin(np.abs(f1_thresholds-threshold))
            marker_color = 'orange' if threshold != 0.5 else 'red'
            ax.plot(f1_thresholds[closest_value_idx], f1_scores[closest_value_idx], color=marker_color, marker='X', markersize=7)
        ax.set_xlim([-0.02, 1.02])    
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('threshold')
        ax.set_ylabel('F1')
        ax.legend(loc='lower center')
        ax.set_title(f'F1 Score') 

        # ROC
        ax = axs[1]    
        ax.plot(fpr, tpr, color=color, label=f'{type}, ROC AUC={roc_auc:.2f}')
        # setting crosses for some thresholds
        for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
            closest_value_idx = np.argmin(np.abs(roc_thresholds-threshold))
            marker_color = 'orange' if threshold != 0.5 else 'red'            
            ax.plot(fpr[closest_value_idx], tpr[closest_value_idx], color=marker_color, marker='X', markersize=7)
        ax.plot([0, 1], [0, 1], color='grey', linestyle='--')
        ax.set_xlim([-0.02, 1.02])    
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.legend(loc='lower center')        
        ax.set_title(f'ROC Curve')
        
        # PRC
        ax = axs[2]
        ax.plot(recall, precision, color=color, label=f'{type}, AP={aps:.2f}')
        # setting crosses for some thresholds
        for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
            closest_value_idx = np.argmin(np.abs(pr_thresholds-threshold))
            marker_color = 'orange' if threshold != 0.5 else 'red'
            ax.plot(recall[closest_value_idx], precision[closest_value_idx], color=marker_color, marker='X', markersize=7)
        ax.set_xlim([-0.02, 1.02])    
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('recall')
        ax.set_ylabel('precision')
        ax.legend(loc='lower center')
        ax.set_title(f'PRC')        

        eval_stats[type]['Accuracy'] = metrics.accuracy_score(target, pred_target)
        eval_stats[type]['F1'] = metrics.f1_score(target, pred_target)
    
    df_eval_stats = pd.DataFrame(eval_stats)
    df_eval_stats = df_eval_stats.round(2)
    df_eval_stats = df_eval_stats.reindex(index=('Accuracy', 'F1', 'APS', 'ROC AUC'))
    
    print(df_eval_stats)
    
    return


# ## Normalization

# We assume all models below accepts texts in lowercase and without any digits, punctuations marks etc.

# In[16]:


df_reviews['review_norm'] = df_reviews['review'].str.lower().str.replace('[^a-zA-Z]', ' ')


# ## Train / Test Split

# Luckily, the whole dataset is already divided into train/test one parts. The corresponding flag is 'ds_part'.

# In[17]:


df_reviews_train = df_reviews.query('ds_part == "train"').copy()
df_reviews_test = df_reviews.query('ds_part == "test"').copy()

train_target = df_reviews_train['pos']
test_target = df_reviews_test['pos']

print(df_reviews_train.shape)
print(df_reviews_test.shape)


# ## Working with models

# ### Model 0 - Constant

# In[18]:


from sklearn.dummy import DummyClassifier


# In[19]:


#predict most frequent class in set
dummy_clf = DummyClassifier(strategy="most_frequent")


# In[20]:


dummy_clf.fit(df_reviews_train, train_target)


# In[21]:


evaluate_model(dummy_clf, df_reviews_train, train_target, df_reviews_test, test_target)


# ### Model 1 - NLTK, TF-IDF and LR

# TF-IDF

# In[22]:


import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from nltk.corpus import stopwords as nltk_stopwords


# In[23]:


df_reviews_train['review_norm'].head()


# In[24]:


#import english stop words
stop_words = set(nltk_stopwords.words('english'))
#fit and transform train features 
vectorizor=TfidfVectorizer(stop_words=stop_words)
tf_idf_train = vectorizor.fit_transform(df_reviews_train['review_norm'])
#fit and transform test features
tf_idf_test = vectorizor.transform(df_reviews_test['review_norm'])


# In[25]:


#initialize first model 
model_1=LogisticRegression()


# In[26]:


model_1.fit(tf_idf_train, train_target)


# In[27]:


evaluate_model(model_1, tf_idf_train, train_target, tf_idf_test, test_target)


# ### Model 3 - spaCy, TF-IDF and LR

# In[28]:


import spacy

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])


# In[29]:


def text_preprocessing_3(text):
    
    doc = nlp(text)
    #tokens = [token.lemma_ for token in doc if not token.is_stop]
    tokens = [token.lemma_ for token in doc]
    
    return ' '.join(tokens)


# In[30]:


# Apply preprocessing to train and test features
train_features_processed = df_reviews_train['review_norm'].apply(text_preprocessing_3)
test_features_processed = df_reviews_test['review_norm'].apply(text_preprocessing_3)


# In[31]:


# Initialize TF-IDF Vectorizer
vectorizer_3 = TfidfVectorizer(stop_words=stop_words)


# In[32]:


# Fit and transform train features
train_features_3 = vectorizer_3.fit_transform(train_features_processed)

# Transform test features
test_features_3 = vectorizer_3.transform(test_features_processed)

# Initialize Logistic Regression model
model_3 = LogisticRegression()

# Train the model
model_3.fit(train_features_3, train_target)


# In[33]:


evaluate_model(model_3, train_features_3, train_target, test_features_3, test_target)


# ### Model 4 - spaCy, TF-IDF and LGBMClassifier

# In[34]:


from lightgbm import LGBMClassifier


# In[35]:


#define new model for LGBM classifier
model_4 = LGBMClassifier(random_state=42)


# In[36]:


#use fit and transform from model 3
model_4.fit(train_features_3, train_target)


# In[37]:


evaluate_model(model_4, train_features_3, train_target, test_features_3, test_target)


# ###  Model 9 - BERT

# In[38]:


import torch
import transformers


# In[39]:


tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
config = transformers.BertConfig.from_pretrained('bert-base-uncased')
model = transformers.BertModel.from_pretrained('bert-base-uncased')


# In[39]:


#edited 
def BERT_text_to_embeddings(texts, max_length=100, batch_size=1, force_device=None, disable_progress_bar=False): 
    # Tokenizing and encoding the text inputs
    encoded = tokenizer.batch_encode_plus(
        texts, 
        padding=True, 
        truncation=True, 
        max_length=max_length, 
        return_tensors="pt"
    )
    
    ids_list = encoded["input_ids"].tolist()
    attention_mask_list = encoded["attention_mask"].tolist()

    # Selecting the device
    if force_device is not None:
        device = torch.device(force_device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    if not disable_progress_bar:
        print(f'Using the {device} device.')
    
    # Getting embeddings in batches
    embeddings = []
    for i in tqdm(range(math.ceil(len(ids_list) / batch_size)), disable=disable_progress_bar):
        ids_batch = torch.LongTensor(ids_list[batch_size*i:batch_size*(i+1)]).to(device)
        attention_mask_batch = torch.LongTensor(attention_mask_list[batch_size*i:batch_size*(i+1)]).to(device)

        with torch.no_grad():            
            model.eval()
            batch_embeddings = model(input_ids=ids_batch, attention_mask=attention_mask_batch)   
        embeddings.append(batch_embeddings[0][:,0,:].detach().cpu().numpy())

    return np.concatenate(embeddings)


# In[ ]:


#limit df for time frame runs 
df_reviews_train['review_norm']=df_reviews_train['review_norm'].loc[:100]
# Ensure valid text input
df_reviews_train['review_norm'] = df_reviews_train['review_norm'].astype(str)
df_reviews_train = df_reviews_train.dropna(subset=['review_norm'])
df_reviews_train = df_reviews_train[df_reviews_train['review_norm'].str.strip() != ""]

# Convert to list
text_list = df_reviews_train['review_norm'].tolist()

# Run BERT embedding function
train_features_9 = BERT_text_to_embeddings(text_list, force_device=None)


# In[ ]:


print(df_reviews_train['review_norm'].shape)
print(train_features_9.shape)
print(train_target.shape)


# Attempted to run a splice of the text through BERT.  However the kernel constantly failed after 23%.  Local computer does not have a discrete video card so unable to run.  Section unable to complete even after installing drivers. 

# ## My Reviews

# In[38]:


my_reviews = pd.DataFrame([
    'I did not simply like it, not my kind of movie.',
    'Well, I was bored and felt asleep in the middle of the movie.',
    'I was really fascinated with the movie',    
    'Even the actors looked really old and disinterested, and they got paid to be in the movie. What a soulless cash grab.',
    'I didn\'t expect the reboot to be so good! Writers really cared about the source material',
    'The movie had its upsides and downsides, but I feel like overall it\'s a decent flick. I could see myself going to see it again.',
    'What a rotten attempt at a comedy. Not a single joke lands, everyone acts annoying and loud, even kids won\'t like this!',
    'Launching on Netflix was a brave move & I really appreciate being able to binge on episode after episode, of this exciting intelligent new drama.'
], columns=['review'])

my_reviews['review_norm'] = my_reviews['review'].str.lower().str.replace('[^a-zA-Z]', ' ') # <put here the same normalization logic as for the main dataset>

my_reviews


# ### Model 2

# In[39]:


from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords


# In[41]:


texts = my_reviews['review_norm']

my_reviews_pred_prob = model_1.predict_proba(vectorizor.transform(texts.apply(lambda x: text_preprocessing_3(x))))[:, 1]

for i, review in enumerate(texts.str.slice(0, 100)):
    print(f'{my_reviews_pred_prob[i]:.2f}:  {review}')


# ### Model 3

# In[42]:


texts = my_reviews['review_norm']

my_reviews_pred_prob = model_3.predict_proba(vectorizer_3.transform(texts.apply(lambda x: text_preprocessing_3(x))))[:, 1]

for i, review in enumerate(texts.str.slice(0, 100)):
    print(f'{my_reviews_pred_prob[i]:.2f}:  {review}')


# ### Model 4

# In[43]:


texts = my_reviews['review_norm']

tfidf_vectorizer_4 = vectorizer_3
my_reviews_pred_prob = model_4.predict_proba(tfidf_vectorizer_4.transform(texts.apply(lambda x: text_preprocessing_3(x))))[:, 1]

for i, review in enumerate(texts.str.slice(0, 100)):
    print(f'{my_reviews_pred_prob[i]:.2f}:  {review}')


# ### Model 9

# In[ ]:


texts = my_reviews['review_norm']

my_reviews_features_9 = BERT_text_to_embeddings(texts, disable_progress_bar=True)

my_reviews_pred_prob = model_9.predict_proba(my_reviews_features_9)[:, 1]

for i, review in enumerate(texts.str.slice(0, 100)):
    print(f'{my_reviews_pred_prob[i]:.2f}:  {review}')


# unable to run without graphics card locally. 

# ## Conclusions

# In Sprint 14 we uilized spaCy to advanced text preprocessing.  This included tasks of tokenization and lemmatization to tranform raw text data into a manageable form. TF-IDF vectorization converted the preprocessed text into numerical features to represent the importance of owrds in the reviews relative to the dataset.  Each model was then trained and evaluated on the TF-IDF features which were split into training and seting sets.  Speed in processing through machine learning models varied greately.  This is influenced by how effective the  model functions.  The accuracy across models on the test set varied.  Linear Regression and TF-IDF both achieved 0.88  LGBM Classifier was lower at .86.  BERT was unable to be run without a graphics card locally.  This project was performed without tunign hyperparameters to imrpove performance.  In the future if class imbalances were explored it woudl inprove the robustness of models. The methods here serve as a foundation for futher research into analyzing large data sets. 
# 
