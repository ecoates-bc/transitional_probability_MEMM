# Supervised word segmentation with transitional probability

This Maximum-Entropy Markov Model uses a feature space of 
word-internal transitional probabilities to predict word 
boundaries in unsegmented text.

## Usage
> $ python3 tp_run.py {--s, --d} text

Where `text` is an unsegmented string, and `--s` or `--d` are
decoding methods: just logistic regression, or using the Viterbi
algorithm.

## Background
In phonology, transitional probability (TP) is a measure of how 
likely two symbols occur together in a corpus. It can further
be divided into forward TP, or the likelihood that a symbol B 
follows a symbol A; and backward TP, which is the likelihood 
of the symbol A preceding B.

In statistical language learning, TP is a method for discovering 
word boundaries in unsegmented speech (Pelucci et al., 2009). 
Anghelescu (2016) investigated the accuracy of TP as a
word segmentation metric for Sesotho, but did not find it to be
a useful predictor.

I predicted that TP would be a useful word segmentation metric
when used in a supervised learning model. I chose to use a
Maximum-Entropy Markov Model (McCallum et al, 2000) because of
its combination of logistic regression, hand-selected features,
and sequential decoding. 

## Methods
I used the three Jane Austen texts in NLTK's Project Gutenberg 
corpus as a dataset. Each row of the source dataset contains
an unsegmented line of text, and the corresponding target line
is the segmented text.

### Feature transformations
The model's feature function takes a character, and calculates
its forward and backward TP given the previous character or next
character. In addition, it calculates TP with the previous and 
next bigrams when the given character is the first element in 
its bigram.

I used scikit-learn's KBinsDiscretizer to transform the continuous
TP into binary features. This was necessary, because of the MEMM
architecture. It also reflected the experimental choices in
Pelucci et al. (2009), which discretized the TP values into
High-TP and Low-TP when creating sample speech for infants.

### Model construction
I used scikit-learn's LogisticRegression classifier to start 
with, and implemented two decoding methods. The first simply used
the regression model's predictions to segment words, and the
second used the Viterbi algorithm with the model's softmax 
predictions.

### Computation
I tried my best to reduce the computational complexity of
generating features. The calculation of TP across the dataset and
training the discretizers is relatively quick, and the current
bottleneck is generating a feature matrix from a line of text.
To get around this, I pre-computed the feature transforms for the
training and validation sets.

## Results and Discussion
Test accuracy was computed using two metrics: line-level recall,
or the percentage of lines that were segmented perfectly; and
average edit distance. The former was a dismal 0.02%, and the
latter was 4.78, or an average of 4-5 incorrect segmentations
per line.

I found that both of the decoding methods resulted in identical
segmentations (but if there was an error in my implementation
of the Viterbi algorithm, please do tell me).

I conclude that TP is not useful as the sole feature space for
supervised, maximum-entropy word segmentation. However, it does
provide some useful information for the segmentation task. I
predict that a sequential neural network using character-level
embeddings leverages the character distribution of a dataset
similarly to TP, but is more accurate.

I also predict that TP would be more useful for unsupervised word
segmentation— perhaps as part of a K-means clustering model; and
that TP might be more useful in subword segmentation, i.e.
predicting morpheme or syllable boundaries. Despite this model's
low accuracy, I still believe that TP has potential as a
segmentation metric.

## Python files
* `create_dataset.py`: My code for generating the corpus from
NLTK's Project Gutenberg files.
  
* `feature_creator.py`: Code for generating features from text 
and calculating TP.
  
* `tp_memm.py`: The MEMM implementation.

* `tp_run.py`: Code for trying out the model on the command line.

* `evaluate_model.py`: Calculating training accuracy and edit distance.

## References
>Anghelescu, Andrei. (2016). Distinctive transitional probabilities across words and morphemes in Sesotho. University of British Columbia Working Papers in Linguistics. 44.1-17.

>Pelucci, Bruna, Hay, Jessica F., and Saffran, Jenny R. (2009). Learning in Reverse: Eight-month-old infants track backward transitional probabilities. Cognition, 2.

>McCallum, Andrew, Freitag, Dayne, and Pereira, Fernando. (2000). Maximum Entropy Markov Models for Information Extraction and Segmentation. Proceedings of the Seventeenth International Conference on Machine Learning. 591-598.