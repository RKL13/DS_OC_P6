# Automatically classify e-commerce products

### Overview: Clustering of products with nontabular data (text and images).

Even if the supervised (classification) version of the mission performs very well, it is asked to implement a clustering algorithm to classify the products. The ARI score is therefore chosen as a metric as the ground truth of the labels is available.
 
The text data is treated with spacy library (tokenization, case, stop words, POS, OOV, lemmatization, etc.). Feature are extracted from the cleaned text with Bag of Words techniques (Count, Tf-Idf) and Word Embeddings techniques (Word2Vec, Glove, FastText, BERT, USE). 

The feature extraction of images is implemented with Bag of Visual Words techniques (SIFT, ORB) as well as Convolutional Neural Networks architectures from which the last layers were replaced with a flattened one (VGG-16, ResNet50V2, EfficientNetB0). 

A Kmeans with seven components (as there are seven labels) are implemented as well as different dimensionality reduction techniques (PCA, PCA + T-SNE, Latent Dirichet Allocation, SVD, NMF). The spacy's native word embedding with a PCA + T-SNE dimensionality reduction performed the best over both text and image cases (and over a voting classifier that concatenates the two best extraction techniques spacy's embedding and EfficientNetB0).
