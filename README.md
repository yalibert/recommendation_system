This repo contains the final project of CAS NLP.
It is a recommendation system, which was built in the following system:

1- for a given author name (typically the user's name), a database is created by collecting the abstracts of all papers cited once by the user

2- each paper is given a score that depends on the number of authors in the author list (where the user is), and the position of the user in this author list (these are the papers that CITE the one refers at point 1)

3- a training sample was built taking astroph.EP abstracts and organising in triplets
    - one anchor
    - one positive sample which is a paper cited by the anchor, belonging to astroph.EP
    - one negative sample which is a paper NOT cited by the anchor, belonging to astroph.EP, and at least one year older than the anchor
    
4- sciBERT (allenAI) was trained such that the embedding of the anchor and positive samples are as close as possible, and embeddings of anchor and negative samples are as different as possible. In this way, the embeddings of two abstracts should be similar (cosine similarity) if one abstract cites the other, and dissimilar otherwise.

5- this fine-tuned model is then used to produce the embeddings of all papers of point 1. This is done in 'generate database' page of the app

6- then in rank_abstract page, new astroph.EP abstracts (from the X last days) are queried and embedded with the fine-tune model

7- finally, the cosine similarity between these new abstracts embeddings and the embeddings of the databases is computed. All the cosine similarities (new versus database) are computed, and the average value, weighted by the weights mentioned in point 2, is computed

8- this final value is used to rank new abstracts from 'most similar to the database' to 'less similar to the database'. In principle, the rank should correspond to papers that the user would cite, and therefore interesting for them.
The abstract are listed on the webpage by decreasing order of similarity, with the title, the abstract, and a clicable link to the pdf of the paper.
    
