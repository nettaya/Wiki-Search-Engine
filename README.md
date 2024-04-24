# IR-Project---Search-Engine-on-Wikipedia
This project is our final project in the "Information Retrieval" course. We have built a search engine for all the information in the English Wikipedia. We built indexes that contained all the information and ran them in a GCP work environment. Some of the attachments files in Git are for building the indexes. Some of the attachments files for running index searches.

Search frontend

Search: Search over Wikipedia. The documents relevancy is ranked by the length of the query. Long query is ranked by BM25 retrieval function and short query is ranked by anchor and title search.

# Indexes: 
Index_body_posting_locs,
Index_body_DL,
Index_title,
Index_body,
Index_anchor

# Three parts:
In the index creation as well as in the query, tokenize the function and preprocess it for the query.
Load all indices from buckets.
An implementation of the search method based on the length of the query in the backend file.
