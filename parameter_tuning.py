# Authors: Alex Johannesson and Saumya Shukla
# Description: This script performs document summarization using different models and measures their performance with ROUGE scores. It supports command-line arguments for flexible execution configurations.

import os
import itertools

# Parameters for grid search: defining various configurations to test
p_stop_thres_values = [0.5, 0.6, 0.7]  # Threshold probabilities for stopping the summarization
max_sentences_values = [5, 7, 10]  # Maximum number of sentences to extract per document
N_enc_l_values = [1, 2, 3]  # Number of layers in the local encoder
N_enc_g_values = [1, 2, 3]  # Number of layers in the global encoder
embed_dim_values = [150, 200, 250]  # Dimensions for word embeddings
ngram_values = [3, 4, 5]  # Sizes of n-grams for blocking
ngram_blocking_values = [True, False]  # Boolean flag to enable or disable n-gram blocking

# Generate all possible combinations of parameters for grid search
parameter_combinations = list(itertools.product(
    p_stop_thres_values, max_sentences_values, N_enc_l_values, N_enc_g_values,
    embed_dim_values, ngram_values, ngram_blocking_values
))

# Iterate over each combination of parameters
for combination in parameter_combinations:
    # Unpack the parameters from each combination
    p_stop_thres, max_sentences, N_enc_l, N_enc_g, embed_dim, ngram, ngram_blocking = combination
    
    # Construct the command to run the model with specific parameters
    command = f"python my_test.py -model_type MemSum_Final -summarizer_model_path model/MemSum_Final/pubmed_full/200dim/best/model.pt -vocabulary_path model/glove/vocabulary_200dim.pkl -corpus_path data/pubmed_full/test_PUBMED.jsonl -gpu 0 -max_extracted_sentences_per_document {max_sentences} -p_stop_thres {p_stop_thres} -output_file results/MemSum_Final/pubmed_full/200dim/test_results_{p_stop_thres}_{max_sentences}_{N_enc_l}_{N_enc_g}_{embed_dim}_{ngram}_{ngram_blocking}.txt -max_doc_len 500 -max_seq_len 100 -N_enc_l {N_enc_l} -N_enc_g {N_enc_g} -embed_dim {embed_dim} -ngram {ngram} -ngram_blocking {str(ngram_blocking).lower()}"
    
    # Execute the command: this will run the test with the current set of parameters
    os.system(command)

