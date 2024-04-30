# Authors: Alex Johannesson and Saumya Shukla
# Description: This script performs document summarization using different models and measures their performance with ROUGE scores. It supports command-line arguments for flexible execution configurations.

import json
from my_summarizers import ExtractiveSummarizer_MemSum_Final
import argparse

parser = argparse.ArgumentParser(description="Summarization and scoring with ROUGE")
parser.add_argument("-model_type", required=True)
parser.add_argument("-summarizer_model_path", required=True)
parser.add_argument("-vocabulary_path", required=True)
parser.add_argument("-corpus_path", required=True)
parser.add_argument("-gpu", type=int, default=None)
parser.add_argument("-max_extracted_sentences_per_document", type=int, default=5)
parser.add_argument("-p_stop_thres", type=float, default=0.7)
parser.add_argument("-output_file", required=True)
parser.add_argument("-max_count", type=int, default=None)
parser.add_argument("-use_fast_rouge", type=int, default=1)
parser.add_argument("-system_dir")
parser.add_argument("-model_dir")
parser.add_argument("-N_enc_l", type=int, default=2)
parser.add_argument("-N_enc_g", type=int, default=2)
parser.add_argument("-N_dec", type=int, default=3)
parser.add_argument("-embed_dim", type=int, default=200)
parser.add_argument("-ngram_blocking", default="False")
parser.add_argument("-ngram", type=int, default=4)
parser.add_argument("-max_doc_len", type=int, default=500)
parser.add_argument("-max_seq_len", type=int, default=100)

args = parser.parse_args()


def setup_summarizer(model_type, summarizer_model_path, vocabulary_path, gpu, N_enc_l, N_enc_g, N_dec, embed_dim, max_doc_len, max_seq_len):
    """
    Initializes and returns a summarizer based on the model type and provided configurations.
    
    Args:
        model_type (str): Type of model to use ('MemSum_Final', 'MemSum_wo_history', or 'MemSum_with_stop_sentence').
        summarizer_model_path (str): Path to the model's weight file.
        vocabulary_path (str): Path to the vocabulary file.
        gpu (int): GPU index to use. If None or a non-integer, CPU is used.
        N_enc_l (int): Number of local encoder layers.
        N_enc_g (int): Number of global encoder layers.
        N_dec (int): Number of decoder layers.
        embed_dim (int): Dimension of embeddings.
        max_doc_len (int): Maximum document length (number of sentences).
        max_seq_len (int): Maximum sequence length (number of tokens per sentence).

    Returns:
        An instance of a summarizer based on the specified model type.
    """
    if model_type == "MemSum_Final":
        from my_summarizers import ExtractiveSummarizer_MemSum_Final
        summarizer = ExtractiveSummarizer_MemSum_Final(
            summarizer_model_path,
            vocabulary_path,
            gpu=gpu,
            N_enc_l=N_enc_l,
            N_enc_g=N_enc_g,
            N_dec=N_dec,
            embed_dim=embed_dim,
            max_doc_len=max_doc_len,
            max_seq_len=max_seq_len
        )
    elif model_type == "MemSum_wo_history":
        from my_summarizers import ExtractiveSummarizer_MemSum_wo_history
        summarizer = ExtractiveSummarizer_MemSum_wo_history(
            summarizer_model_path,
            vocabulary_path,
            gpu=gpu,
            N_enc_l=N_enc_l,
            N_enc_g=N_enc_g,
            N_dec=N_dec,
            embed_dim=embed_dim,
            max_doc_len=max_doc_len,
            max_seq_len=max_seq_len
        )
    elif model_type == "MemSum_with_stop_sentence":
        from my_summarizers import ExtractiveSummarizer_MemSum_with_stop_sentence
        summarizer = ExtractiveSummarizer_MemSum_with_stop_sentence(
            summarizer_model_path,
            vocabulary_path,
            gpu=gpu,
            N_enc_l=N_enc_l,
            N_enc_g=N_enc_g,
            N_dec=N_dec,
            embed_dim=embed_dim,
            max_doc_len=max_doc_len,
            max_seq_len=max_seq_len
        )
    else:
        raise ValueError("Unsupported model type: {}".format(model_type))

    return summarizer

# Function to process and print summaries
def process_documents(corpus_path, summarizer, max_count=None):
    extracted_summaries = []
    with open(corpus_path, 'r') as f:
        for i, line in enumerate(f):
            if max_count is not None and i >= max_count:
                break
            data = json.loads(line)
            # Assuming `extractor` returns a list of lists (each sentence is a list of words)
            summary = summarizer.extract(data['text'])  # Modify according to actual method call
            # Flatten the list of lists to a list of sentences (strings)
            flattened_summary = [' '.join(sentence) for sentence in summary[0]]
            extracted_summary = ' '.join(flattened_summary)
            extracted_summaries.append(extracted_summary)
    return extracted_summaries


if __name__ == "__main__":
    # Setting up the summarizer with all required parameters from args
    summarizer = setup_summarizer(
        args.model_type,
        args.summarizer_model_path,
        args.vocabulary_path,
        args.gpu,
        args.N_enc_l,
        args.N_enc_g,
        args.N_dec,
        args.embed_dim,
        args.max_doc_len,
        args.max_seq_len
    )

    # Processing the documents to generate summaries
    extracted_summaries = process_documents(args.corpus_path, summarizer, args.max_count)

    # Optionally write the summaries to the output file
    with open(args.output_file, 'w') as f:
        for summary in extracted_summaries:
            f.write(summary + '\n')
