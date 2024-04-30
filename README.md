```markdown
# MemSum Setup and Usage Guide

## Step 1: Set Up Environment
Create an Anaconda environment named 'memsum' and activate it:
```
conda create -n memsum python=3.7
conda activate memsum
```

Install Jupyter Notebook and IPython within the environment:
```
conda install ipython
conda install -c anaconda jupyter
```

## Step 2: Install Dependencies and Pretrained Model
Install required Python packages:
```
pip install -r requirements.txt
```

Install PyTorch with the appropriate CUDA toolkit version (replace `<CUDA_VERSION>` with your actual CUDA version):
```
conda install pytorch cudatoolkit=<CUDA_VERSION> -c pytorch -y
```

Download and set up pretrained GloVe word embeddings for the MemSum model:
```
python download_and_load_word_embedding.py
```

## Step 3: Test Trained Model
Evaluate the full MemSum model's performance on the Pubmed's test set using the following command. The output includes ROUGE 1/2/L scores:
```
python my_test.py -model_type MemSum_Final \
-summarizer_model_path model/MemSum_Final/pubmed_full/200dim/best/model.pt \
-vocabulary_path model/glove/vocabulary_200dim.pkl \
-corpus_path data/pubmed_full/test_PUBMED.jsonl -gpu 0 \
-max_extracted_sentences_per_document 7 -p_stop_thres 0.6 \
-output_file results/MemSum_Final/pubmed_full/200dim/test_results.txt \
-max_doc_len 500 -max_seq_len 100
```

## Step 4: Use Pretrained Summarizer Module
Load the full MemSum model:
```python
from my_summarizers import ExtractiveSummarizer_MemSum_Final
memsum_model = ExtractiveSummarizer_MemSum_Final(
    "model/MemSum_Final/pubmed_full/200dim/best/model.pt",
    "model/glove/vocabulary_200dim.pkl",
    gpu=0, embed_dim=200, max_doc_len=500, max_seq_len=100
)
```

Prepare a document for summarization (The document should be a list of sentences):
```python
import json
# Load the document
database = [json.loads(line) for line in open("data/pubmed_full/test_PUBMED.jsonl").readlines()]
pos = 6  # Example position in the database
document = database[pos]["text"]
gold_summary = database[pos]["summary"]
```

Generate a summary using MemSum:
```python
extracted_summary = memsum_model.extract(
    [document], p_stop_thres=0.6, max_extracted_sentences_per_document=7, 
    return_sentence_position=False
)[0]
```

Evaluate the summary with ROUGE scores:
```python
from rouge_score import rouge_scorer
rouge_cal = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)
print(rouge_cal.score("\n".join(gold_summary), "\n".join(extracted_summary)))
```

## Step 5: Train Model
Train the full MemSum model on the PubMed dataset by running the train.py script in the appropriate directory:
```
cd src/MemSum_Final/; python train.py -config_file_path config/pubmed_full/200dim/run0/training.config
```
```
