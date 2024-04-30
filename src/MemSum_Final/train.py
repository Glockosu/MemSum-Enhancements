# Authors: Alex Johannesson and Saumya Shukla
# Description: This script performs document summarization using different models and measures their performance with ROUGE scores. It supports command-line arguments for flexible execution configurations.


import os
import pickle
import time
import copy
import argparse
import os, sys

from torch.distributions import Categorical
from rouge_score import rouge_scorer
from tqdm import tqdm
from datautils import *
from model import *
from utils import *


def update_moving_average( m_ema, m, decay ):
    """
    Updates the moving average of model parameters for use in techniques like SWA (Stochastic Weight Averaging).
    
    Args:
        m_ema (torch.nn.Module): The model whose parameters are the exponential moving averages.
        m (torch.nn.Module): The model from which parameters are used to update m_ema.
        decay (float): The decay rate used to balance between the previous averages and the new parameters.
    
    This function directly modifies m_ema by blending its parameters with those from m, according to the decay rate.
    """
    with torch.no_grad():
        param_dict_m_ema =  m_ema.module.parameters()  if isinstance(  m_ema, nn.DataParallel ) else m_ema.parameters() 
        param_dict_m =  m.module.parameters()  if isinstance( m , nn.DataParallel ) else  m.parameters() 
        for param_m_ema, param_m in zip( param_dict_m_ema, param_dict_m ):
            param_m_ema.copy_( decay * param_m_ema + (1-decay) *  param_m )

def LOG( info, end="\n" ):
    """
    Logs the given information to a file specified by the global variable 'log_out_file'.
    
    Args:
        info (str): The information to log.
        end (str): The end character to use after the log message, default is newline.
    
    This function is a utility for logging runtime information or errors.
    """
    global log_out_file
    with open( log_out_file, "a" ) as f:
        f.write( info + end )

def load_corpus( fname, is_training  ):
    """
    Loads a corpus from a file, filtering out entries based on certain conditions.
    
    Args:
        fname (str): File name of the corpus to load.
        is_training (bool): Flag indicating if the loading is for training purposes.
    
    Returns:
        list: A list of data entries that meet the required conditions.
    
    This function reads a JSONL file (JSON lines) where each line is a JSON entry representing training or validation data.
    It filters out entries where the 'text' or 'summary' fields are empty. For training data, it further checks if 'indices' or 'score' are empty.
    """
    corpus = []
    with open( fname, "r" ) as f:
        for line in tqdm(f):
            data = json.loads(line)
            if len(data["text"]) == 0 or len(data["summary"]) == 0:
                continue
            if is_training:
                if len( data["indices"] ) == 0 or len( data["score"] ) == 0:
                    continue

            corpus.append( data )
    return corpus

def worker_init(worker_id):
    seed = int(time.time()) + worker_id
    np.random.seed(seed)
    torch.manual_seed(seed)

def worker_init_fn(worker_id):
        seed = int(time.time()) + worker_id
        np.random.seed(seed)
        torch.manual_seed(seed)


def main():
    # parser = argparse.ArgumentParser()

    # parser.add_argument("-training_corpus_file_name" )
    # parser.add_argument("-validation_corpus_file_name" )
    # parser.add_argument("-model_folder")
    # parser.add_argument("-log_folder")
    # parser.add_argument("-vocabulary_file_name" )
    # parser.add_argument("-pretrained_unigram_embeddings_file_name" )

    # parser.add_argument("-num_heads", type = int, default = 8 )
    # parser.add_argument("-hidden_dim", type = int, default = 1024 )
    # parser.add_argument("-N_enc_l", type = int, default = 3 )
    # parser.add_argument("-N_enc_g", type = int, default = 3 )
    # parser.add_argument("-N_dec", type = int, default = 3 )
    # parser.add_argument("-max_seq_len", type = int, default = 100 )
    # parser.add_argument("-max_doc_len", type = int, default = 50 )
    # parser.add_argument("-num_of_epochs", type = int, default = 50 )
    # parser.add_argument("-print_every", type = int, default = 100 )
    # parser.add_argument("-save_every", type = int, default = 500 )
    # parser.add_argument("-validate_every",  type = int, default= 1000 )
    # parser.add_argument("-restore_old_checkpoint", type = bool, default = True)
    # parser.add_argument("-learning_rate", type = float, default = 1e-4 )
    # parser.add_argument("-warmup_step",  type = int, default= 1000 )
    # parser.add_argument("-weight_decay", type = float, default = 1e-6)
    # parser.add_argument("-dropout_rate", type = float, default = 0.1)
    # parser.add_argument("-n_device", type = int, default = 8)
    # parser.add_argument("-batch_size_per_device", type = int, default = 16)
    # parser.add_argument("-max_extracted_sentences_per_document", type = int)
    # parser.add_argument("-moving_average_decay", type = float)
    # parser.add_argument("-p_stop_thres", type = float, default = 0.7 )
    # parser.add_argument("-apply_length_normalization", type = int, default = 1 )

    # args = parser.parse_args()


    parser = argparse.ArgumentParser()
    parser.add_argument("-config_file_path" )
    args_input = parser.parse_args()

    args = Dict2Class(json.load(open(args_input.config_file_path)))



    training_corpus_file_name  = args.training_corpus_file_name
    validation_corpus_file_name = args.validation_corpus_file_name
    model_folder = args.model_folder
    log_folder = args.log_folder
    vocabulary_file_name = args.vocabulary_file_name
    pretrained_unigram_embeddings_file_name = args.pretrained_unigram_embeddings_file_name
    num_heads = args.num_heads
    hidden_dim = args.hidden_dim
    N_enc_l = args.N_enc_l
    N_enc_g = args.N_enc_g
    N_dec = args.N_dec
    max_seq_len = args.max_seq_len
    max_doc_len = args.max_doc_len
    num_of_epochs = args.num_of_epochs
    print_every = args.print_every
    save_every = args.save_every
    validate_every = args.validate_every
    restore_old_checkpoint = args.restore_old_checkpoint
    learning_rate = args.learning_rate
    warmup_step = args.warmup_step
    weight_decay = args.weight_decay
    dropout_rate = args.dropout_rate
    n_device = args.n_device
    batch_size_per_device = args.batch_size_per_device
    max_extracted_sentences_per_document = args.max_extracted_sentences_per_document
    moving_average_decay = args.moving_average_decay
    p_stop_thres = args.p_stop_thres


    if not os.path.exists( log_folder ):
        os.makedirs(log_folder)
    if not os.path.exists( model_folder ):
        os.makedirs(model_folder)
    log_out_file = log_folder + "/train.log"

    training_corpus = load_corpus( training_corpus_file_name, True )
    validation_corpus = load_corpus( validation_corpus_file_name, False )


    print("Initializing models and datasets...")
    with open( vocabulary_file_name, "rb") as f:
        words = pickle.load(f)
    with open(pretrained_unigram_embeddings_file_name, "rb") as f:
        pretrained_embedding = pickle.load(f)
    vocab = Vocab(words)
    vocab_size, embed_dim = pretrained_embedding.shape

    train_dataset = ExtractionTrainingDataset(  training_corpus,  vocab , max_seq_len,  max_doc_len)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size_per_device * n_device, shuffle=True, num_workers=4, drop_last=True, worker_init_fn=worker_init_fn, pin_memory=True)
    total_number_of_samples = train_dataset.__len__()
    val_dataset = ExtractionValidationDataset( validation_corpus, vocab, max_seq_len, max_doc_len )
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size_per_device * n_device, shuffle=False, num_workers=4, drop_last=False, worker_init_fn=worker_init_fn, pin_memory=True)


    local_sentence_encoder = LocalSentenceEncoder( vocab_size,vocab.pad_index, embed_dim,num_heads,hidden_dim,N_enc_l, pretrained_embedding )
    global_context_encoder = GlobalContextEncoder( embed_dim, num_heads, hidden_dim, N_enc_g )
    extraction_context_decoder = ExtractionContextDecoder( embed_dim, num_heads, hidden_dim, N_dec )
    extractor = Extractor( embed_dim, num_heads )

    # restore most recent checkpoint
    if restore_old_checkpoint:
        ckpt = load_model( model_folder )
    else:
        ckpt = None

    if ckpt is not None:
        local_sentence_encoder.load_state_dict( ckpt["local_sentence_encoder"] )
        global_context_encoder.load_state_dict( ckpt["global_context_encoder"] )
        extraction_context_decoder.load_state_dict( ckpt["extraction_context_decoder"] )
        extractor.load_state_dict( ckpt["extractor"] )
        LOG("model restored!")
        print("model restored!")

    gpu_list = np.arange(n_device).tolist()
    device = torch.device(  "cuda:%d"%( gpu_list[0] ) if torch.cuda.is_available() else "cpu" )

    local_sentence_encoder_ema = copy.deepcopy( local_sentence_encoder ).to(device)
    global_context_encoder_ema = copy.deepcopy( global_context_encoder  ).to(device)
    extraction_context_decoder_ema = copy.deepcopy( extraction_context_decoder ).to(device)
    extractor_ema = copy.deepcopy( extractor ).to(device)

    local_sentence_encoder.to(device)
    global_context_encoder.to(device)
    extraction_context_decoder.to(device)
    extractor.to(device)

    if device.type == "cuda" and n_device > 1:
        local_sentence_encoder = nn.DataParallel( local_sentence_encoder, gpu_list )
        global_context_encoder = nn.DataParallel( global_context_encoder, gpu_list )
        extraction_context_decoder = nn.DataParallel( extraction_context_decoder, gpu_list )
        extractor = nn.DataParallel( extractor, gpu_list )

        local_sentence_encoder_ema = nn.DataParallel( local_sentence_encoder_ema, gpu_list )
        global_context_encoder_ema = nn.DataParallel( global_context_encoder_ema, gpu_list )
        extraction_context_decoder_ema = nn.DataParallel( extraction_context_decoder_ema, gpu_list )
        extractor_ema = nn.DataParallel( extractor_ema, gpu_list )    

        model_parameters = [ par for par in local_sentence_encoder.module.parameters() if par.requires_grad  ]  + \
                        [ par for par in global_context_encoder.module.parameters() if par.requires_grad  ]   + \
                        [ par for par in extraction_context_decoder.module.parameters() if par.requires_grad  ]  + \
                        [ par for par in extractor.module.parameters() if par.requires_grad  ]  
    else:
        model_parameters =  [ par for par in local_sentence_encoder.parameters() if par.requires_grad  ]  + \
                        [ par for par in global_context_encoder.parameters() if par.requires_grad  ]   + \
                        [ par for par in extraction_context_decoder.parameters() if par.requires_grad  ]  + \
                        [ par for par in extractor.parameters() if par.requires_grad  ]  

    optimizer = torch.optim.Adam( model_parameters , lr= learning_rate , weight_decay = weight_decay )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x:  min( (x+1)**(-0.5), (x+1)*(warmup_step**(-1.5))  ), last_epoch=-1, verbose=True)

    if ckpt is not None:
        try:
            optimizer.load_state_dict( ckpt["optimizer"] )
            scheduler.load_state_dict( ckpt["scheduler"] )
            LOG("optimizer restored!")
            print("optimizer restored!")
        except:
            pass

    current_epoch = 0
    current_batch = 0

    if ckpt is not None:
        current_batch = ckpt["current_batch"]
        current_epoch = int( current_batch * batch_size_per_device * n_device / total_number_of_samples )
        LOG("current_batch restored!")
        print("current_batch restored!")

    np.random.seed()

    rouge_cal = rouge_scorer.RougeScorer(['rouge1','rouge2', 'rougeLsum'], use_stemmer=True)


    def train_iteration(batch):
        """
        Performs a single training iteration over a given batch, handling the training process for extractive summarization.

        Args:
            batch (tuple): Contains all input and target data needed for the training step:
                        - seqs: Tensor representing tokenized sentences.
                        - doc_mask: Mask for documents to indicate padded areas.
                        - selected_y_label: Labels for training.
                        - selected_score: Scores related to the quality of the summaries.
                        - summary_seq: Sequence of summaries.
                        - valid_sen_idxs: Indices of valid sentences.

        Returns:
            float: The loss value computed for this training iteration.

        This function processes the batch data, performs forward propagation through the model,
        calculates the loss, and performs a backward pass to update model weights.
        """
        seqs, doc_mask, selected_y_label, selected_score, summary_seq, valid_sen_idxs = batch
        seqs = seqs.to(device)
        doc_mask = doc_mask.to(device)
        selected_y_label = selected_y_label.to(device)
        selected_score = selected_score.to(device)
        
        valid_sen_idxs_np = valid_sen_idxs.detach().cpu().numpy()
        valid_sen_idxs = -1*np.ones_like( valid_sen_idxs_np )
        valid_sen_idxs_np[ valid_sen_idxs_np>=doc_mask.size(1) ] = -1
        for doc_i in range( valid_sen_idxs_np.shape[0] ):
            valid_idxs = valid_sen_idxs_np[doc_i][ valid_sen_idxs_np[doc_i] != -1]
            valid_sen_idxs[doc_i, : len(valid_idxs)] = valid_idxs
        valid_sen_idxs = torch.from_numpy(valid_sen_idxs).to(device)
        
        num_documents = seqs.size(0)
        num_sentences = seqs.size(1)
        
        local_sen_embed = local_sentence_encoder( seqs.view(-1, seqs.size(2) ) , dropout_rate )
        local_sen_embed = local_sen_embed.view( -1, num_sentences, local_sen_embed.size(1) )
        global_context_embed = global_context_encoder( local_sen_embed, doc_mask , dropout_rate )
        
        doc_mask_np = doc_mask.detach().cpu().numpy()
        remaining_mask_np = np.ones_like( doc_mask_np ).astype( np.bool_ ) | doc_mask_np
        extraction_mask_np = np.zeros_like( doc_mask_np ).astype( np.bool_ ) | doc_mask_np
        
        log_action_prob_list = []
        log_stop_prob_list = []
        
        done_list = []
        extraction_context_embed = None
        
        for step in range(valid_sen_idxs.shape[1]):
            remaining_mask = torch.from_numpy( remaining_mask_np ).to(device)
            extraction_mask = torch.from_numpy( extraction_mask_np ).to(device)
            if step > 0:
                extraction_context_embed = extraction_context_decoder( local_sen_embed, remaining_mask, extraction_mask, dropout_rate )
            p, p_stop, baseline = extractor( local_sen_embed, global_context_embed, extraction_context_embed , extraction_mask , dropout_rate )
            
            p_stop = p_stop.unsqueeze(1)
            m_stop = Categorical( torch.cat( [ 1-p_stop, p_stop  ], dim =1 ) )
            
            sen_indices = valid_sen_idxs[:, step]
            done = sen_indices == -1
            if len(done_list) > 0:
                done = torch.logical_or(done_list[-1], done)
                just_stop = torch.logical_and( ~done_list[-1], done )
            else:
                just_stop = done
            
            if torch.all( done ) and not torch.any(just_stop):
                break
                
            p = p.masked_fill( extraction_mask, 1e-12 )  
            normalized_p = p / p.sum(dim=1, keepdims = True)
            ## Here the sen_indices is actually pre-sampled action
            normalized_p = normalized_p[ np.arange( num_documents ), sen_indices ]
            log_action_prob = normalized_p.masked_fill( done, 1.0 ).log()
            
            log_stop_prob = m_stop.log_prob( done.to(torch.long)  )
            log_stop_prob = log_stop_prob.masked_fill( torch.logical_xor( done, just_stop ), 0.0 )
            
            log_action_prob_list.append( log_action_prob.unsqueeze(1) )
            log_stop_prob_list.append( log_stop_prob.unsqueeze(1) )
            done_list.append(done)
            
            for doc_i in range( num_documents ):
                sen_i = sen_indices[ doc_i ].item()
                if sen_i != -1:
                    remaining_mask_np[doc_i,sen_i] = False
                    extraction_mask_np[doc_i,sen_i] = True
        
            
        log_action_prob_list = torch.cat( log_action_prob_list, dim = 1 )
        log_stop_prob_list = torch.cat( log_stop_prob_list, dim = 1 )
        log_prob_list = log_action_prob_list + log_stop_prob_list

        if args.apply_length_normalization:
            log_prob_list = log_prob_list.sum(dim=1)  / ( (log_prob_list != 0).to(torch.float32).sum(dim=1) )  
        else:
            log_prob_list = log_prob_list.sum(dim=1) 


        loss = (-log_prob_list * selected_score).mean()
        
        optimizer.zero_grad()
        loss.backward()    
        optimizer.step()

        return loss.item()


    def validation_iteration(batch):
        """
        Processes a single validation batch to evaluate the performance of the summarization model using the exponential moving average (EMA) parameters.
        
        Args:
            batch (tuple): Contains the input data needed for the validation:
                        - seqs: The sequences of tokens representing the documents.
                        - doc_mask: The mask indicating valid token positions within the documents.
                        - summary_seq: The sequences representing the summaries.
        
        Returns:
            list: A list of tuples containing ROUGE scores for each document in the batch.

        This function computes the summarization based on the model's ability to predict the importance of sentences and measures the quality using ROUGE metrics.
        """
        seqs, doc_mask, summary_seq = batch
        seqs = seqs.to(device)
        doc_mask = doc_mask.to(device)

        num_sentences = seqs.size(1)
        local_sen_embed  = local_sentence_encoder_ema( seqs.view(-1, seqs.size(2) )  )
        local_sen_embed = local_sen_embed.view( -1, num_sentences, local_sen_embed.size(1) )
        global_context_embed = global_context_encoder_ema( local_sen_embed, doc_mask  )
        
        num_documents = seqs.size(0)
        doc_mask = doc_mask.detach().cpu().numpy()
        remaining_mask_np = np.ones_like( doc_mask ).astype( np.bool ) | doc_mask
        extraction_mask_np = np.zeros_like( doc_mask ).astype( np.bool ) | doc_mask
        
        seqs = seqs.detach().cpu().numpy()
        summary_seq = summary_seq.detach().cpu().numpy()
        
        done_list = []
        extraction_context_embed = None
        
        for step in range(max_extracted_sentences_per_document):
            remaining_mask = torch.from_numpy( remaining_mask_np ).to(device)
            extraction_mask = torch.from_numpy( extraction_mask_np ).to(device)
            if step > 0:
                extraction_context_embed = extraction_context_decoder_ema( local_sen_embed, remaining_mask, extraction_mask )
            p, p_stop, baseline = extractor_ema( local_sen_embed, global_context_embed, extraction_context_embed , extraction_mask  )
            
            p = p.masked_fill( extraction_mask, 1e-12 )  
            normalized_p = p / (p.sum(dim=1, keepdims = True))

            stop_action = p_stop > p_stop_thres
            
            done = stop_action | torch.all(extraction_mask, dim = 1) 
            if len(done_list) > 0:
                done = torch.logical_or(done_list[-1], done)
            if torch.all( done ):
                break
                
            sen_indices = torch.argmax(normalized_p, dim =1)
            done_list.append(done)
            
            for doc_i in range( num_documents ):
                if not done[doc_i]:
                    sen_i = sen_indices[ doc_i ].item()
                    remaining_mask_np[doc_i,sen_i] = False
                    extraction_mask_np[doc_i,sen_i] = True
                    
        scores = []
        for doc_i in range(seqs.shape[0]):
            ref = "\n".join( [ vocab.seq2sent( seq ) for seq in summary_seq[doc_i] ]  ).strip()
            extracted_sen_indices = np.argwhere( remaining_mask_np[doc_i] == False )[:,0]
            hyp = "\n".join(  [ vocab.seq2sent( seq ) for seq in seqs[doc_i][extracted_sen_indices]] ).strip()
        
            score = rouge_cal.score( hyp, ref )
            scores.append( (score["rouge1"].fmeasure,score["rouge2"].fmeasure,score["rougeLsum"].fmeasure) )

        return scores

    optimizer = torch.optim.Adam(model_parameters, lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=False)

    initial_lr = next(iter(optimizer.param_groups))['lr']
    current_learning_rate = initial_lr

    for epoch in range(args.num_of_epochs):
        print(f"Starting Epoch {epoch+1}/{args.num_of_epochs}")
        running_loss = 0.0
        for count, batch in enumerate(train_data_loader):
            loss = train_iteration(batch)
            running_loss += loss
            current_batch += 1

            if count % args.print_every == 0:
                current_learning_rate = [group['lr'] for group in optimizer.param_groups][0]
                print(f"[Epoch {epoch+1} Batch {count+1}] Loss: {loss:.3f}, Avg Loss: {running_loss/(count+1):.3f}, LR: {current_learning_rate}")

            if current_batch % args.save_every == 0:
                model_save_path = os.path.join(args.model_folder, f"model_epoch_{epoch+1}_batch_{current_batch}.pt")
                save_model({
                    'local_sentence_encoder': local_sentence_encoder_ema,
                    'global_context_encoder': global_context_encoder_ema,
                    'extraction_context_decoder': extraction_context_decoder_ema,
                    'extractor': extractor_ema,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'batch': current_batch
                }, model_save_path, max_to_keep=5)
                print(f"Model saved at {model_save_path}")

            if current_batch % args.validate_every == 0:
                print("Starting validation...")
                validation_scores = []
                for val_batch in val_data_loader:
                    scores = validation_iteration(val_batch)
                    validation_scores.extend(scores)
                validation_metric = np.mean([score['metric_to_improve'] for score in validation_scores])
                scheduler.step(validation_metric)
                print(f"Validation results updated scheduler with metric {validation_metric:.4f}")

        print(f"Epoch {epoch+1} completed. Average Loss: {running_loss / len(train_data_loader):.3f}")

    print("Training complete.")


if __name__ == '__main__':
    main()