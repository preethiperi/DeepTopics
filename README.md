# Introduction 
Source code of DeepTopics paper on Dirichlet VAEs for learning cellular expression programs from spatial transcriptomics data.

https://www.biorxiv.org/content/10.1101/2025.01.08.631928v1


# Installation
1.	Installation process for the machine learning model
Please create a conda environment as shown below OR using the yaml file: tfp.yaml

Using the supplied yaml file:

	conda env create --name tfp --file=tfp.yaml

If you have most of the dependencies already installed, the following simpler setup will suffice:

	conda create -n tfp python=3.7
 	conda activate tfp
	conda install tensorflow-gpu
	conda install tensorflow-probability


Note: In some versions of tensorflow / tensorflow-probability, you might get a "KL divergence is negative" error during training. We have not yet figured out why this appears.

2. Dependencies for the feature generation

The feature generation code uses scanpy, which can be installed using the following instructions: https://scanpy.readthedocs.io/en/stable/installation.html

Numpy and pandas are also needed.

## TRAINING

	python src/run_modified.py --model_dir models/RA379A/ --train_path data/RA379A_input_with_counts.txt --eval_path data/RA379A_input_with_counts.txt --test_path data/RA379A_input_with_counts.txt --num_topics 14 --prior_initial_value 10 --mode train --vocab_path data/RA379A_vocab.npy

Parameters that are most sensitive and best ones to tweak:

batch_size  (currently set at 32)<br>
num_topics  (size of the hidden bottleneck layer)<br>
prior_initial_value<br>
prior_burn_in_steps<br>

Modifying the number of layers and their width in the Encoder.

## TEST (or getting TOPIC POSTERIORS)

If you want to use a previously saved model to do inference on new data, use the code in "test" mode as follows:

	python src/run_modified.py --model_dir models/RA379A/ --train_path data/RA379A_input_with_counts.txt --eval_path data/RA379A_input_with_counts.txt --test_path data/RA379A_input_with_counts.txt --num_topics 14 --prior_initial_value 10 --mode test --vocab_path data/RA379A_vocab.npy --preds_file RA379A_infer_alpha10_K14

Output: a matrix of size N x K, where N = number of examples in the input file, K = number of topics / latent dimensions.
--preds_file is for the name of the .npy file that will contain the document-topic matrix.

## GENE DISTRIBUTIONS (DECODER PARAMETERS that encode the TOPIC distributions over words)

	python src/run_modified.py --model_dir models/RA379A/ --num_topics 10 --prior_initial_value 10 --mode beta --vocab_path data/RA379A_vocab.npy

# FILE FORMATS

## Data file format:
A list of feature-ids separated by spaces. The training / test files are formatted as lists of features where if a feature has count k, then it appears k times in the list. Each line of the file is one example.
If your dataset is a 10x Visium ST dataset, the code used in feature_generation/create_DeepTopics_input_from_ST.py can be used to create the input directly. If you have a .h5ad file, this same script can be used. 

If you want to change this input format, please look at sparse_matrix_dataset (or let me know and I can help with it). See below for a file with two input examples (documents). The feature ids should be in an increasing order. Also see attached sample file (data/RA379A_input_with_counts.txt).

112 113 113 113 122 134 144 144 144 144 159 178<br>
115 115 189 194 194 202 202 202

## Vocabulary format:
Please see the sample vocabulary file (data/RA379A_vocab.npy file) for how to format the <feature-id>  <feature-name>  mapping.  It is in a dictionary format. For example, below are the top few lines of the vocabulary for the k-mer model, which was converted into the RA379A_vocab.npy file. So, if you load the dictionary, d['EMPTY']=0  and d['MIR1302-2HG']=1 and so on. 
    
Please keep the first dictionary entry a dummy feature like 'EMPTY' and assign it to the index 0. Obviously, none of the examples will contain this feature :)! This is due to how the indexing is done after loading the vocabulary (i.e. the useful features should have indices >=1).

EMPTY<br>
AAAAAAAA<br>
AAAAAAAC<br>
AAAAAAAG<br>
AAAAAAAT<br>
AAAAAACA<br>
AAAAAACC<br>
AAAAAACG<br>
AAAAAACT<br>
AAAAAAGA<br>
AAAAAAGC<br>
AAAAAAGG<br>

# OUTPUTS

To monitor what the model is learning, you can look at the periodic outputs. The frequency of outputs is controlled by the parameter viz_steps in the code. It is currently set to 20000, but feel free to set it to 1000 or so in the initial runs till you understand what's going on.

Here's what it looks like for genes and ST spots. Only the top few are printed. Again this can be controlled by looking at the method get_topics_strings.

Finished reading  3660  documents....
elbo
-19958.9

kl
18.809708

loss
19956.559

perplexity
1455.0908

reconstruction
-19940.096

topics
b'index=5 alpha=0.82 IGHG4 IGKC IGHG3 IGLC2 IGLC1 IGHM FTL B2M FTH1 CST3'
b'index=0 alpha=0.82 IGKC IGHG4 IGHG3 IGHA1 IGHG1 IGLC2 IGHM IGLC1 JCHAIN SSR4'
b'index=6 alpha=0.69 FTH1 FTL CST3 CD74 B2M TMSB10 ITM2B EEF1A1 RPS12 HLA-DRB1'
b'index=2 alpha=0.69 FN1 FTL HLA-DRA B2M CTSB CD74 FTH1 HLA-B RNASE1 IFI30'
b'index=7 alpha=0.65 FTL FN1 FTH1 B2M HLA-B HLA-DRA CTSB PLA2G2A CST3 TMSB4X'
b'index=9 alpha=0.64 FN1 MMP3 PRG4 CLU CRTAC1 MMP1 MT2A TIMP3 PLA2G2A TIMP1'
b'index=12 alpha=0.63 COL1A2 COL3A1 FTL COL1A1 SPARC PLA2G2A COL6A2 FSTL1 CST3 DCN'
b'index=8 alpha=0.59 B2M MT-CO3 HLA-DRA FTH1 MT-CO1 MT-ATP6 CD74 PLA2G2A MT-CO2 HLA-B'
b'index=10 alpha=0.58 FTL FTH1 RNASE1 CD74 CTSB B2M PLTP CCL18 PLA2G2A CST3'
b'index=11 alpha=0.54 IGFBP7 B2M TMSB10 EEF1A1 VIM HLA-B A2M ACTB IFITM3 SPARC'

global_step
260000


# Contribute
TODO: Explain how other users and developers can contribute to make your code better. 
