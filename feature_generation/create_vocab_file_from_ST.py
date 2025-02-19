import sys
import numpy as np
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import anndata as ad

def main():
    # Check if two file paths are provided as command-line arguments
    if len(sys.argv) != 3:
        print("Usage: python create_vocab_file.py path/to/data/outs/ vocab_output_file_name.npy")
        
        print("Usage: python create_vocab_file.py <input_path> <output_filename>")
        print("  <input_path>       Path to ST data directory ending with '/outs/' or to a '.h5ad' file")
        print("  <output_filename>  Name for the output file (e.g., vocab_output_file_name.npy)")
        return

    # Extract the file paths from command-line arguments
    adata_filepath = sys.argv[1]
    vocab_filepath = sys.argv[2]
    
    # Use filepath1 and filepath2 in other parts of the script
    print(f"adata filepath: {adata_filepath}")
    print(f"vocab output filename: {vocab_filepath}")
    
    create_input_vocab(adata_filepath, vocab_filepath)
    print("Vocab file present in: " + vocab_filepath)
    
def create_input_vocab(adata_filepath, vocab_filepath):
    
    # Read in the adata file
    adata = None
    if adata_filepath.endswith(".h5ad"):
        adata = sc.read_h5ad(adata_filepath)
    elif adata_filepath.endswith("/outs/"):
        adata = sc.read_visium(adata_filepath)
    else:
        raise ValueError("Unsupported file path. Please provide a .h5ad file or a path ending with '/outs/'.")
        
    adata.var_names_make_unique()
    print("Read in adata from: " + adata_filepath)
    
    gene_list = adata.var.index
    
    vocabulary = {}
    vocabulary['EMPTY'] = 0
    count = 1
    for gene in gene_list:
        if gene in vocabulary.keys():
            print(gene)
        vocabulary[gene] = count
        count += 1
    # The length should be the number of genes in the adata + 1 for the EMPTY one
    assert(len(vocabulary) == len(adata.var) + 1)
    np.save(vocab_filepath, vocabulary)
    
if __name__ == "__main__":
    main()
