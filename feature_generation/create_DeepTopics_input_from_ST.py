import sys
import numpy as np
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import anndata as ad

def main():
    # Check if two file paths are provided as command-line arguments
    if len(sys.argv) != 4:
        print("Usage: python create_DeepTopics_input_from_ST.py <input_path> <vocab_path> <output_filename>")
        print("  <input_path>       Path to ST data directory ending with '/outs/' or to a '.h5ad' file")
        print("  <vocab_path>       Path to the vocabulary file (e.g., vocab.npy)")
        print("  <output_filename>  Name for the output file (e.g., VAE_input.txt)")
        return

    # Extract the file paths from command-line arguments
    adata_filepath = sys.argv[1]
    vocab_filepath = sys.argv[2]
    output_file = sys.argv[3]
    
    create_input_documents(adata_filepath, vocab_filepath, output_file)
    print("Created the inputs file at: " + output_file)
    
def create_input_documents(adata_filepath, vocab_filepath, output_file):
    
    # Read in the adata file, can be a .h5ad or a path to a 10x Visium /outs/ subdirectory
    adata = None
    if adata_filepath.endswith(".h5ad"):
        adata = sc.read_h5ad(adata_filepath)
    elif adata_filepath.endswith("/outs/"):
        adata = sc.read_visium(adata_filepath)
    else:
        raise ValueError("Unsupported file path. Please provide a .h5ad file or a path ending with '/outs/'.")
        
    adata.var_names_make_unique()
    print("Read in the adata file from: " + adata_filepath)
    
    # Use adata.X for the raw counts instead of reading in the matrix.mtx.gz file directly
    # Depending on the Visium format, certain genes might be removed, so best to be safe and go for
    # the filtered matrix instead of the raw matrix
    raw_counts = adata.X
    
    # Read in the vocab file
    vocab = np.load(vocab_filepath, allow_pickle=True)
    print("Loaded the vocab from: " + vocab_filepath)

    # FOR NOW, THE GENE NAMES ARE THE KEYS AND THE NUMBERS ARE THE VALUES. For testing
    # purposes. Can change this later as needed.
    vocab_map = {k: v for k, v in vocab.item().items()}
    
    # No need to transpose since adata.X is spot x gene
    spots_by_genes = pd.DataFrame.sparse.from_spmatrix(raw_counts)
    
    # Remove all columns (GENES) that have fewer than 10 nonzero values
    lower_bound = (1 - (10 / len(spots_by_genes)))
    spots_by_genes = spots_by_genes.loc[:, (spots_by_genes==0).mean() < lower_bound]
    
    # Also remove all columns that have nonzero values in > 95% of rows
    upper_bound = 0.95 * len(spots_by_genes)
    spots_by_genes = spots_by_genes.loc[:, (spots_by_genes!=0).mean() < upper_bound]
    
    # Save the spot indices and the indices of the FILTERED GENES from above. There should be
    # no spots filtered out. Each of these indices corresponds to a gene in adata.var, 
    # and these indices have been filtered by our greater than 10 spots, fewer than 95% 
    # of spots criteria.
    spot_indices = list(spots_by_genes.index)
    gene_indices = list(spots_by_genes.columns)
    
    # Turn the df into a numpy array so that it's WAY faster to iterate over.
    df_values = spots_by_genes.to_numpy()
    
    f = open(output_file, "a")
    
    for spot in range(len(df_values)):
        print("Spot", spot)
        # create the string for the gene list
        genes_in_spot = ""

        # Iterate over each column (gene) in every spot 
        for gene in range(len(df_values[0])):
            # Reset this variable just in case
            gene_occurrences = ""
            
            # Find the number of times this gene appears in this spot
            num_times_gene_appears = df_values[spot][gene]

            # Get the gene index that the current gene represents, from gene_indices
            gene_index_in_adata = gene_indices[gene]

            # Do a reverse lookup just to be safe. Get the gene name from adata, compare that
            # gene name's value in the vocab dict to gene_index_in_vocab
            # The gene index in the vocab should be that in the adata + 1
            gene_name_in_adata = adata.var.index[gene_index_in_adata]
            gene_index_in_vocab = vocab_map[gene_name_in_adata]
#             assert(gene_index_in_vocab == gene_index_in_adata + 1)

            # Repeat that gene as many times as it occurs in that spot
            # ADD A SPACE AT THE END.
            # If the number of occurrences is 0, just skip it or there will be a TON of extra spaces.
            if num_times_gene_appears > 0:
                gene_occurrences = ' '.join([str(gene_index_in_vocab)] * int(num_times_gene_appears)) + " "
            # Add the genes to the list for that spot
            genes_in_spot += gene_occurrences
        # write it to the file
        f.write(genes_in_spot + "\n")
    # End of for loop
    
    f.close()

if __name__ == "__main__":
    main()