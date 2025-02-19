Commands to generate features

The feature generation scripts can take either a filepath to a 10x Visium datasets (the /outs/ directory), or a .h5ad file

# Input: 10x Visium spot data

Command: Usage: python create_DeepTopics_input_from_ST.py <input_path> <vocab_path> <output_filename>

Example:
python create_DeepTopics_input_from_ST.py ../data/RA379A/outs/ ../data/RA379A_vocab.npy RA379A_input.txt


# Input: Gene set

Command: python create_vocab_file.py <input_path> vocab_output_file_name.npy
