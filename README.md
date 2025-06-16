# two-towers-overlords
MLX8 Week2 - Document retrieval to rule them all

# Initialise UV and keep updated
after performing a git pull run the following:

source ./scripts/setup.sh
source ./scripts/uv_activate.sh

# Run Bes's example CBOW for a simple w2v embedding dataset
if you don't already have the file ./checkpoints/bes-basic-cbow.cbow.pth you can run the following

source ./scripts/bes_cbow.sh

this will run the example cbow implementation that bes provided and save the results in ./checkpoints/{timestamp}.5.cbow.pth

you shouldn't need to run this as you should get this from a git pull

see ./bes-cbow/03_load_model_weights.py for examples of how to load and work with the pickle and pth files to get from word-index-embedding.



