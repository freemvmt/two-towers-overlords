mkdir -p ./data
mkdir -p ./checkpoints

python bes-cbow/00_train_tkn.py
python bes-cbow/01_train_w2v.py