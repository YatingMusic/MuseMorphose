# MuseMorphose

## Preprocessing
```bash
# download REMI-pop-1.7K dataset
wget -O remi_dataset.tar.gz https://zenodo.org/record/4782721/files/remi_dataset.tar.gz?download=1
tar xzvf remi_dataset.tar.gz
rm remi_dataset.tar.gz

# compute attributes classes
python3 attributes.py
```

## Training
```bash
python3 train.py [config file]
```
* e.g.
```bash
python3 train.py config/default.yaml
```
* Or, you may download the pretrained weights straight away
```bash
wget -O musemorphose_pretrained_weights.pt https://zenodo.org/record/5119525/files/musemorphose_pretrained_weights.pt?download=1
```

## Inference
```bash
python3 generate.py [config file] [ckpt path] [output dir] [num pieces] [num samples per piece]
```
* e.g.
```bash
python3 generate.py config/default.yaml musemorphose_pretrained_weights.pt generations/ 10 5
```

This script will randomly draw the specified # of pieces from the test set.  
For each sample of a piece, the _rhythmic intensity_ and _polyphonicity_ will be shifted entirely and randomly by \[-3, 3\] classes for the model to generate style-transferred music.  
You may modify `random_shift_attr_cls()` in `generate.py` or write your own function to set the attribute classes.
