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
