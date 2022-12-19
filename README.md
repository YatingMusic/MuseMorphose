# MuseMorphose

This repository contains the official implementation of the following paper:  

* Shih-Lun Wu, Yi-Hsuan Yang  
**_MuseMorphose_: Full-Song and Fine-Grained Piano Music Style Transfer with One Transformer VAE**  
accepted to _IEEE/ACM Trans. Audio, Speech, & Language Processing (TASLP)_, Dec 2022 [<a href="https://arxiv.org/abs/2105.04090" target="_blank">arXiv</a>] [<a href="https://slseanwu.github.io/site-musemorphose/" target="_blank">demo website</a>]

## Prerequisites
* Python >= 3.6
* Install dependencies
```bash
pip3 install -r requirements.txt
```
* GPU with >6GB RAM (optional, but recommended)

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

## Generation
```bash
python3 generate.py [config file] [ckpt path] [output dir] [num pieces] [num samples per piece]
```
* e.g.
```bash
python3 generate.py config/default.yaml musemorphose_pretrained_weights.pt generations/ 10 5
```

This script will randomly draw the specified # of pieces from the test set.  
For each sample of a piece, the _rhythmic intensity_ and _polyphonicity_ will be shifted entirely and randomly by \[-3, 3\] classes for the model to generate style-transferred music.  
You may modify `random_shift_attr_cls()` in `generate.py` or write your own function to set the attributes.

## Customized Generation (To Be Added)
We welcome the community's suggestions and contributions for an interface on which users may
 * upload their own MIDIs, and 
 * set their desired bar-level attributes easily

## Citation BibTex
If you find this work helpful and use our code in your research, please kindly cite our paper:
```
@article{wu2023musemorphose,
    title={{MuseMorphose}: Full-Song and Fine-Grained Piano Music Style Transfer with One {Transformer VAE}},
    author={Shih-Lun Wu and Yi-Hsuan Yang},
    year={2023},
    journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
}
```
