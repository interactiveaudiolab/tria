<div style="text-align: center;">
  <img src="assets/img/tria_logo.svg" alt="TRIA logo" style="width: 50%;">
</div>

# The Rhythm In Anything: Audio-Prompted Drums Generation with Masked Language Modeling 

This repository contains training and inference code for the TRIA "anything-to-drums" system proposed in the paper **The Rhythm In Anything: Audio-Prompted Drums Generation with Masked Language Modeling**.

![](https://static.arxiv.org/static/browse/0.3.4/images/icons/favicon-16x16.png) [arXiv Paper: The Rhythm In Anything: Audio-Prompted Drums Generation with Masked Language Modeling
](https://arxiv.org/abs/2509.15625) <br>
📈 [Demo Site](https://therhythminanything.github.io)<br>
⚙ [Model Weights](https://github.com/interactiveaudiolab/tria/releases/download/0.0.1/weights.pth)

## Installation

Clone the repo:

```
git clone https://github.com/interactiveaudiolab/tria
cd tria
pip install -r requirements.txt
```


Grant permissions:
```
chmod -R u+x scripts
```

## Inference

Launch the [Gradio](https://www.gradio.app/) interface:

```
python app.py
```

<span style="color:red">More models and configurations coming soon!</span>


## Training

### Download Datasets

__Base Configuration (`26G`)__: the TRIA models discussed in our [paper](https://arxiv.org/abs/2509.15625) were trained on a subset of the [MusDB-HQ](https://sigsep.github.io/datasets/musdb.html) dataset, totalling roughly 8 hours of drum data. To download this data, run:

```
./scripts/download/download_data.sh <DATA_DIR>
python scripts/setup/create_manifests.py
```

where `<DATA_DIR>` is the directory in which you want to store data. At this point, you should be ready to [train](#single-gpu-training) TRIA from scratch!

__Additional Augmentations (`88G`)__: to enable additional noise and reverb augmentations on source audio for robust rhythm feature extraction, you can download room impulse response and background noise data:

```
./scripts/download/download_extra_augs.sh
python scripts/setup/create_extra_aug_manifests.py
```

__Additional High-Quality Drum Data (`190G`)__: to obtain additional high-quality isolated drum data, you can download the [MoisesDB](https://music.ai/research/#datasets) dataset via the Moises.ai website; you will be prompted to fill out a form to access the dataset. Once you have downloaded the dataset and extracted it to your `<DATA_DIR>`, run:

```
python scripts/setup/consolidate_moises.py
python scripts/setup/create_moises_manifests.py
```

__Additional Drum Loops (`11G`)__: to obtain additional drum loops and improve the timbral diversity of training data, you can download the [FreeSound Loop Dataset](https://arxiv.org/abs/2008.11507). Filtering to remove short (<4s) and non-drum recordings results in a dataset of roughly 1800 loops spanning 7 hours. To download and prepare the dataset, run:

```
./scripts/download/download_loops.py.sh
python scripts/setup/create_loops_manifests.py
```

__Large-Scale Low-Quality Drum Data__: another way to scale drum data is to run a pre-trained source separation model on a large corpus of musical mixtures such as the [MTG-Jamendo](https://mtg.github.io/mtg-jamendo-dataset/) dataset (`152G`). In our experiments, training on [HDEMUCS](https://docs.pytorch.org/audio/stable/tutorials/hybrid_demucs_tutorial.html)-separated drum stems resulted in low-quality generations due to the prevalence of separation artifacts. However, it may still be possible to leverage such noisy data data by using it to train only "early" generation steps (e.g. coarse RVQ codebooks for masked language modeling).


### Configuration


We provide configuration files for the five TRIA variants evaluated in our paper in the `conf/` directory, with `small_2b_musdb.yml` corresponding to the "main" TRIA system.

We use [`argbind`](https://github.com/pseeth/argbind) for training configuration. Once you've downloaded data and created manifests, training/validation datasets can be modified by providing paths in the relevant portions of the config file:

```
train/StemDataset.sources:
  - manifests/moisesdb/train.csv

val/StemDataset.sources:
  - manifests/moisesdb/val.csv
```

as can noise and impulse response sources for data augmentation:

```
train/build_transform.names: [
  ...
  "Reverb",
  "BackgroundNoise",
]

...

Reverb.drr: [uniform, 0.0, 30.0]
Reverb.sources:
  - manifests/rir_real/train.csv

BackgroundNoise.snr: [uniform, 10.0, 30.0]
BackgroundNoise.sources:
  - manifests/noise_room/train.csv
```



### Single-GPU Training

One you have downloaded your chosen datasets, you can train on a single GPU with:

```
export CUDA_VISIBLE_DEVICES=0
python scripts/train.py --args.load conf/small_2b_musdb.yml
```

### Multi-GPU Training

You can train on multiple GPUs (e.g. 2) with:

```
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node gpu scripts/train.py --args.load conf/small_2b_musdb.yml
```


## Licenses

The training and inference code in this repository are licensed under the [MIT License](LICENSE). The pretrained model weights are obtained from data licensed under [Creative Commons Attribution-NonCommercial-ShareAlike (CC BY-NC-SA)](https://creativecommons.org/licenses/by-nc-sa/4.0/) and are therefore released under the same license.


## Model Versions

This repository is an open-source reimplementation of the TRIA system described in [our paper](https://arxiv.org/abs/2509.15625), and as a result models trained using this repository may differ from those presented in the paper and supplementary materials. During the re-implementation process, we found that minor differences in random seeding, data augmentation, and dataset split can affect model performance in the small-data regime explored in the paper. Anecdotally, we find that __scaling training data reliably improves performance, with models exhibiting much stronger timbre adherence and reduced sensitivity to inference parameter configurations__. 

Therefore:
* If you want a TRIA model trained on licensed, publicly available data (i.e. MusDB, MoisesDB, and FreeSound Loops), we recommend using the [default configuration](conf/small_musdb_moises_fsl_2b.yml)
* If you want to explore the settings discussed in the TRIA paper, we provide [matching configurations](conf/exp/)
* If you have access to large-scale high-quality licensed drum data, we recommend re-training TRIA on that data. 


## 📝 To-Do:
* Add configs/weights for ablations and offload weights from repo
* Both the `Reverb` and `BackgroundNoise` transforms are slow due to inefficient file reads and salient excerpting
* Add support for additional discrete and continuous tokenizers; currently, only [DAC](https://github.com/descriptinc/descript-audio-codec) is supported, as the code and weights are MIT-licensed
* Switch rhythm features from perceptual to RMS loudness normalization to match original TRIA
* Allow training on variable feature sparsity / quantization, akin to [Sketch2Sound](https://arxiv.org/abs/2412.08550), to allow for inference-time control over conditioning granularity
* Additional learning rate schedules (currently using DAC exponential decay schedule)

## Citation

```
@inproceedings{tria2025,
    author = {Patrick O'Reilly and Julia Barnett and Hugo Flores Garcia and Annie Chu and Nathan Pruyne and Prem Seetharaman and Bryan Pardo},
    title = {The Rhythm In Anything: Audio-Prompted Drums Generation with Masked Language Modeling},
    booktitle = {International Society for Music Information Retrieval Conference (ISMIR)},
    year = {2025},
}
```
