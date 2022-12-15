# NLP Final Project - Music Disentangling

Codebase for submission of the final project of the Natural Language Processing course at Seoul National University, fall semester 2022.

Music and natural language processing (NLP) are two areas of research that have traditionally been studied separately. However, recent advances in machine learning have shown that it is possible to use techniques from one domain to improve performance in the other.

In this project, we investigate the use of a transformer architecture for music disentanglement. This architecture has been shown to be effective for encoding sequences in natural language processing tasks.

We observe that using a transformer architecture for encoding the music scores results in a better reconstruction and disentanglement performance.

## dMelodies data
We train and evaluate our model on the [dMelodies data set](https://github.com/ashispati/dmelodies_dataset) [1] which is a synthetic data set that comprises of 2-bar monophonic melodies where each melody is the result of a unique combination of [nine latent factors](https://github.com/ashispati/dmelodies_dataset#factors-of-variation).

## dMelodies baseline
The authors authors also published benchmark experiments that will serve as the baseline for our experiments. Their [benchmark implementations](https://github.com/ashispati/dmelodies_benchmarking) is used as the basis for our project implementation.

## Usage
To use dMelodies, simply clone the repository and install the required dependencies (`python>=3.7.15`):

```bash
$ git clone https://github.com/yhsure/nlp
$ cd nlp
$ pip install -r requirements.txt
```

Download the compressed [dMelodies_dataset.npz](https://github.com/ashispati/dmelodies_dataset/blob/master/data/dMelodies_dataset.npz) and place it in `dmelodies_dataset/data/`.
Then, run the training script to reproduce the results:

```bash
$ !PYTHONPATH=dmelodies_dataset python3 train_transformer.py
```

or alternatively, to reproduce the baseline results:

```bash
$ !PYTHONPATH=dmelodies_dataset python3 train_baseline.py
```

## References
[1] Pati, A., Gururani, S., & Lerch, A. (2020). dmelodies: A music dataset for disentanglement learning. arXiv preprint arXiv:2007.15067.
