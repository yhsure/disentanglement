# NLP Final Project - Music Disentangling

Music and natural language processing (NLP) are two areas of research that have traditionally been studied separately. However, recent advances in machine learning have shown that it is possible to use techniques from one domain to improve performance in the other.

In this project, we investigate the use of a transformer architecture for music disentanglement. This architecture has been shown to be effective for encoding sequences in natural language processing tasks.

We observe that using a transformer architecture for encoding the music scores results in a better reconstruction and disentanglement performance.

## Data set

We train and evaluate our model on the [dMelodies data set](https://github.com/ashispati/dmelodies_dataset) which is a synthetic data set that comprises of 2-bar monophonic melodies where each melody is the result of a unique combination of [nine latent factors](https://github.com/ashispati/dmelodies_dataset#factors-of-variation).

## Baseline

The authors authors also published benchmark experiments that will serve as the baseline for our experiments. Their [benchmark implementations](https://github.com/ashispati/dmelodies_benchmarking) is used as the basis for our project implementation.
