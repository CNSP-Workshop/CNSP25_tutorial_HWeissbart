## Dataset

> ⚠️ All data are directly downloaded via the notebook tutorial, so you do not _have to_ do it yourself.
> However, if you were to worry about your internet connection being slow, you can go and download it in advance.

Besides simulated data we will be using MEG data recorded at the Donders Institute (REF). Those data were recorded while participant were simply asked to listen to speech audiobook from which we will extract relevant features (both acoustic and linguistic).

### MEG data

The MEG data supplied consist of one sample participant (sub-019 in original dataset). The data have been preprocessed, low pass filtered and downsampled at 200Hz. ICA was run beforehand in order to identify eye components which have been removed.
The stimulus channel in the data contains the id of the story which is started at a given sample. The information about which audio corresponds to which id is described in the stimulus section.

The data can be directly downloaded from [OSF](https://osf.io/gsvbd/files/). Again, a reminder that you do not _have to_ download anything as this will be covered **within the tutorial notebook**. If you want to do it yourself (for instance so that you have it prepared beforehand in case of slow internet connections), make sure you place all files under `data/sub-001/`.

> Note that more subjects are available directly from the paper's [Figshare dataset](https://doi.org/10.6084/m9.figshare.24236512.v1)

### Stimulus

The `stim` folder contains data relative to the stimulus, in particular it consists of several time series corresponding to the following features:

* envelope
* word surprisal
* word entropy
* syntactic depth
* syntactic "close"

### Reference

The data presented here, together with some of the method of the tutorial are inspired from results and analysis from 
> The Structure and Statistics of Language jointly shape Cross-frequency Neural Dynamics during Spoken Language Comprehension;
> Hugo Weissbart, Andrea E. Martin;
> Nature Comms. (2024); [![DOI](https://img.shields.io/badge/DOI-10.1038/s41467--024--53128--1-blue.svg)](https://doi.org/10.1038/s41467-024-53128-1)
