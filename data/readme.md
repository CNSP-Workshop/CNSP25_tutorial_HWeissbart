## Dataset

Besides simulated data we will be using MEG data recorded at the Donders Institute (REF). Those data were recorded while participant were simply asked to listen to speech audiobook from which we will extract relevant features (both acoustic and linguistic).

### MEG data

The MEG data supplied consist of one sample participant. The data have been preprocessed, low pass filtered and downsampled at 200Hz. ICA was run beforehand in order to identify eye components which have been removed.
The stimulus channel in the data contains the id of the story which is started at a given sample. The information about which audio corresponds to which id is described in the stimulus section.

### Stimulus

The `stim` folder contains data relative to the stimulus, in particular it consists of several time series corresponding to the following features:

* envelope
* word surprisal
* word entropy
* syntactic depth
* syntactic "close"
