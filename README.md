# Polars Extension for General Data Science Use

Currently not published. I will publish to PyPI once the top items in the todo list are finished.

Feel free to submit feature requests in the issues section of the repo.


## Numeric Extension

### Existing Features

1. GCD, LCM for integers
2. harmonic mean, geometric mean...
3. Other simple metrics used in industry...

### Todo list

1. Simple Regression (solved by the method of least square. E.g. `pl.col(target).num_ext.lstsq([pl.col(x1), pl.col(x2)], bias=True)`)
2. Fourier Transform (using Rust FFT, Tentative)

## String Extension

### Existing Features

1. Levenshtein distance, Hamming distance, str Jaccard similarity
2. Simple Tokenize
3. Stemming (Right now only Snowball stemmer for English)

### Todo list

1. Longest common subsequence as string distance metric
2. Vectorizers (Count + TFIDF)
3. Similarity version of the distances, and more variations and parameters.

## Other Extension ?

E.g. stats_ext, list_ext, dist_ext (L^p distance, etc.) etc.


# Disclaimer

Rust Snowball Stemmer is taken from Tsoding's Seroost project (MIT). See [here](https://github.com/tsoding/seroost)