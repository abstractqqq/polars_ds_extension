# Polars Extension for General Data Science Use

**Currently in Alpha. Feel free to submit feature requests in the issues section of the repo.**

The goal for this package is to provide data scientists/analysts/engineers/quants more tools to manipulate, transform, and make sense of data, without the need to leave DataFrame land (aka Wonderland).

Performance is a focus, but sometimes it's impossible to beat NumPy/SciPy performance for a single operation on a single array. There can be many reasons: Interop cost, null checks, lack of support for complex number (e.g We have to do multiple copies in the FFT implementation), or we haven't found the most optimized way to write some algorithm, etc.

However, there are greater benefits for staying in DataFrame land:

1. Works with Polars expression engine and more expressions can be executed in parallel. E.g. running fft for 1 series may be slower than NumPy, but if you are running some fft, together with some other non-trivial operations, the story changes completely.
2. Works in group_by context. E.g. run multiple linear regressions in parallel in a group_by context.
3. Staying in DataFrame land typically keeps code cleaner and less confusing.

```Python
df.group_by("dummy").agg(
    pl.col("y").num_ext.lstsq(pl.col("a"), pl.col("b"), add_bias = True)
)
```

The package right now contains two extensions:

## Numeric Extension

### Existing Features

1. GCD, LCM for integers
2. harmonic mean, geometric mean, other common, simple metrics used in industry.
3. Common loss functions, e.g. L1, L2, L infinity, huber loss, MAPE, SMAPE, wMAPE, etc.
4. Common mini-models, lstsq, condition entropy. 
5. Discrete Fourier Transform, returning the real and complex part of the new series.


## String Extension

### Existing Features

1. Levenshtein distance, Hamming distance, str Jaccard similarity
2. Simple Tokenize
3. Stemming (Right now only Snowball stemmer for English)

### Todo list

1. Longest common subsequence as string distance metric
2. Vectorizers (Count + TFIDF)?
3. Similarity version of the distances, and more variations and parameters.

## Other Extensions ?

E.g. stats_ext, dist_ext (L^p distance for vectors (scalar version is implemented) etc.) etc.

Simple unsupervised clusters can also be done. It is simply a matter of willingness and market demand.


# Disclaimer

Rust Snowball Stemmer is taken from Tsoding's Seroost project (MIT). See [here](https://github.com/tsoding/seroost)