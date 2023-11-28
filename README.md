# Polars Extension for General Data Science Use

A Polars Plugin aiming to simplify common numerical/string data analysis procedures. This means that the most basic data science, stats, NLP related tasks can be done natively inside a dataframe. Its goal is not to replace SciPy, or NumPy, but rather it tries reduce dependency for common workflows and simple analysis, and tries to reduce Python side code and UDFs.

**Currently in Alpha. Feel free to submit feature requests in the issues section of the repo.**

This package will also be a "lower level" backend for another package of mine called dsds. See [here](https://github.com/abstractqqq/dsds).

Performance is a focus, but sometimes it's impossible to beat NumPy/SciPy performance for a single operation on a single array. There can be many reasons: Interop cost (sometimes copies needed), null checks, lack of support for complex number (e.g We have to do multiple copies in the FFT implementation), or we haven't found the most optimized way to write some algorithm, etc.

However, there are greater benefits for staying in DataFrame land:

1. Works with Polars expression engine and more expressions can be executed in parallel. E.g. running fft for 1 series may be slower than NumPy, but if you are running some fft, together with some other non-trivial operations, the story changes completely.
2. Works in group_by context. E.g. run multiple linear regressions in parallel in a group_by context.
3. Staying in DataFrame land typically keeps code cleaner and less confusing.

See examples [here](./examples/basics.ipynb).

## Plans?

1. Some more string similarity like: https://www.postgresql.org/docs/9.1/pgtrgm.html

## Other Extensions ?

More stats, clustering, etc. It is simply a matter of willingness and market demand.

## Future Plans

I am open to make this package a Python frontend for other machine learning processes/models with Rust packages at the backend. There are some very interesting packages to incorporate, such as k-medoids. But I do want to stick with Faer as a Rust linear algebra backend and I do want to keep it simple for now.

Right now most str similarity/dist is dependent on the strsim crate, which is no longer maintained and has some very old code. The current plan is to keep it for now and maybe replace it with higher performance code later (if there is the need to do so). 

# Credits

1. Rust Snowball Stemmer is taken from Tsoding's Seroost project (MIT). See [here](https://github.com/tsoding/seroost)
2. Some statistics functions are taken from Statrs (MIT). See [here](https://github.com/statrs-dev/statrs/tree/master)

# Other related Projects

1. Take a look at our friendly neighbor [functime](https://github.com/TracecatHQ/functime)
2. My other project [dsds](https://github.com/abstractqqq/dsds). This is currently paused because I am developing polars-ds, but some modules in DSDS, such as the diagonsis one, is quite stable.