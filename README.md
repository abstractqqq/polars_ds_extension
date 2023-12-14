# Polars Extension for General Data Science Use

A Polars Plugin aiming to simplify common numerical/string data analysis procedures. This means that the most basic data science, stats, NLP related tasks can be done natively inside a dataframe. 

Its goal is not to replace SciPy, or NumPy, but rather it tries reduce dependency for common workflows and simple analysis, and tries to reduce Python side code and UDFs.

See examples [here](./examples/basics.ipynb).

**Currently in Alpha. Feel free to submit feature requests in the issues section of the repo.**

## Plans?

1. Some more string similarity like: https://www.postgresql.org/docs/9.1/pgtrgm.html

## Other Extensions ?

More stats, clustering, etc. It is simply a matter of willingness and market demand.

## Future Plans

I am open to make this package a Python frontend for other machine learning processes/models with Rust packages at the backend. There are some very interesting packages to incorporate, such as k-medoids. But I do want to stick with Faer as a Rust linear algebra backend and I do want to keep it simple for now.

# Credits

1. Rust Snowball Stemmer is taken from Tsoding's Seroost project (MIT). See [here](https://github.com/tsoding/seroost)
2. Some statistics functions are taken from Statrs (MIT). See [here](https://github.com/statrs-dev/statrs/tree/master)

# Other related Projects

1. Take a look at our friendly neighbor [functime](https://github.com/TracecatHQ/functime)
2. My other project [dsds](https://github.com/abstractqqq/dsds). This is currently paused because I am developing polars-ds, but some modules in DSDS, such as the diagonsis one, is quite stable.
3. String similarity metrics is soooo fast and easy to use because of [RapidFuzz](https://github.com/maxbachmann/rapidfuzz-rs)