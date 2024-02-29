# Simple Guidelines

For all feature related work, it would be great to ask yourself the following questions before submitting a PR:

1. Is your code correct? Proof of correctness and at least one Python side test. It is ok to test against well-known packages. Don't forget to add to requirements-test.txt if more packages need to be downloaded for tests. If it is a highly technical topic, can you provide some reference docs?
2. Is your code faster/as fast as SciPy, NumPy, Scikit-learn? **It is ok to be slower**. Do you have any idea why? Is it because of data structure? Some data input/copy issues that make performance hard to achieve? Again, it is ok to be slower when the functionality provides more convenience to users. It will be great to provide some notes/comments. We have the power of Rust so thereotically we can optimize many things away.
3. Are you using a lot of unwraps in your Rust code? Are these unwraps justified? Same for unsafe code.
4. If an additional dependency is needed, how much of it is really used? Will it bloat the package? What other features can we write with the additional dependency? I would discourage the addition of a dependency if we are using 1 or 2 function out of that package.
5. **Everything can be discussed**. 
6. New feature are generally welcome if it is commonly used in some field.


## Remember to run these before committing:
1. pre-commit. We use ruff.
2. cargo fmt

## How to get started? 

Take a look at the Makefile. Set up your environment first. Then take a look at the tutorial [here](https://github.com/MarcoGorelli/polars-plugins-tutorial), and grasp the basics of maturin [here](https://www.maturin.rs/tutorial).

Then find a issue/feature that you want to improve/implement!

## A word on Doc, Typo related PRs

For docs and typo fix PRs, we welcome changes that:

1. Fix actual typos and please do not open a PR for each typo.

2. Add explanations, docstrings for previously undocumented features/code.

3. Improve clarification for terms, explanations, docs, or docstrings.

4. Fix actual broken UI/style components in doc/readme.

Simple stylistic change/reformatting that doesn't register any significant change in looks, or doesn't fix any previously noted problems will not be approved.

Please understand, and thank you for your time.