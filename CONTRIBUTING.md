# How to contribute to rust-bert?

Code contributions to the library are very welcome, especially for areas of focus given below. 
However, please note that direct contributions to the `rust-bert` library are not the only way you can help the project.
Building applications and supporting the applications built on top of the library, sharing
the word about them in the Rust and NLP community or simply a star go a long way in helping this project develop.

## Code contributions areas of focus

Rust is a very efficient, safe and fast language when implemented in the right way.
The field of transformers in NLP and especially their implementation in Rust is a rather new development.
Considering this, the general execution speed of pipelines built using `rust-bert` is a priority for the project (along with the correctness of the results).

Contributions are therefore welcome on the following areas:
- Improvement of execution performance at a module, model or pipeline level
- Reduction of memory footprint for the models

For other areas of contributions, opening an issue to discuss the feature addition would be very welcome. 
As this started out as a personal project, it would be great to coordinate as this may be something that is already in the implementation pipeline.

## General contribution guidelines

- Please try running the suite of integration tests locally before submitting a pull request. Most features are tested automatically in the Travis CI - but due to the large size of some models some tests cannot be run in the virtual machines provided.
- The code should be formatted using `cargo +nightly fmt` to format both the code and the documentation
- As much as possible, please try to adhere to the coding style of the crate. I am open to discuss non-idiomatic code.
- When providing a performance improvement, please provide benchmarks to illustrate the performance gain, if possible with and without GPU support.
- Please try to ensure that the documentation always reflects the actual state of the code.

## Did you find a bug?

Thank you - identifying and sharing bugs is one of the best way to improve the overall quality of the crate!
When submitting the bug as an issue, it would be very helpful if you could share the full stack trace of the error, and the input provided to reproduce the error.
Since several models are non deterministic (generation pipelines are using random sampling to generate text), it would be very useful to manually turn off sampling (`do_sample: false` in the relevant configuration) and reproduce the error with a given input.


This guide was inspired by the original [Transformers guide to contributing](https://github.com/huggingface/transformers/blob/master/CONTRIBUTING.md)