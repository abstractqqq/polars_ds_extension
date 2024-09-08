# Going off from Scikit-learn Standard

I use Scikit-learn daily and I think I have my fair share of critiques on it. There are many other things to say about Scikit-learn, such as it is slow, hard to get true parallelism working, not Dataframe-centric, the functions/Transformers it provides are not expressive enough, not easy to serialize, or jsonify,  etc. All these issues are addressed or partially addressed by PDS (polars_ds). However, my biggest complaint is the Pipeline API that Scikit-learn provides. 

![auto_complete](examples/auto_complete.png)

PDS offers an all-in-one class for the most common ML pipeline transforms. You can also extend and customize it, which you can find in examples/pipeline.ipynb. You can even completely describe a pipeline with transforms using dictionaries and lists if you want.  You don't have to remember the names of these transformers becasue your linter will auto complete for you. The pipeline construction is designed to be dataframe centric and as expressive as possible: any data scientist who spends a few seconds looking at it will know what it will do. Everything in PDS just comes with the package and requires no dependency, not even on SciPy.

I also don't like the idea of putting model in a Pipeline, which Scikit-learn allows, but is not allowed by PDS. Doing so complicates the expected output of the pipeline. In practice, people track the raw features and the transformed features and then also the model scores. The PDS pipeline will only take in raw features and get transformed features out. Models are separate entities and should be kept separate. Thus, the API is clean, and the jobs of data transform and model prediction are isolated in their own realm. Now let's say you only want to tune hyperparamters for the model, not the pipeline, then you don't have to touch the data pipeline at all. There is no such clean separation in Scikit-learn, or at least the package doesn't encourage it.

As it stands now, I have no intention to make the API Scikit-learn compliant. I think there is a lack of spirit in exploring new APIs that actually faciliates  developer experience and I do not want to confine myself to the pre-defined world of "works with Scikit-learn".

That said, common names for certain functions should stay the same. I will still be using .fit(...) and .transform(...) as in the Scikit-learn sense.