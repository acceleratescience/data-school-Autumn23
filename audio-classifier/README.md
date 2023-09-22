Let's move our exploratory analysis into scripts. We are using Poetry, because it makes our life incredibly easy.

We can simply navigate to a parent directory that houses our projects and run

```bash
$ poetry new audio-classifier
```

This will create the following file structure:

```bash
audio-classifier
├── pyproject.toml
├── README.md
├── audio_classifier
│   └── __init__.py
└── tests
    └── __init__.py
```

The `audio_classifier` directory is where we will build our scripts, and the `tests` directory is where we will write all of our tests.

The `pyproject.toml` file is the workhorse of the project - it manages all the metadata and the project dependencies. The neat thing about Poetry is that if you want to add a package, Poetry will update the `pyproject.toml` file to include it. We can add packages, by running

```bash
$ poetry add <package>
```

Poetry will also manage your virtual environment for you, but if you would prefer to use an existing environment, just activate it in the usual way before adding any packages, and Poetry will automatically update the environment as well!

Poetry allows us to also install and run local versions of our software, and can manage uploading to pypi.

## Creating our software package
We will create a number of files that constitute our software package, so that our overall structure looks like this:

```bash
audio-classifier
├── pyproject.toml
├── poetry.lock
├── README.md
├── audio_classifier
│   ├── __init__.py
│   ├── seeds.py
│   ├── summary_stats.py
│   ├── tune.py
│   └── utils.py
├── recordings
├── searches
└── tests
    ├── __init__.py
    ├── test_audio_functions.py
    ├── test_hyper_tuning.py
    └── test_SummaryStats.py
```

The `seeds.py` file is just the same file we used in the first and second labs. `utils.py` contains the audio loading functions. `summary_stats.py` contains only our custom stats transformer. `tune.py` contains a few functions that will load and train our data for us. It will also save the results of the experiment to a file in the `searches` folder so that we can load the model and run it on some new data if we want.

The important thing to note here is that the `SummaryStats` class is completely separate from the rest of the project. That way, people are free to use the transformation on their own data with their own parameters and use a completely different downstream model. It's also easy to modify the existing functions to collect additional statistics, or to even build new methods that transform the data in some other way.

Another thing to note is that we could create a data class from the stuff in the `utils.py` file.

Your data pipeline can be very model-dependent. For example, we could treat the spectrograms like images, and try and use a convolutional neural network on them. This would involve reshaping in some way so that they are all the same size, and then creating a custom torch dataset and dataloader. If you can get the hang of creating custom transformations for scikit-learn, creating torch datasets is only a small step - we can even keep some of the core class methods.

## Testing
How do we know that everything is working as intended? Suppose for example the `generate_mel_spectrogram` class method was the following:

```python
def generate_mel_spectrogram(self, audio : ndarray) -> ndarray:
    transform = T.MelSpectrogram(n_mels=self.n_mels, normalized=True)
    S = transform(torch.from_numpy(audio).type(torch.float))
    S = 10 * torch.log10(S)
    return S.numpy()
```

What is the problem with this?

As with the testing Lab, we again use unittest. Most of our functions and methods just have one job, so we can write relatively simple functions to test them. But how do we test reading and writing files? If we look at the `test_audio_functions.py` file, we see two methods called `setUp` and `tearDown`. We can create dummy folders and populate them with fake data so that we can test reading and writing data.

We can also see this in the `test_hyper_tuning.py` file, where we have to test writing and loading the best model found by the optimization process.

Now that everything is written, where do we go from here? It's important to write a brief overview of how to reproduce your results. Usually this is done in the README file, or in a Jupyter notebook. Ideally, you would include your entire data analysis via a notebook. So we have added a file called `audio_classification.ipynb`, that other researchers can run.

It is also possible to publish our work...