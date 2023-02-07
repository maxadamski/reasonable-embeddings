<div align="center"><img src="/local/out/img/demo.png" width="100%"></div>

**[Reason-able Embeddings: Learning Concept Embeddings with a Transferable Deep Neural Reasoner](http://www.semantic-web-journal.net/content/reason-able-embeddings-learning-concept-embeddings-transferable-deep-neural-reasoner)**
<br/>
[Max Adamski](https://maxadamski.com), [Jędrzej Potoniec](https://scholar.google.pl/citations?user=Z-hEFe0AAAAJ&hl=pl)
<br/>
[[Paper]](http://www.semantic-web-journal.net/content/reason-able-embeddings-learning-concept-embeddings-transferable-deep-neural-reasoner)
[[BibTex]](#link-to-bibtex)

```
Space reserved for bibtex entry
```

## Overview

<div align="center"><img src="/local/out/img/reasoner.png" width="80%" alt="Reasoner architecture overview"></div>

Reason-able concept embeddings are learned in a data-driven way, by simply asking queries about logical consequences for a given knowledge base.

Our method uses a logical consequence (entailment) classifier based on a recursive deep neural network, so after learning embeddings, one can both construct embeddings of arbitrarily complex concepts, and use the resulting classifier to perform fast approximate reasoning.

A significant part of our reasoner (the reasoner head) is transferable across knowledge bases in the ALC description logic, including real-world knowledge bases.

Hopefully, reason-able concept embeddings and deep neural reasoners will allow for greater use of symbolic knowledge in deep learning architectures.

## Repository structure

Notable files in this repository are described below. In parentheses, we indicate if a figure, table, or experiment was based on a given file.

```
reasonable-embeddings/
├─ factplusplus/     # Git submodule <https://bitbucket.org/dtsarkov/factplusplus/>
├─ scripts/
│  └─ build          # Builds FaCT++ and our Cython extensions
├─ src/
│  ├─ simplefact/
│  │  ├─ factpp.pyx  # Python bindings to the FaCT++ reasoner
│  │  ├─ owlfun.pyx  # OWL functional style syntax parser
│  │  └─ syntax.py   # Constants used to construct concept expressions
│  ├─ exp1.ipynb  # Training the relaxed reasoner on the synthetic data set (Fig. 4, Table 4)
│  ├─ exp2.ipynb  # Training the restricted reasoner on the synthetic data set (Fig. 3, Table 2, 3)
│  ├─ exp3.ipynb  # Training the restricted reasoner on the pizza taxonomy
│  ├─ exp4.ipynb  # Learning and visualizing embeddings of arbitrary concepts in the pizza ontology (Fig. 5, Table 6)
│  ├─ exp5.ipynb  # Evaluating reasoner head transfer from sythetic data to real-world ontologies (Table 7, 10)
│  ├─ extra.ipynb # Finding axioms that are hard for the reasoner (Table 9)
│  ├─ tests.ipynb # Results of statistical tests (Table 5, 8)
│  ├─ reasoner.py # Implementation of the relaxed reasoner and the training procedure
│  └─ generate.py # Axiom generator, synthetic data set generator, and data set loading and saving procedures
└─ local/out/
   ├─ dataset/
   │  ├─ sub-100.json     # The synthetic data set used (Experiment 1, 2)
   │  └─ *.ofn            # Pizza ontology and ontologies from the CQ2SPARQLOWL data set,
   │                      # converted to OWL functional style syntax (Experiment 3, 4)
   └─ exp/
      ├─ 20220715T194328/ # Artifacts of exp1.ipynb
      ├─ 20220715T194304/ # Artifacts of exp2.ipynb
      ├─ 20220719T213232/ # Artifacts of exp5.ipynb
      └─ table3-k...e../  # Artifacts of exp2.ipynb, with a set embedding size,
                          # and number of neurons in the reasoner head
```

## How to run

You will need to install Python 3.9, Jupyter and all required dependencies. The easiest way to do this is with Conda. If you don't have conda, install a distribution (for example [miniconda](https://docs.conda.io/en/latest/miniconda.html)). Note that the code was only tested on Linux.

If you want to use conda, then create and enter a new environment with the following commands. If not, skip to the next step.

```
conda create --name reasoner python=3.9 jupyter
conda activate reasoner
```

Install the requirements.

```
pip install -r requirements.txt
```

With the requirements in place, run the build script for the FaCT++ reasoner and our Python bindings. Of course, you will need to clone the factplusplus submodule repository into the `factplusplus/` directory, if you haven't done so already (only required if you cloned our repository from Github and didn't use the `--recurse-submodules` flag).

```
chmod +x scripts/build
scripts/build
```

Lastly, start Jupyter.

```
jupyter notebook
```

To re-run an experiment open a Jupyter notebook in a web browser and click `Restart & Run All`. Results of experiments are shown in notebooks. In addition to that, some experiments save results (including model weights) on disk in a directory based on the time stamp (for example, `local/out/exp/20220715T194304` stores the result of exp2.ipynb).

