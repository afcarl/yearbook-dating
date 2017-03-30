# Dating Historical Yearbook Photos
This is the reference implementation of the dating experiments from the updated version of https://arxiv.org/abs/1511.02575:

    A Century of Portraits: A Visual Historical Record of American High School Yearbooks
    Shiry Ginosar, Kate Rakelly, Sarah M. Sachs, Brian Yin, Crystal Lee, Philipp Krähenbühl and Alexei A. Efros
    arXiv: Coming soon!

**Contents**

- `src`: net, solver, and data layer specifications to train Yearbook dating models
    * `yearbook_layers.py`: a Python data layer that loads images and feeds them into a Caffe network
    * `solver.py`: set solver parameters in Python
    * `net.py`: NetSpec defines the network architecture
    * `solve.py`: training script 
- `notebooks`: visualize training curves, evaluate trained networks and reproduce paper figures
- `data`: data splits
- `caffe`: the Caffe framework, included as a [git submodule](https://git-scm.com/book/en/v2/Git-Tools-Submodules) pointing to a compatible version

## License

This project is licensed for open non-commercial distribution under the UC Regents license; see [LICENSE](./LICENSE).
Its dependencies, such as Caffe, are subject to their own respective licenses.

## Requirements & Installation

Caffe, Python, and Jupyter are necessary for all of the experiments.
Any installation or general Caffe inquiries should be directed to the [caffe-users](mailto:caffe-users@googlegroups.com) mailing list.

1. After cloning this repository, do `git submodule init` and `git submodule update` inside the Caffe directory to clone the caffe submodule.
2. Install Caffe dependencies. See the [installation guide](http://caffe.berkeleyvision.org/installation.html) and try [Caffe through Docker](https://githIub.com/BVLC/caffe/tree/master/docker) (recommended).
*Make sure to configure pycaffe, the Caffe Python interface, too.*
3. Follow Caffe installation instructions to install required Python packages. [Miniconda](https://conda.io/miniconda.html) comes with many of the requirements (recommended).
4. Install [Jupyter](http://jupyter.org/), the interface for viewing, executing, and altering the notebooks.
5. Configure your `PYTHONPATH` as indicated by the included `.envrc` so that this project dir and pycaffe are included.
6. Download the Yearbook dataset from the [project page](http://people.eecs.berkeley.edu/~shiry/projects/yearbooks/yearbooks.html). Place the women dataset in `data/faces/women/images` and the men dataset in `data/faces/men/images`.


If you don't want to train your own model, the Python notebooks can be used for inference with our [trained dating models](https://gist.github.com/katerakelly/842f948d568d7f1f0044)
To train a dating model, download the [model weights](http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel) for [VGG-16](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) pre-trained on [ILSVRC](http://www.image-net.org/challenges/LSVRC/) and place them in a directory called `models`. 
Configure training settings in `train_example.sh`, then call this script to train a dating network.

