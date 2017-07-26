# Paper abstract / Intro: 

Accurate annotation of protein functions is important for a profound understanding of molecular biology. A large number of proteins remain uncharacterized because of the sparsity of available supporting information. For a large set of uncharacterized proteins, the only type of information available is their amino acid sequence. In this paper, we propose DeepSeq -- a deep learning architecture -- that utilizes only the protein sequence information to predict its associated functions. The prediction process does not require handcrafted features; rather, the architecture automatically extracts representations from the input sequence data. Results of our experiments with DeepSeq indicate significant improvements in terms of prediction accuracy when compared with other sequence-based methods. Our deep learning model achieves an overall validation accuracy of 86.72%, with an F1 score of 71.13%. Moreover, using the automatically learned features and without any changes to DeepSeq, we successfully solved a different problem i.e. protein function localization, with no human intervention. Finally, we discuss how this same architecture can be used to solve even more complicated problems such as prediction of 2D and 3D structure as well as protein-protein interactions. 

![Deep Learning for Protein Function Prediction](https://github.com/recluze/deepseq/raw/master/imgs/dl-arch.png "Deep Learning for Protein Function Prediction")

 # Authors:

- Nauman (mohamamd.nauman@nu.edu.pk, mnauman@mpi-sws.org, recluze.wordpress.com) -- Queries about ML should go here.
- Hafeez ur Rehman (hafeez.urrehman@nu.edu.pk) -- Queries about Bioinformatics should go here.

Preprint of related publication available here: http://www.biorxiv.org/content/early/2017/07/25/168120 
 
![Domain localization](https://github.com/recluze/deepseq/raw/master/imgs/domain-localization.png "Domain Localization")


# Import points:
    - Requires python2.7
    - See requirements.txt for exact version of libraries used. Keras v1.2.1 gives errors so use keras 1.2.0
    - I've used theano backend for keras. If you use tensorflow, I think you will have issues.
    - It's suggested that you use `virtualenv` to create a new environment and then install required packages.
        ```
        pip install virtualenv
        virtualenv bi
        cd bi
        . bin/activate
        git cone <git_repo_url>
        pip install -r <git_repo_name>src/requirements.txt
        ```
    - Set keras/theano to use the GPU. (Only do this on the GPU machine.) Put the following in `~/.theanorc`
        ```
        [global]
        device = gpu
        floatX = float32
        ```
    - Set keras to use theano. In `~/.keras/keras.json`, put the following:
        ```
        {
            "image_dim_ordering": "th",
            "epsilon": 1e-07,
            "floatx": "float32",
            "backend": "theano"
        }
        ```

# Execution
The source is executed in several steps.

1. First, data needs to be downloaded to `data-scrapes` folder.
    - Needs to be in FASTA format along with annotation file in .txt
    - This is already done for human proteins


2. These scrapes need to be converted to a format that we read later.
    - This is done through the `python src/scrape2vec.py`.
    - Variables to set: `scrape_dir`, `out_file`, `out_file_fns`, `out_file_unique_functions`
    - Use `function_usage_cutoff` variable to remove function used fewer times than this number
    - This step has already been done for human proteins downloaded in step 1.


3. Once output files are created from above step, you can run training/validation.
    - This is done through `python train.py`
    - Some variables need to be set (although current `train.py` can be executed as is to reproduce our experiments):
        * See top of `train.py` for parameters of training that you can set
        * `target_function` can be set to train for a particular function. Set to empty string to train for all functions
        * To quickly check code on slow machines, set `restrict_sample_size` to, say, 10.
        * `results_dir` is where results will be stored. These will be `-console.txt`, `-results.txt` and `-saved-model.h5` prefixed with `exp_id` i.e. the experiment ID.
        * Bottom of `train.py`, need to set `sequences_file` and `funtions_file` created from step 2 above
        * In `utils.py`, need to set `unique_function_file` variable. (Sorry for this clumsiness. I'm too lazy to fix this.)
    - Actual model is defined in `get_*_model` functions in `models.py`. This is called from `train.py` during training.

# License

This code is provided under the MIT License. 

Copyright 2017 Mohammad Nauman, Hafeez-ur-Rehman  

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
