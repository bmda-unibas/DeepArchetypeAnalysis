# Learning Extremal Representations with Deep Archetypal Analysis
![Latent Traversal inbetween the Archetypes](https://github.com/bmda-unibas/DeepArchetypeAnalysis/blob/master/animation.gif "Traversal inbetween the Archetypes")
## Repository Description
We present a sample implementation (TensorFlow 1.12) of Deep Archetypal Analysis [1] applied to the JAFFE dataset [2][3]. 
This repository was created in the context of the <b>NeurIPS 2019 Workshop on <i>Learning Meaningful Representations of 
Life</i></b> and the <b>IJCV 2019 <i>GCPR Special Issue</i></b>. 

The workshop poster is provided as a pdf: [NeurIPS19_LMRL_poster.pdf](NeurIPS19_LMRL_poster.pdf)

## Abstract
Archetypes are typical population representatives  in  an  extremal  sense,  where  typicality  is  understood as the 
most extreme manifestation of a trait or feature. 

In linear feature space, archetypes approximate the data convex hull 
allowing all data points to be expressed as convex mixtures of archetypes. However, it might not always be possible to 
identify meaningful archetypes in a given feature space. As features are selected a priori, the resulting representation 
of the data might  only  be  poorly  approximated  as  a  convex  mixture. Learning an appropriate feature space and 
identifying suitable archetypes simultaneously addresses this problem.  This  paper  introduces  a  generative 
formulation  of  the  linear  archetype  model,  parameterized  by neural networks. 

By introducing the distance dependent archetype loss, the linear archetype model can be integrated  into  the  latent 
space  of  a  variational  autoencoder,  and  an  optimal  representation  with  respect  to the unknown archetypes can be
learned end-to-end. The reformulation of linear Archetypal Analysis as a variational  autoencoder  naturally  leads  to 
an  extension  of the model to a deep variational information bottleneck, allowing  the  incorporation  of  arbitrarily 
complex  sideinformation during training. 

 As a consequence, the answer to the question ”What is typical in a given dataset?” can be guided by this additional information.

# How to run the Code

We provide a Makefile to set up your environment in which you can run the code.

Note that:

- The setup relies on conda and the required libraries are specified in `environment.yml`.
- The implementation is meant to be executed on a <b>GPU</b>. In order to run the code on a CPU some modifications are required. 
For instance tensorflow-gpu needs to be replace by the CPU version.
- The JAFFE labels (emotion scores) as a CSV are expected to be at `DeepArchetypeAnalysis/jaffe/labels.csv` with the corresponding 
images in the folder `DeepArchetypeAnalysis/jaffe/images`.

### Makefile Setup
For automatically downloading, preprocessing JAFFE and setting up the conda environment, a Makefile is provided. Navigate to the 
`DeepArchetypeAnalysis` folder and perform the following step in your terminal.
```
make
```
downloads JAFFE and creates the conda environment `deepaa`.

## Access to JAFFE changed
Since publishing this code, the access to the JAFFE dataset has been restricted.
Consequently the makefile that downloaded the data will not be working anymore.
For research purposes, the data can be instead requested via https://zenodo.org/record/3451524 .


### Running the Script

With
```
source activate deepaa
```
the conda environment is activated and the script can be executed. Use
```
python daa_JAFFE.py
```
to execute the script with the default arguments for number of epochs, batch size etc. Check out `daa_JAFFE.py` for the available arguments, e.g. 
```
python daa_JAFFE.py --n-epochs 100 --dim-latentspace 5
```

### Code Structure

The main code is given in `daa_JAFFE.py`. The main components of the neural network architecture are available in `AT_lib/lib_vae.py`.
Plotting and other utilities are provided in `AT_lib/lib_plt.py` and `AT_lib/lib_at.py`, respectively.

Aside from the default settings, different priors as well as a vanilla VAE with the same architecture are available.

# References
[1] [Keller S.M., Samarin M., Wieser M., Roth V. (2019) Deep Archetypal Analysis. In: Fink G., Frintrop S., Jiang X. (eds) Pattern Recognition. DAGM GCPR 2019. Lecture Notes in Computer Science, vol 11824. Springer, Cham](https://doi.org/10.1007/978-3-030-33676-9_12)


[2] Lyons, Michael J., et al. "The Japanese female facial expression (JAFFE) database." Proceedings of third international conference on automatic face and gesture recognition. 1998.

[3] Lyons, Michael J., Miyuki Kamachi, and Jiro Gyoba. "Coding facial expressions with Gabor wavelets (IVC special issue)." arXiv preprint arXiv:2009.05938 (2020).
