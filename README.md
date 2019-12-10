# Deep Archetype Analysis
![Latent Traversal inbetween the Archetypes](https://github.com/bmda-unibas/DeepArchetypeAnalysis/blob/master/animation.gif "Traversal inbetween the Archetypes")

## Repository Description
Code of the Deep Archetype Analysis [1] applied to the JAFFE dataset [2].
Created in context of the <b>NeurIPS 2019 Workshop "Learning Meaningful Representations of Life"</b> and the <b>IJCV 2019 "GCPR Special Issue"</b>. 

## Abstract
<i>“Deep Archetypal Analysis”</i> generates latent representations of high-dimensional datasets in
terms of fractions of intuitively understandable basic entities called archetypes. 
The proposed method is an extension of linear “Archetypal Analysis” (AA), an unsupervised method to represent
multivariate data points as sparse convex combinations of extremal elements of the dataset. 
Unlike the original formulation of AA, <i>“Deep AA”</i> can also handle side information and provides the ability for data-driven representation learning which reduces the dependence on expert knowledge.

# Usage
<b>This setup relies on conda.</b> The setup is provided in `environment.yml`

For automatically setting up a virtual environment and downloading & preprocessing JAFFE, a Makefile is provided.

```
# Download JAFFE; create virtenv in directory ./venv
>make
# Activate virtual environment
>source activate venv
# run code with default settings
>python daa_JAFFE.py
```


# Authors
Sebastian Matthias Keller, Maxim Samarin, Fabricio Arend Torres, Mario Wieser, Volker Roth

# References
[1] [Keller, Sebastian Mathias, et al. "Deep Archetypal Analysis." arXiv preprint arXiv:1901.10799 (2019).](https://arxiv.org/abs/1901.10799)

[2] [Lyons, Michael J., et al. "The Japanese female facial expression (JAFFE) database." Proceedings of third international conference on automatic face and gesture recognition. 1998.](https://zenodo.org/record/3451524)
