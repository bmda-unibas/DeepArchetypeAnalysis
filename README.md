# Deep Archetype Analysis
## Repository Description
Code of the Deep Archetype Analysis[1] applied to the JAFFE dataset[2].
Created in context of the <b>NeurIPS 2019 Workshop <i>Learning Meaningful Representations of Life</i></b> and the <b>IJCV 2019 <i>GCPR Special Issue</i> </b>. 

## Abstract
<i>“Deep Archetypal Analysis”</i> generates latent representations of high-dimensional datasets in
terms of fractions of intuitively understandable basic entities called archetypes. 
The proposed method is an extension of linear “Archetypal Analysis” (AA), an unsupervised method to represent
multivariate data points as sparse convex combinations of extremal elements of the dataset. 
Unlike the original formulation of AA, <i>“Deep AA”</i> can also handle side information and provides the ability for data-driven representation learning which reduces the dependence on expert knowledge.

# Usage

The JAFFE labels / emotions as a CSV have to be at 'DeepArchetypeAnalysis/jaffe/labels.csv'.
The images in 'DeepArchetypeAnalysis/jaffe/images'.

Helper functions for downloading the data will be added in the future.


# References
[1] Keller, Sebastian Mathias, et al. "Deep Archetypal Analysis." arXiv preprint arXiv:1901.10799 (2019).

[2] Lyons, Michael J., et al. "The Japanese female facial expression (JAFFE) database." Proceedings of third international conference on automatic face and gesture recognition. 1998.
