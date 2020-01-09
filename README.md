# Deep Archetype Analysis
![Latent Traversal inbetween the Archetypes](https://github.com/bmda-unibas/DeepArchetypeAnalysis/blob/master/animation.gif "Traversal inbetween the Archetypes")
## Repository Description
Code of the Deep Archetype Analysis[1] applied to the JAFFE dataset[2].
Created in context of the <b>NeurIPS 2019 Workshop <i>Learning Meaningful Representations of Life</i></b> and the <b>IJCV 2019 <i>GCPR Special Issue</i></b>. 

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

# Usage

The JAFFE labels / emotions as a CSV have to be at 'DeepArchetypeAnalysis/jaffe/labels.csv'.
The images in 'DeepArchetypeAnalysis/jaffe/images'.

For downloading the data you can use the provided 'Makefile', i.e. just run 'make'.

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


### Code Structure
The main components & NN architecture are available in 'AT_lib/lib_vae.py'.
Plotting and other utilities are in 'AT_lib/lib_plt.py' and 'AT_lib/lib_at.py' respectively.
The main code is given in 'daa_JAFFE.py'. 

Aside from the default settings, different priors as well as a vanilla VAE with the same architecture are available.

# References
[1] [Keller S.M., Samarin M., Wieser M., Roth V. (2019) Deep Archetypal Analysis. In: Fink G., Frintrop S., Jiang X. (eds) Pattern Recognition. DAGM GCPR 2019. Lecture Notes in Computer Science, vol 11824. Springer, Cham](https://doi.org/10.1007/978-3-030-33676-9_12)

[2] Lyons, Michael J., et al. "The Japanese female facial expression (JAFFE) database." Proceedings of third international conference on automatic face and gesture recognition. 1998.
