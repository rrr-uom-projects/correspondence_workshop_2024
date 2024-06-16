# Deep learning-based shape correspondence workshop (June 2024)

<a target="_blank" href="https://colab.research.google.com/github/rrr-uom-projects/correspondence_workshop_2024/blob/main/workshop_notebook.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open Notebook In Colab"/>
</a>

# Pre-requisites:
- Google account
- Basic knowledge of Python

### Overview:

In this hands-on tutorial we're going to go therough the process of setting up, training and evaluating the results produced by a popular learning-based correspondence method.

For this tutorial we shall use the SHREC'20 dataset of 3D animal shapes: http://robertodyke.com/shrec2020/index2.html

We will be using the "Neuromorph" correspondence model designed by Eisenberger et al. that was described briefly in the talk this morning. Paper: https://arxiv.org/pdf/2106.09431 Original code: https://github.com/facebookresearch/neuromorph

We'll be going through this tutorial step-by-step and please feel free to ask any questions along the way.

---
### Contents
The steps in this notebook are as follows:

0. Install prerequisites and set up the google colab environment
1. Explore the dataset
2. Pre-processing - Supervised vs. unsupervised learning for correspondence models
3. Setting up a learning-based correspondence model (Neuromorph)
4. Training the correspondence model
5. Evaluation - geodesic distance and landmarks
6. Exploration of limitations and possible solutions
---
