## Modelling forest fire dynamics using conditional variational auto-encoders (Under Review)

## TL;DR

This research applies deep networks to simulate the spatial and temporal evolution of a forest fire's impact on a region. Using a Conditional Variational Autoencoder (CVAE) model, we generate intermediate representations of the burnt area's evolution. we also employ a CVAE model to predict future fire propagation behavior, estimating the burnt area at various time horizons and propagation stages. Real-world data is used to assess the model's performance, with results showing high similarity and temporal consistency metrics, indicating that the application of CVAE in this context is capable.

## Prerequisites

- `conda` for managing Python environments.
- `git` for version control.
- `zenodo-get` downloader.
- Python 3.10.

## Operating System

This project has been tested on the following operating systems:

- Pop! OS 22.04 LTS (Nvidia)
- Windows 11

## Main Settings

- Tensorflow 2.15.0
- CUDA 11.5

## Setting up the environment

```bash
# Create a new Conda environment named 'cvae' with Python version 3.10
conda create --name cvae python=3.10  

# Clone the 'spatiotemporal-vae-reconstruction' project from GitHub
git clone https://github.com/Tiago1Ribeiro/spatiotemporal-vae-reconstruction.git

# Change the current directory to the cloned project's directory
cd spatiotemporal-vae-reconstruction

# Install the Python dependencies listed in the 'requirements.txt' file
pip install -r requirements.txt  

# Downloads *BurnedAreaUAU* dataset and save it in the 'dataset' directory
zenodo_get --output dataset 10.5281/zenodo.7944963 # or dowload it from the link below
```

## To do list (🚧):

- [ ] Incluir as *U-Net samples (Base, RED, 3D)* e *sampled_masks.txt* no dataset (ZENODO v2)
- [ ] Test repo in other machines

## Dataset Citation

```bibtex
@misc{ba_uav_dataset,
  author       = {Ribeiro, Tiago F. R. and Silva, Fernando and Moreira, Jos\'e and Costa, Ro\'erio Lu\'is de C.},
  title        = {BurnedAreaUAV Dataset (v1.1)},
  month        = may,
  year         = 2023,
  publisher    = {Zenodo},
  version      = {1.1},
  doi          = {10.5281/zenodo.7944963},
}
```
