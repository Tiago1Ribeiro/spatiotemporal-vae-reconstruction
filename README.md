# Reconstructing Spatiotemporal Data with C-VAEs 

## To do list:
- [ ] Testar CVAE refatorado (resolver problema com mse_kl_loss)
- [ ] Incluir as U-Net samples no dataset!
- [ ] Implementar os 4 modos de treino
- [ ] Construir modelos alternativos ao CVAE do artigo ADBIS 2023
- [ ] Adaptar código para facilmente mudar a configuração do dataset  
      (total, amostrado, total extrapolação, amostrado extrapolação)

### Overview

Summary here.

### Prerequisites:
- <tt>conda</tt> for managing Python environments.
- <tt>git</tt> for version control.
- <tt>zenodo-get</tt> downloader.
- Python 3.10.


## Operating System

This project has been tested on the following operating systems:

- Pop! OS 22.04 LTS (Nvidia)
- Windows 11

## CUDA Versions

HERE


### Setting up the environment
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

