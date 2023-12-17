```bash
# Create a new Conda environment named 'cvae' with Python version 3.10
conda create --name cvae python=3.10  

# Clone the 'spatiotemporal-vae-reconstruction' project from GitHub
git clone https://github.com/Tiago1Ribeiro/spatiotemporal-vae-reconstruction.git

# Change the current directory to the cloned project's directory
cd spatiotemporal-vae-reconstruction

# Install the Python dependencies listed in the 'requirements.txt' file
pip install -r requirements.txt  

# Download the dataset from Zenodo using its DOI (Digital Object Identifier) and save it in the 'dataset' directory
zenodo_get --output dataset 10.5281/zenodo.7944963
```
### To do list:
- [] Testar CVAE refatorado (resolver problema com mse_kl_loss)
- [] Implementar os 4 modos de treino
- [] Construir modelos alternativos ao CVAE do artigo ADBIS 2023
- [] Adaptar código para facilmente mudar a configuração do dataset (total, amostrado, total extrapolação, amostrado extrapolação)

