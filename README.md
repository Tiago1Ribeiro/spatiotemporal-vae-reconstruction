```bash
conda create --name cvae python=3.10  
git clone https://github.com/Tiago1Ribeiro/spatiotemporal-vae-reconstruction.git
cd spatiotemporal-vae-reconstruction
pip install -r requirements.txt  
zenodo_get --output data 10.5281/zenodo.7944963  
```

### To do list:
- [] Testar CVAE refatorado (resolver problema com mse_kl_loss)
- [] Implementar os 4 modos de treino
- [] Construir modelos alternativos ao CVAE do artigo ADBIS 2023
- [] Adaptar código para facilmente mudar a configuração do dataset (total, amostrado, total extrapolação, amostrado extrapolação)

