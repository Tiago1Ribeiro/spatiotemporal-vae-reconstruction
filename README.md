conda create --name cvae python=3.10  
pip install -r requirements.txt  
zenodo_get --output data 10.5281/zenodo.7944963D  

### To do list:
- [] Testar CVAE refatorado (resolver problema com mse_kl_loss)
- [] Implementar os 4 modos de treino
- [] Construir modelos alternativos ao CVAE do artigo ADBIS 2023
- [] Adaptar código para facilmente mudar a configuração do dataset (total, amostrado, total extrapolação, amostrado extrapolação)

