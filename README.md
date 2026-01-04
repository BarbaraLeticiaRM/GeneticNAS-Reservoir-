# GeneticNAS + Reservoir

### Título do Projeto
**Evoluindo Arquiteturas 1D: Busca Neural Genética com Reservoir Computing**

### Sobre o Projeto (About)
Este repositório contém a implementação oficial da dissertação de mestrado apresentada ao Programa de Pós-Graduação em Ciência da Computação da UFOP. O projeto propõe uma extensão à estratégia **[Genetic Network Architecture Search (GeneticNAS)](https://github.com/haihabi/GeneticNAS.git)**, adaptando-a para o processamento de sinais unidimensionais (1D) e adicionando a operação de **Reservoir Computing (RC)** ao seu espaço de busca,.

O objetivo central é automatizar a criação de arquiteturas de redes neurais otimizadas para classificação de séries temporais, combinando a eficiência do *Weight Sharing* (compartilhamento de pesos) com a capacidade do Reservoir Computing de modelar dinâmicas temporais,.

### Principais Funcionalidades
*   **GeneticNAS Adaptado para 1D:** Modificação do algoritmo original (focado em imagens) para operar com sinais unidimensionais como ECG e EEG.
*   **Novo Operador de Reservoir Computing:** Implementação de uma *Echo State Network* (ESN) encapsulada como um operador no espaço de busca do NAS. Inclui controle de hiperparâmetros como tamanho do reservatório, raio espectral e esparsidade,.
*   **Espaço de Busca Híbrido:** O algoritmo pode escolher entre operações convolucionais (Conv1D), Pooling e Reservoir Computing para compor as células da rede,.
*   **Otimização em Dois Estágios:** Busca da arquitetura utilizando Algoritmo Genético com SGD, seguida pela otimização dos pesos da melhor arquitetura encontrada,.

### Bases de Dados Utilizadas
Os experimentos foram configurados para rodar nas seguintes bases de dados (com pré-processamento específico incluído no código):
1.  **MIT-BIH Arrhythmia Database:** Classificação de arritmias cardíacas em 4 classes, utilizando janelas de batimentos normalizadas,.
2.  **Physionet EEG Database:** Tarefa de biometria baseada em eletroencefalograma (Motor Imagery), utilizando a banda de frequência gama (30-50 Hz),.

### Requisitos do Sistema
Conforme o setup experimental descrito na dissertação, o código foi validado no seguinte ambiente:
*   **Python:** 3.10
*   **PyTorch:** 2.6.0
*   **TorchVision:** 0.21.0
*   **CUDA:** 12.4
  
São utilizados os seguintes pacotes:
* Scikit-learn
* Numpy
* PyGraphviz 

### Resultados Alcançados
A inclusão do operador de Reservoir Computing demonstrou desempenho superior ao GeneticNAS padrão. Destaques:
*   **ECG (MIT-BIH):** Aumento de aproximadamente 2% na acurácia e melhoria na estabilidade (menor desvio padrão) com um reservatório de tamanho 360 e densidade de 5%.
*   **EEG (Physionet):** Aumento de 5,5% no F1-score utilizando um reservatório de tamanho 1920.

### Como rodar a fase de busca da arquitetura
A primeira fase dos experimentos consiste em rodar a busca da melhor arquiteura no espaço de busca.

**MIT-BIH Arrhythmia Database:**
```
       python main.py --dataset_name MIT --config_file ./configs/config_search.json
```
**Physionet EEG Database:** 
```
       python main.py --dataset_name EEG --config_file ./configs/config_search.json
```
### Como rodar a fase de otimização dos pesos
Após encontrar a melhor arquitetura, o próximo passo é otimizar os pesos desta. 

**MIT-BIH Arrhythmia Database:**
```
       python main.py --dataset_name MIT --final 1 --serach_dir $LOG_DIR --config_file ./configs/config_final.json
```
**Physionet EEG Database:** 
```
       python main.py --dataset_name EEG --final 1 --serach_dir $LOG_DIR --config_file ./configs/config_final.json
```

Para inclusão da operação de reservoir, basta incluir os argumentos:
```
       --reservoir_size $TAM --sparsity $VAL --spectral_radius $VAL --reservoir_operation $OPERATION
``` 

### Como rodar a etapa de teste
Após encontrados os melhores pesos, é feita a etapa de teste para verificar a arquitetura e os pesos encontrados no conjunto de teste da base de dados utilizada. 

**MIT-BIH Arrhythmia Database:**
```
       python test.py --dataset_name MIT --model_dir $LOG_DIR
```
**Physionet EEG Database:** 
```
       python test.py --dataset_name EEG --model_dir $LOG_DIR
```

####As bases de dados adaptadas para o formato .pt, utilizado nos experimentos, podem ser acessadas em: [Dataset](https://drive.google.com/file/d/1coIzuGMc5Hxv-EEvKsKFGpj6i-10XAw1/view?usp=sharing) 
---



