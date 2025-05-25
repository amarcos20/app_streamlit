
# Sistema de Intervenção de Estudantes


## Descrição do Projeto

Esta aplicação web interativa, desenvolvida com Streamlit, tem como objetivo analisar e visualizar dados de desempenho de estudantes e utilizar um modelo de Machine Learning para prever o sucesso académico. A aplicação baseia-se no dataset `student-data.csv` e num pipeline de Machine Learning (pré-processamento e modelo) previamente treinado e guardado.

A ferramenta permite explorar os dados, fazer previsões individuais para um aluno e analisar as métricas de avaliação e a interpretabilidade do modelo treinado.

## Funcionalidades

*   **Início:** Visão geral da aplicação e das suas capacidades.
*   **Exploração de Dados:** Analise as características do dataset original através de resumos estatísticos, distribuições e visualizações de relações entre variáveis, incluindo a correlação com a variável alvo.
*   **Previsão Individual:** Introduza os dados de um aluno e obtenha uma previsão instantânea sobre a sua probabilidade de passar ou não no exame final, utilizando o modelo treinado. Permite selecionar diferentes modelos guardados na pasta `artefacts`.
*   **Análise do Modelo Treinado:** Veja as métricas de desempenho (Acurácia, Precisão, Recall, F1-Score, AUC ROC) e a matriz de confusão do modelo principal guardado (`best_model.joblib`) no conjunto de teste. Inclui, se aplicável, visualizações de importância de características ou coeficientes para entender as decisões do modelo.
*   **Avaliação e Comparação de Modelos:** Permite selecionar diferentes tipos de algoritmos de Machine Learning, treiná-los temporariamente nos dados de treino processados e avaliar a sua performance no conjunto de teste. Útil para comparar abordagens.
*   **Documentação:** Descrição detalhada do dataset, das características, dos artefactos do modelo e das secções da aplicação.

## Pré-requisitos

Para executar esta aplicação, precisa de ter o Python instalado no seu sistema. Recomenda-se a utilização de um ambiente virtual.

As bibliotecas Python necessárias estão listadas abaixo. Pode instalá-las manualmente ou através de um ficheiro `requirements.txt`.

## Instalação

1.  Clone ou descarregue os ficheiros do projeto para o seu computador.
2.  Navegue para o diretório do projeto no terminal.
3.  Instale as bibliotecas Python necessárias. Se tiver um ficheiro `requirements.txt` fornecido com o projeto, execute:
    ```bash
    pip install -r requirements.txt
    ```
    Caso contrário, instale manualmente as dependências usadas no código:
    ```bash
    pip install streamlit pandas numpy scikit-learn plotly streamlit-option-menu joblib
    ```

## Configuração dos Dados e Artefactos

Esta aplicação espera que os seguintes ficheiros existam no diretório correto:

*   `student-data.csv`: O dataset original (ou uma versão processada inicial) usado para EDA. Deve estar na pasta raiz do projeto.
*   `artefacts/`: Uma subpasta na raiz do projeto contendo os artefactos do pipeline de Machine Learning. É necessário que existam pelo menos:
    *   `artefacts/preprocessor.joblib`: O objeto pré-processador treinado (e.g., StandardScaler, OneHotEncoder, ColumnTransformer).
    *   `artefacts/best_model.joblib`: O modelo de Machine Learning treinado (e.g., LogisticRegression, RandomForestClassifier). Este é o modelo principal analisado e usado por defeito.
    *   `artefacts/original_input_columns.joblib`: Uma lista (ou similar) contendo os nomes das colunas de input originais que o pré-processador espera.
    *   `artefacts/processed_feature_names.joblib`: Uma lista (ou similar) contendo os nomes das características após o pré-processamento.
    *   *(Opcional)* Outros ficheiros `.joblib` de modelos treinados que queira disponibilizar na secção "Previsão Individual".
*   `data/processed/`: Uma subpasta na raiz do projeto contendo os dados processados. É necessário que existam:
    *   `data/processed/train_processed.csv`: Dados de treino processados (DataFrame).
    *   `data/processed/test_processed.csv`: Dados de teste processados (DataFrame).
    *   Ambos os ficheiros CSV processados devem conter a coluna alvo (`passed_mapped`, por defeito) e as características processadas.

**Nota:** Estes ficheiros (`*.joblib` e `*_processed.csv`) devem ser gerados a partir do seu script ou notebook de treino do modelo antes de executar a aplicação Streamlit. Certifique-se de que os caminhos no código (`artefacts/` e `data/processed/`) correspondem à estrutura de pastas do seu projeto.

## Como Executar

1.  Certifique-se de que está no diretório raiz do projeto no terminal.
2.  Execute o seguinte comando :
```bash
    streamlit run sua_aplicacao.py
    ```
  
    (Substitua `app.py` pelo nome do ficheiro Python que contém o código da aplicação Streamlit).
    
4.  A aplicação será aberta no seu navegador padrão.

## Estrutura de Pastas Esperada

```
.
├── artefacts/
│   ├── preprocessor.joblib
│   ├── best_model.joblib
│   ├── original_input_columns.joblib
│   ├── processed_feature_names.joblib
│   └── (outros_modelos).joblib  
├── data/
│   └── processed/
│       ├── train_processed.csv
│       └── test_processed.csv
├── student-data.csv             # Dataset original
└── sua_aplicacao.py             # Este ficheiro Streamlit
```

## Projeto Académico

Esta aplicação foi desenvolvida no âmbito de um projeto académico da cadeira de Elementos de Inteligência Artifcial e Ciência de Dados pelos alunos:

Afonso Marcos(202404088)
Pedro Afonso(202404125)
Afonso Silva(202406661)


© 2025 Sistema de Intervenção de Estudantes. Desenvolvido com Streamlit.
