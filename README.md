# Análise de Dados para Otimização de Recrutamento

Este projeto tem como objetivo processar dados de vagas, candidatos e prospecções para construir um modelo de Machine Learning capaz de prever a probabilidade de "match" (compatibilidade) entre um candidato e uma vaga.

O painel de otimização de recrutamento, como visualizado na imagem, permite que os usuários filtrem vagas e busquem candidatos, facilitando o processo de seleção.

## Autores

Este projeto foi desenvolvido por:
* Barbara Rodrigues Prado RM357381
* Edvaldo Torres RM357417

Criado para a Fase 5 do curso de Data Analytics, da Pós Tech - FIAP.

## Visão Geral do Projeto

O pipeline de Machine Learning implementado neste projeto envolve as seguintes etapas:

1.  **Importação de Bibliotecas e Configurações Iniciais**:
    * Configuração do ambiente e importação das bibliotecas necessárias como Pandas, Scikit-learn, Matplotlib, Seaborn, etc.
    * Criação de pastas para dados brutos, processados e artefatos do modelo (`data/raw/`, `data/`, `artifacts/`).

2.  **Processamento de Dados**:
    * **Dados de Vagas**:
        * Download do arquivo `vagas.json` do Google Drive.
        * Transformação do JSON em DataFrame.
        * Seleção de campos de interesse e pré-limpeza.
        * **Engenharia de Features**: Extração de modalidade de trabalho (Remoto, Híbrido, Presencial), flag para vaga SAP, codificação ordinal de níveis de idioma (inglês, espanhol), limpeza de áreas de atuação, criação de texto combinado da vaga para NLP, e extração de tecnologias (features `tech_*`) a partir das descrições.
        * Generalização do título da vaga em categorias.
    * **Dados de Candidatos**:
        * Download do arquivo `applicants.json` do Google Drive.
        * Transformação do JSON em DataFrame, concatenando descrições e títulos de experiências profissionais.
        * Seleção de campos de interesse e limpeza profunda dos dados.
        * **Engenharia de Features**: Criação de texto combinado do candidato para NLP, extração de tecnologias/habilidades (features `skill_*`), generalização do título profissional, codificação ordinal de níveis de idioma, padronização de níveis acadêmico e profissional, e padronização da informação de PCD.
    * **Dados de Prospecções**:
        * Download do arquivo `prospects.json` do Google Drive.
        * Transformação do JSON aninhado em um DataFrame plano.
        * Limpeza e pré-processamento, incluindo conversão de datas.
        * **Engenharia de Features**: Cálculo da duração da etapa do processo, padronização da modalidade da vaga (a partir dos dados de prospecção), agrupamento da situação do candidato (Ex: Em Processo, Finalizado - Contratado), e análise de comentários para identificar menção a valores monetários.

3.  **Merge dos DataFrames**:
    * União dos DataFrames processados de prospecções, vagas e candidatos em um DataFrame mestre (`df_master`).

4.  **Preparação Final dos Dados para Modelagem**:
    * Limpeza de linhas com merges incompletos.
    * Criação da variável alvo `foi_contratado` (1 para sucesso, 0 para não sucesso) com base na `situacao_candidato`.
    * Filtragem do DataFrame para incluir apenas os casos com desfecho conhecido (contratado ou não contratado), resultando no `df_modelagem`.

5.  **Análise Exploratória de Dados (EDA)**:
    * Criação de features de compatibilidade no `df_modelagem`, como `compat_ingles`, `compat_espanhol` (se o nível do candidato atende ao da vaga), `total_techs_vaga`, `skills_match_count` e `skills_faltantes_vaga`.

6.  **Preparação das Features para Modelagem**:
    * Seleção de features numéricas e categóricas.
    * Aplicação de One-Hot Encoding nas features categóricas.
    * Seleção final das colunas de features para o modelo, incluindo as `tech_*` da vaga e `skill_*` do candidato.
    * Divisão dos dados em conjuntos de treino e teste (`X_train`, `X_test`, `y_train`, `y_test`) com estratificação.

7.  **Treinamento e Avaliação de Modelos Baseline**:
    * **Regressão Logística**: Treinado com `class_weight='balanced'`.
    * **Random Forest Classifier**: Treinado com `class_weight='balanced'`.
    * Avaliação utilizando Acurácia, AUC-ROC, Matriz de Confusão e Relatório de Classificação.
    * Análise inicial da importância das features do Random Forest.

8.  **Análise dos Resultados e Otimização do Modelo**:
    * Ajuste do threshold de decisão do Random Forest para otimizar métricas como F1-score, precisão e recall para a classe de interesse.
    * **Otimização de Hiperparâmetros**: Uso do `RandomizedSearchCV` para encontrar os melhores hiperparâmetros para o Random Forest, focando no F1-score da classe positiva.
    * Avaliação do modelo Random Forest otimizado.
    * Análise final da importância das features do modelo otimizado e salvamento do gráfico.

9.  **Demonstração do Modelo**:
    * Seleção de exemplos de Verdadeiro Positivo (TP) e Verdadeiro Negativo (TN) do conjunto de teste.
    * Análise dos valores das features mais importantes para esses exemplos.

10. **Preparação de Arquivos para Aplicação Streamlit**:
    * Salvamento dos DataFrames processados (`vagas_processadas.csv`, `candidatos_processados.csv`).
    * Salvamento do modelo final treinado (`modelo_recrutamento_rf.joblib`).
    * Salvamento das colunas do modelo (`colunas_modelo.joblib`).
    * Salvamento de artefatos de engenharia de features (mapas de codificação, listas de tecnologias) em `artefatos_engenharia.joblib`.
    * Salvamento dos exemplos TP e TN para demonstração no Streamlit (`exemplo_tp_streamlit.csv`, `exemplo_tn_streamlit.csv`).

## Tecnologias Utilizadas

* Python 3.x
* **Bibliotecas Principais**:
    * Pandas: Para manipulação e análise de dados.
    * NumPy: Para operações numéricas.
    * Scikit-learn: Para Machine Learning (modelos, métricas, pré-processamento).
    * Matplotlib & Seaborn: Para visualização de dados.
    * Gdown: Para download de arquivos do Google Drive.
    * Joblib: Para salvar e carregar modelos e artefatos Python.
    * JSON: Para manipulação de arquivos JSON.
    * Re: Para operações com expressões regulares.
    * OS: Para interações com o sistema operacional (criação de pastas).

## Estrutura de Arquivos

.
├── artifacts/
│   ├── artefatos_engenharia.joblib
│   ├── colunas_modelo.joblib
│   ├── exemplo_tn_streamlit.csv
│   ├── exemplo_tp_streamlit.csv
│   ├── feature_importances.png
│   └── modelo_recrutamento_rf.joblib
├── data/
│   ├── raw/
│   │   ├── applicants_raw.json
│   │   ├── prospects_raw.json
│   │   └── vagas_raw.json
│   ├── candidatos_processados.csv
│   └── vagas_processadas.csv
├── desenvolvimento do modelo.py
└── README.md

# Como Executar

1.  **Pré-requisitos**:
    * Python 3.x instalado.
    * Acesso à internet para download dos datasets do Google Drive.
    * As IDs dos arquivos no Google Drive estão especificadas no script (`file_id_vagas`, `file_id_applicants`, `file_id_prospects`).

2.  **Instalação de Dependências**:
    É recomendável criar um ambiente virtual.
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn gdown joblib
    ```

3.  **Executar o Script**:
    Navegue até o diretório do projeto e execute o script principal:
    ```bash
    python "desenvolvimento do modelo.py"
    ```
    O script irá:
    * Criar as pastas `data/raw/`, `data/` e `artifacts/` se não existirem.
    * Baixar os dados brutos para `data/raw/`.
    * Processar os dados e salvar os arquivos processados em `data/`.
    * Treinar o modelo de Machine Learning.
    * Salvar o modelo treinado e outros artefatos (como listas de colunas, mapas de engenharia de features e exemplos) na pasta `artifacts/`.

      
4.  **Executar a Aplicação Streamlit**:
    Após a execução bem-sucedida do script `desenvolvimento do modelo.py` e a geração dos artefatos, execute o aplicativo Streamlit:
    ```bash
    streamlit run app.py
    ```
    Isso iniciará um servidor local e abrirá o "Painel de Otimização de Recrutamento" no seu navegador.

## Painel de Otimização de Recrutamento (Streamlit)

Os artefatos gerados pelo script `desenvolvimento do modelo.py` alimentam uma aplicação interativa construída com Streamlit.
