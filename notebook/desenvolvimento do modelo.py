"""
# **Análise de dados para otimização de recrutamento**
Este notebook tem como objetivo processar dados de vagas, candidatos e prospecções para construir um modelo de Machine Learning capaz de prever a probabilidade de "match" entre um candidato e uma vaga.

Criado para a fase 5 do curso de Data Analytics, da Pós Tech - FIAP
Grupo 32:
- Barbara Rodrigues Prado RM357381
- Edvaldo Torres RM357417
"""

# ## **1. Importação das bibliotecas e configurações iniciais**
import json
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gdown # ADICIONADO para downloads do Google Drive
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, make_scorer, f1_score, precision_recall_fscore_support
from scipy.stats import randint
import joblib

pd.set_option('display.max_columns', None)

# --- Criar pastas necessárias para dados brutos, processados e artefatos ---
path_data_raw = 'data/raw/'
path_data_processed = 'data/' 
path_artifacts = 'artifacts/'

for path in [path_data_raw, path_data_processed, path_artifacts]:
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Pasta '{path}' criada.")

# --- IDs dos arquivos no Google Drive ---
file_id_vagas = "1NNzV_w90OlbONFq6oeO4T5xV5lTb4TuM"
file_id_applicants = "1TQoEwhpYOwxjjZYmKVgAtEbN57kI2Slc"
file_id_prospects = "1QIlM6G2bdWkYbWKp9The5bEL4Kr-hxfR"

# --- Caminhos para salvar os arquivos brutos baixados ---
output_path_vagas_raw = os.path.join(path_data_raw, "vagas_raw.json")
output_path_applicants_raw = os.path.join(path_data_raw, "applicants_raw.json")
output_path_prospects_raw = os.path.join(path_data_raw, "prospects_raw.json")

# ---------------------------------------------------------------------------
# ## **2. Processamento de Dados de Vagas**
# ---------------------------------------------------------------------------
print("\n--- Iniciando Processamento de Dados de Vagas ---")
# ### *2.1. Download e Carregamento do Arquivo vagas.json*
data_vagas_json = {}
try:
    print(f"Baixando vagas.json de ID: {file_id_vagas} para {output_path_vagas_raw}...")
    gdown.download(id=file_id_vagas, output=output_path_vagas_raw, quiet=False, fuzzy=True)
    with open(output_path_vagas_raw, 'r', encoding='utf-8') as f:
        data_vagas_json = json.load(f)
    print(f"Arquivo '{output_path_vagas_raw}' carregado com sucesso.")
except Exception as e:
    print(f"Erro Crítico ao baixar ou carregar '{os.path.basename(output_path_vagas_raw)}': {e}")
    data_vagas_json = {}

# ### *2.2. Transformação do JSON de Vagas em DataFrame*
lista_vagas = []
if isinstance(data_vagas_json, dict):
    for vaga_id, detalhes_vaga in data_vagas_json.items():
        vaga_info = {"id_vaga": vaga_id}
        if isinstance(detalhes_vaga, dict):
            informacoes_basicas = detalhes_vaga.get("informacoes_basicas", {})
            perfil_vaga = detalhes_vaga.get("perfil_vaga", {})
            vaga_info.update(informacoes_basicas if isinstance(informacoes_basicas, dict) else {})
            vaga_info.update(perfil_vaga if isinstance(perfil_vaga, dict) else {})
        lista_vagas.append(vaga_info)
else:
    print("Estrutura do JSON de vagas não é um dicionário no nível raiz como esperado.")
df_vagas = pd.DataFrame(lista_vagas)

# ### *2.3. Seleção dos campos de interesse para vagas*
campos_selecionados_vagas = [
    "id_vaga", "titulo_vaga", "tipo_contratacao", "vaga_sap", "cliente",
    "empresa_divisao", "estado", "cidade", "nivel profissional", "nivel_academico",
    "nivel_ingles", "nivel_espanhol", "areas_atuacao", "principais_atividades",
    "competencia_tecnicas_e_comportamentais", "demais_observacoes"
]
df_vagas_processado = pd.DataFrame()
if not df_vagas.empty:
    for campo in campos_selecionados_vagas:
        if campo in df_vagas.columns:
            df_vagas_processado[campo] = df_vagas[campo]
        else:
            df_vagas_processado[campo] = pd.Series(dtype='object')
            print(f"Aviso (Vagas): Coluna '{campo}' não encontrada no DataFrame original. Será criada com NA.")
else:
    print("DataFrame de vagas original está vazio. df_vagas_processado será criado com colunas vazias.")
    for campo in campos_selecionados_vagas:
        df_vagas_processado[campo] = pd.Series(dtype='object')

# ### *2.4. Pré-limpeza de campos textuais de vagas*
if not df_vagas_processado.empty:
    text_cols_vagas_feature_eng = ["titulo_vaga", "principais_atividades", "competencia_tecnicas_e_comportamentais", "demais_observacoes", "areas_atuacao", "vaga_sap", "nivel_ingles", "nivel_espanhol", "cliente", "empresa_divisao"]
    for col in text_cols_vagas_feature_eng:
        if col in df_vagas_processado.columns:
            df_vagas_processado[col] = df_vagas_processado[col].fillna("Não Informado").astype(str)
        else:
            df_vagas_processado[col] = "Não Informado"
            print(f"Aviso (Vagas): Coluna '{col}' para pré-limpeza não existia e foi criada como 'Não Informado'.")

else:
    print("df_vagas_processado está vazio. Pré-limpeza de campos textuais de vagas ignorada.")

# ### *2.5. Engenharia de features para vagas*
# 2.5.1. Modalidade de Trabalho
if not df_vagas_processado.empty and "demais_observacoes" in df_vagas_processado.columns:
    def extrair_modalidade(texto):
        texto_lower = str(texto).lower() 
        if re.search(r"100% remoto|totalmente remoto|home office|trabalho remoto|remoto", texto_lower):
            if re.search(r"h[íi]brido", texto_lower): return "Híbrido"
            return "Remoto"
        elif re.search(r"h[íi]brido", texto_lower): return "Híbrido"
        elif re.search(r"presencial|no escrit[óo]rio|na planta|loca[çl][ãa]o", texto_lower) and not re.search(r"remoto|h[íi]brido", texto_lower): return "Presencial"
        return "Não Informado"
    df_vagas_processado["modalidade_trabalho"] = df_vagas_processado["demais_observacoes"].apply(extrair_modalidade)
else:
    if df_vagas_processado.empty: print("df_vagas_processado está vazio. Extração de modalidade de trabalho ignorada.")
    else:
        print("Coluna 'demais_observacoes' não encontrada em df_vagas_processado. Extração de modalidade de trabalho ignorada.")
        df_vagas_processado["modalidade_trabalho"] = "Não Informado"

# 2.5.2. Flag Vaga SAP
if not df_vagas_processado.empty and "vaga_sap" in df_vagas_processado.columns:
    df_vagas_processado["vaga_sap_bool"] = df_vagas_processado["vaga_sap"].apply(lambda x: 1 if str(x).lower() == "sim" else 0)
else:
    if df_vagas_processado.empty: print("df_vagas_processado está vazio. Criação de 'vaga_sap_bool' ignorada.")
    else:
        print("Coluna 'vaga_sap' não encontrada em df_vagas_processado. 'vaga_sap_bool' será definida como 0.")
        df_vagas_processado["vaga_sap_bool"] = 0

# 2.5.3. Nível de Idioma Ordinal
mapa_nivel_idioma = { "não informado": 0, "nenhum": 0, "básico": 1, "técnico": 2, "intermediário": 2, "avançado": 3, "fluente": 4, "nativo": 5 }
def codificar_idioma(nivel): return mapa_nivel_idioma.get(str(nivel).lower(), 0)

if not df_vagas_processado.empty:
    if "nivel_ingles" in df_vagas_processado.columns:
        df_vagas_processado["nivel_ingles_ordinal"] = df_vagas_processado["nivel_ingles"].apply(codificar_idioma)
    else:
        print("Coluna 'nivel_ingles' não encontrada em df_vagas_processado. 'nivel_ingles_ordinal' será definida como 0.")
        df_vagas_processado["nivel_ingles_ordinal"] = 0
    if "nivel_espanhol" in df_vagas_processado.columns:
        df_vagas_processado["nivel_espanhol_ordinal"] = df_vagas_processado["nivel_espanhol"].apply(codificar_idioma)
    else:
        print("Coluna 'nivel_espanhol' não encontrada em df_vagas_processado. 'nivel_espanhol_ordinal' será definida como 0.")
        df_vagas_processado["nivel_espanhol_ordinal"] = 0
else:
    print("df_vagas_processado está vazio. Codificação de nível de idioma ignorada.")
    df_vagas_processado["nivel_ingles_ordinal"] = 0
    df_vagas_processado["nivel_espanhol_ordinal"] = 0

# 2.5.4. Limpeza de 'areas_atuacao'
if not df_vagas_processado.empty and "areas_atuacao" in df_vagas_processado.columns:
    def limpar_area_atuacao(area): return str(area).replace("-", "").strip()
    df_vagas_processado["area_atuacao_limpa"] = df_vagas_processado["areas_atuacao"].apply(limpar_area_atuacao)
else:
    if df_vagas_processado.empty: print("df_vagas_processado está vazio. Limpeza de 'areas_atuacao' ignorada.")
    else:
        print("Coluna 'areas_atuacao' não encontrada em df_vagas_processado. 'area_atuacao_limpa' será definida como 'Não Informado'.")
        df_vagas_processado["area_atuacao_limpa"] = "Não Informado"

# 2.5.5. Texto Combinado da Vaga para NLP
if not df_vagas_processado.empty:
    text_fields_to_combine_nlp_vagas = ["titulo_vaga", "principais_atividades", "competencia_tecnicas_e_comportamentais"]
    df_vagas_processado["texto_completo_vaga"] = ""
    for field in text_fields_to_combine_nlp_vagas:
        if field in df_vagas_processado.columns:
            df_vagas_processado["texto_completo_vaga"] += df_vagas_processado[field].astype(str) + " "
        else:
            print(f"Aviso (Vagas): Campo '{field}' não encontrado para combinação em 'texto_completo_vaga'.")
    df_vagas_processado["texto_completo_vaga"] = df_vagas_processado["texto_completo_vaga"].str.strip().str.lower()
else:
    print("df_vagas_processado está vazio. Criação de 'texto_completo_vaga' ignorada.")
    df_vagas_processado["texto_completo_vaga"] = ""

# 2.5.6. Tratamento dos Campos 'cliente' e 'empresa_divisao'
if not df_vagas_processado.empty:
    for campo_tratamento_vaga in ["cliente", "empresa_divisao"]:
        if campo_tratamento_vaga in df_vagas_processado.columns:
            df_vagas_processado[campo_tratamento_vaga] = df_vagas_processado[campo_tratamento_vaga].astype(str).str.strip()
        else:
            print(f"Aviso (Vagas): Campo '{campo_tratamento_vaga}' não encontrado para tratamento.")
            df_vagas_processado[campo_tratamento_vaga] = "Não Informado"
else:
    print("df_vagas_processado está vazio. Tratamento de 'cliente' e 'empresa_divisao' ignorado.")

# 2.5.7. Extração de Tecnologias (Features tech_*)
tecnologias_lista_vagas = [
    'python', 'java', 'javascript', 'c#', '.net', 'sql', 'nosql', 'aws', 'azure', 'gcp',
    'docker', 'kubernetes', 'react', 'angular', 'vue', 'node.js', 'nodejs', 'php', 'ruby',
    'swift', 'kotlin', 'scala', 'sap', 'oracle', 'power bi', 'powerbi', 'tableau', 'excel', 'git',
    'typescript', 'api', 'rest', 'spring', 'django', 'flask', 'linux', 'html', 'css',
    'salesforce', 'jira', 'trello', 'agile', 'scrum', 'c++', 'c',
    'flutter', 'airflow', 'etl', 'hadoop', 'spark', 'machine learning', 'tensorflow',
    'pytorch', 'devops', 'selenium', 'testing', ' segurança', 'security', 'bi'
]
if not df_vagas_processado.empty and "texto_completo_vaga" in df_vagas_processado.columns:
    processed_tech_clean_names_vagas = set()
    for tech_original_name_vaga in tecnologias_lista_vagas:
        tech_clean_name_vaga = tech_original_name_vaga
        if tech_original_name_vaga == 'c++': tech_clean_name_vaga = 'cpp'
        elif tech_original_name_vaga == 'c': tech_clean_name_vaga = 'c_lang'
        elif tech_original_name_vaga == 'c#': tech_clean_name_vaga = 'csharp'
        elif tech_original_name_vaga == '.net': tech_clean_name_vaga = 'dotnet'
        elif tech_original_name_vaga == 'node.js' or tech_original_name_vaga == 'nodejs': tech_clean_name_vaga = 'nodejs'
        elif tech_original_name_vaga == 'power bi' or tech_original_name_vaga == 'powerbi': tech_clean_name_vaga = 'powerbi'
        else: tech_clean_name_vaga = tech_original_name_vaga.replace(' ', '_').replace('.', 'dot').replace('+', 'plus')

        if tech_clean_name_vaga in processed_tech_clean_names_vagas: continue
        processed_tech_clean_names_vagas.add(tech_clean_name_vaga)
        col_name_vaga = f"tech_{tech_clean_name_vaga}" 
        
        pattern_vaga = ''
        if tech_original_name_vaga == '.net': pattern_vaga = r'\b\.net\b'
        elif tech_original_name_vaga == 'c#': pattern_vaga = r'\bc#\b|\bc\s*sharp\b'
        elif tech_original_name_vaga == 'c++': pattern_vaga = r'\bc\+\+\b|\bc\s*plus\s*plus\b'
        elif tech_original_name_vaga == 'c': pattern_vaga = r'\b(linguagem\s+c|c(?!(?:\+\+|#|-|s|r|u|l|i|p|a|o|e|m|t|v|b|d|k|g|h|f)))\b'
        elif tech_original_name_vaga == 'node.js' or tech_original_name_vaga == 'nodejs': pattern_vaga = r'\bnode\.js\b|\bnodejs\b'
        elif tech_original_name_vaga == 'power bi' or tech_original_name_vaga == 'powerbi': pattern_vaga = r'\bpower\s*bi\b|\bpowerbi\b'
        elif len(tech_original_name_vaga) <= 3 and not tech_original_name_vaga in ['sap', 'aws', 'gcp', 'api', 'sql', 'css', 'git', 'etl', 'bi']:
            pattern_vaga = r'(?:^|\s|,|\(|\)|-|/)' + re.escape(tech_original_name_vaga) + r'(?:$|\s|,|\.|\(|\)|-|/)'
        else:
            pattern_vaga = r'\b' + re.escape(tech_original_name_vaga) + r'\b'

        if pattern_vaga:
             df_vagas_processado[col_name_vaga] = df_vagas_processado["texto_completo_vaga"].apply(lambda x: 1 if re.search(pattern_vaga, x, re.IGNORECASE) else 0)
else:
    if df_vagas_processado.empty: print("df_vagas_processado está vazio. Extração de tecnologias para vagas ignorada.")
    else: print("Coluna 'texto_completo_vaga' não encontrada para extração de tecnologias de vagas.")

# 2.5.8. Generalização do Título da Vaga (categoria_vaga)
if not df_vagas_processado.empty and "titulo_vaga" in df_vagas_processado.columns:
    def generalizar_titulo_vaga(titulo):
        titulo_lower = str(titulo).lower()
        if titulo_lower == "não informado": return "Outros/Não Especificado"
        if "consultor sap" in titulo_lower or "consultora sap" in titulo_lower or ("sap" in titulo_lower and ("consultant" in titulo_lower or "especialista" in titulo_lower or "consultor(a)" in titulo_lower)): return "Consultoria SAP"
        if "arquiteto sap" in titulo_lower or "architect sap" in titulo_lower: return "Arquitetura SAP"
        if "architect" in titulo_lower or "arquiteto" in titulo_lower or "tech lead" in titulo_lower or "líder técnico" in titulo_lower or "lider técnico" in titulo_lower: return "Liderança Técnica & Arquitetura"
        dev_keywords = ["desenvolvedor", "developer", "programador", "software engineer", "dev", "desenvolvimento", "development", "fullstack", "frontend", "backend", "mobile", "abap"]
        if any(keyword in titulo_lower for keyword in dev_keywords): return "Desenvolvimento de Software"
        data_keywords = ["dados", "data", "bi", "business intelligence", "analytics", "cientista", "scientist", "engenheiro de dados", "data engineer", "analista de dados"]
        if any(keyword in titulo_lower for keyword in data_keywords): return "Dados & BI"
        infra_keywords = ["infraestrutura", "infrastructure", "cloud", "aws", "azure", "gcp", "devops", "sysadmin", "rede", "network", "segurança", "security", "sre"]
        if any(keyword in titulo_lower for keyword in infra_keywords): return "Infra, Cloud & DevOps"
        pm_po_keywords = ["gerente de projetos", "project manager", "pm", "gpm", "product owner", "po", "product manager", "coordenador de projetos", "project coordinator", "scrum master", "agile coach"]
        if any(keyword in titulo_lower for keyword in pm_po_keywords): return "Gestão de Projetos & Produtos"
        qa_keywords = ["qa", "quality assurance", "testes", "tester", "analista de testes", "automação de testes"]
        if any(keyword in titulo_lower for keyword in qa_keywords): return "Qualidade & Testes"
        support_keywords = ["suporte", "support", "analista de suporte", "service desk", "operações", "operations", "sustentação"]
        if any(keyword in titulo_lower for keyword in support_keywords): return "Suporte & Operações"
        design_keywords = ["design", "designer", "ux", "ui", "product designer"]
        if any(keyword in titulo_lower for keyword in design_keywords): return "Design (UX/UI)"
        analyst_keywords = ["analista", "analyst", "especialista", "specialist"]
        if any(keyword in titulo_lower for keyword in analyst_keywords):
            if "negócios" in titulo_lower or "business" in titulo_lower: return "Funcional & Negócios"
            if "sistemas" in titulo_lower or "systems" in titulo_lower: return "Análise de Sistemas"
            if "processos" in titulo_lower: return "Análise de Processos"
            if "requisitos" in titulo_lower: return "Análise de Requisitos"
            if "financeiro" in titulo_lower or "contábil" in titulo_lower or "fiscal" in titulo_lower : return "Financeiro & Contábil"
            return "Analista (Genérico)"
        lead_coord_keywords = ["líder de equipe", "team lead", "coordenador", "supervisor", "gerente", "manager"]
        if any(keyword in titulo_lower for keyword in lead_coord_keywords): return "Liderança & Coordenação (Não Técnica)"
        return "Outros/Não Especificado"
    df_vagas_processado["categoria_vaga"] = df_vagas_processado["titulo_vaga"].apply(generalizar_titulo_vaga)
else:
    if df_vagas_processado.empty: print("df_vagas_processado está vazio. Generalização de título de vaga ignorada.")
    else:
        print("Coluna 'titulo_vaga' não encontrada em df_vagas_processado. 'categoria_vaga' será definida como 'Não Informado'.")
        df_vagas_processado["categoria_vaga"] = "Não Informado"

# 2.5.9. Renomear Coluna 'nivel profissional'
if not df_vagas_processado.empty:
    if "nivel profissional" in df_vagas_processado.columns:
        df_vagas_processado.rename(columns={"nivel profissional": "nivel_profissional_vaga"}, inplace=True)
        print("Coluna 'nivel profissional' de vagas renomeada para 'nivel_profissional_vaga'.")
    else:
        print("Aviso (Vagas): Coluna 'nivel profissional' não encontrada para renomear, criando 'nivel_profissional_vaga' como 'Não Informado'.")
        df_vagas_processado["nivel_profissional_vaga"] = "Não Informado"

# ### *2.6. Informações e visualização do DataFrame de vagas processado*
if 'df_vagas_processado' in globals() and not df_vagas_processado.empty:
    print("\n--- Informações do DataFrame de Vagas Processado (df_vagas_processado) ---")
    df_vagas_processado.info(verbose=False)
    print("\n--- Amostra do DataFrame de Vagas Processado ---")
    print(df_vagas_processado.head(2).to_string())
else:
    print("\nDataFrame df_vagas_processado não foi criado ou está vazio após o processamento.")

# ---------------------------------------------------------------------------
# ## **3. Processamento de Dados de Candidatos**
# ---------------------------------------------------------------------------
print("\n--- Iniciando Processamento de Dados de Candidatos ---")
# ### *3.1. Download e Carregamento do arquivo applicants.json*
data_applicants_json = {}
try:
    print(f"Baixando applicants.json de ID: {file_id_applicants} para {output_path_applicants_raw}...")
    gdown.download(id=file_id_applicants, output=output_path_applicants_raw, quiet=False, fuzzy=True)
    with open(output_path_applicants_raw, 'r', encoding='utf-8') as f:
        data_applicants_json = json.load(f)
    print(f"Arquivo '{output_path_applicants_raw}' carregado com sucesso.")
except Exception as e:
    print(f"Erro Crítico ao baixar ou carregar '{os.path.basename(output_path_applicants_raw)}': {e}")
    data_applicants_json = {}

# ### *3.2. Transformação do JSON de Candidatos em DataFrame*
lista_candidatos = []
if isinstance(data_applicants_json, dict) and data_applicants_json:
    for candidato_id, detalhes_candidato in data_applicants_json.items():
        candidato_info = {"id_candidato": candidato_id}
        if isinstance(detalhes_candidato, dict):
            candidato_info.update(detalhes_candidato.get("infos_basicas", {}))
            candidato_info.update(detalhes_candidato.get("informacoes_pessoais", {}))
            candidato_info.update(detalhes_candidato.get("informacoes_profissionais", {}))
            candidato_info.update(detalhes_candidato.get("formacao_e_idiomas", {}))
            experiencias = detalhes_candidato.get("experiencia_profissional", [])
            if isinstance(experiencias, list) and experiencias:
                descricoes_exp = [exp.get("descricao_atividades", "") for exp in experiencias if isinstance(exp, dict) and exp.get("descricao_atividades")]
                titulos_exp = [exp.get("titulo_cargo", "") for exp in experiencias if isinstance(exp, dict) and exp.get("titulo_cargo")]
                candidato_info["experiencia_descricoes_concatenadas"] = " ".join(filter(None, descricoes_exp))
                candidato_info["experiencia_titulos_concatenados"] = " ".join(filter(None, titulos_exp))
            else:
                candidato_info["experiencia_descricoes_concatenadas"] = ""
                candidato_info["experiencia_titulos_concatenados"] = ""
        lista_candidatos.append(candidato_info)
else:
    print("JSON de candidatos está vazio ou não é um dicionário no nível raiz. 'lista_candidatos' estará vazia.")
df_candidatos = pd.DataFrame(lista_candidatos)
if not df_candidatos.empty:
    print(f"DataFrame de candidatos (df_candidatos) criado com {df_candidatos.shape[0]} linhas e {df_candidatos.shape[1]} colunas.")
else:
    print("DataFrame de candidatos (df_candidatos) está vazio.")


# ### *3.3. Seleção de campos de interesse para candidatos*
campos_selecionados_candidatos = [
    'id_candidato', 'nome', 'email', 'local', 'pcd',
    'titulo_profissional', 'area_atuacao', 'objetivo_profissional',
    'conhecimentos_tecnicos', 'certificacoes', 'outras_certificacoes', 'qualificacoes',
    'nivel_profissional', 'nivel_academico', # Note: nivel_profissional aqui é o original do candidato
    'nivel_ingles', 'nivel_espanhol', 'outro_idioma',
    'experiencia_descricoes_concatenadas', 'experiencia_titulos_concatenados'
]
df_candidatos_processado = pd.DataFrame()
if 'df_candidatos' in globals() and not df_candidatos.empty:
    for col_cand in campos_selecionados_candidatos:
        if col_cand in df_candidatos.columns:
            df_candidatos_processado[col_cand] = df_candidatos[col_cand]
        else:
            print(f"Aviso (Candidatos): Coluna '{col_cand}' não encontrada no DataFrame original de candidatos. Será criada com NA.")
            df_candidatos_processado[col_cand] = pd.NA # Usar pd.NA para tipos mistos
else:
    print("DataFrame original de candidatos está vazio ou não definido. df_candidatos_processado será inicializado com colunas NA.")
    for col_cand in campos_selecionados_candidatos:
        df_candidatos_processado[col_cand] = pd.NA

# ### *3.4. Limpeza dos dados dos candidatos*
if not df_candidatos_processado.empty and df_candidatos_processado.shape[1] > 0:
    cols_para_limpeza_profunda_cand = [
        'local', 'titulo_profissional', 'area_atuacao', 'objetivo_profissional',
        'conhecimentos_tecnicos', 'certificacoes', 'outras_certificacoes', 'qualificacoes',
        'nivel_profissional', 'nivel_academico', 'nivel_ingles', 'nivel_espanhol', 'outro_idioma', 
        'experiencia_descricoes_concatenadas', 'experiencia_titulos_concatenados'
    ]
    for col in df_candidatos_processado.columns:

        df_candidatos_processado[col] = df_candidatos_processado[col].fillna("Não Informado").astype(str)
        df_candidatos_processado[col] = df_candidatos_processado[col].str.strip()

    placeholders_lower_cand = ['', 'nan', 'none', 'null', 'na', '<na>', 'undefined', 'nil', '-', '[]', '{}', 'não informado']
    for col in cols_para_limpeza_profunda_cand:
        if col in df_candidatos_processado.columns:
            df_candidatos_processado[col] = df_candidatos_processado[col].str.lower()
            df_candidatos_processado[col] = df_candidatos_processado[col].replace(placeholders_lower_cand, "Não Informado")
        else: 
             df_candidatos_processado[col] = "Não Informado"


    cols_case_preserved_cand = ['nome', 'email', 'pcd'] 
    for col_special_cand in cols_case_preserved_cand:
        if col_special_cand in df_candidatos_processado.columns:
             df_candidatos_processado[col_special_cand] = df_candidatos_processado[col_special_cand].replace(placeholders_lower_cand, "Não Informado", regex=False)
        else:
             df_candidatos_processado[col_special_cand] = "Não Informado"
    print("Limpeza dos campos selecionados dos candidatos concluída.")
else:
    print("df_candidatos_processado está vazio ou sem colunas. Limpeza ignorada.")


# ### *3.5. Engenharia de features para candidatos*
# 3.5.1. Texto Combinado do Candidato para NLP
campos_para_texto_completo_cand = [
    'titulo_profissional', 'objetivo_profissional', 'conhecimentos_tecnicos',
    'certificacoes', 'outras_certificacoes', 'qualificacoes', 'area_atuacao',
    'experiencia_descricoes_concatenadas', 'experiencia_titulos_concatenados'
]
df_candidatos_processado["texto_completo_candidato"] = ""
if not df_candidatos_processado.empty:
    for field_cand in campos_para_texto_completo_cand:
        if field_cand in df_candidatos_processado.columns and not df_candidatos_processado[field_cand].isnull().all():
            
            df_candidatos_processado["texto_completo_candidato"] += df_candidatos_processado[field_cand].apply(lambda x: str(x) + " " if str(x).lower() != "não informado" else "")
    df_candidatos_processado["texto_completo_candidato"] = df_candidatos_processado["texto_completo_candidato"].str.strip().str.lower()
    print("Coluna 'texto_completo_candidato' criada.")
else:
    df_candidatos_processado["texto_completo_candidato"] = ""
    print("df_candidatos_processado vazio, 'texto_completo_candidato' criada como vazia.")

# 3.5.2. Extração de Tecnologias/Habilidades (Features skill_*)
tecnologias_lista_candidatos = [
    'python', 'java', 'javascript', 'c#', '.net', 'sql', 'nosql', 'aws', 'azure', 'gcp',
    'docker', 'kubernetes', 'react', 'angular', 'vue', 'node.js', 'nodejs', 'php', 'ruby',
    'swift', 'kotlin', 'scala', 'sap', 'oracle', 'power bi', 'powerbi', 'tableau', 'excel', 'git',
    'typescript', 'api', 'rest', 'spring', 'django', 'flask', 'linux', 'html', 'css',
    'salesforce', 'jira', 'trello', 'agile', 'scrum',
    'c++', 'c', 'flutter', 'airflow', 'etl', 'hadoop', 'spark', 'machine learning', 'tensorflow',
    'pytorch', 'devops', 'selenium', 'testing', ' segurança', 'security', 'bi',
    'erp', 'crm', 'office', 'project management', 'gestão de projetos'
]
if "texto_completo_candidato" in df_candidatos_processado.columns and not df_candidatos_processado.empty:
    print("Iniciando extração de tecnologias/habilidades para candidatos...")
    processed_skill_clean_names = set()
    for tech_original_name_cand in tecnologias_lista_candidatos:
        tech_clean_name_cand = ""
        if tech_original_name_cand == 'c++': tech_clean_name_cand = 'cpp'
        elif tech_original_name_cand == 'c': tech_clean_name_cand = 'c_lang'
        elif tech_original_name_cand == 'gestão de projetos': tech_clean_name_cand = 'gestao_de_projetos' 
        else: tech_clean_name_cand = re.sub(r'[^a-zA-Z0-9_]', '', tech_original_name_cand.replace(' ', '_').replace('.', 'dot').replace('+', 'plus').replace('#','sharp'))

        if not tech_clean_name_cand: continue 
        if tech_clean_name_cand in processed_skill_clean_names: continue
        processed_skill_clean_names.add(tech_clean_name_cand)

        col_name_cand = f"skill_{tech_clean_name_cand}"
        pattern_cand = ''
        if tech_original_name_cand == '.net': pattern_cand = r'\b\.net\b'
        elif tech_original_name_cand == 'gestão de projetos': pattern_cand = r'\bgest(?:ão|ao)\s*de\s*projetos\b|\bgerenciamento\s*de\s*projetos\b'
        elif len(tech_original_name_cand) <= 3 and not tech_original_name_cand in ['sap', 'aws', 'gcp', 'api', 'sql', 'css', 'git', 'etl', 'bi', 'erp', 'crm']:
             pattern_cand = r'(?:^|\s|,|\(|\)|-|/)' + re.escape(tech_original_name_cand) + r'(?:$|\s|,|\.|\(|\)|-|/)'
        else: pattern_cand = r'\b' + re.escape(tech_original_name_cand) + r'\b'

        if pattern_cand:
            current_col_values_cand = df_candidatos_processado["texto_completo_candidato"].apply(lambda x: 1 if isinstance(x, str) and re.search(pattern_cand, x, re.IGNORECASE) else 0)
            if col_name_cand not in df_candidatos_processado.columns:
                df_candidatos_processado[col_name_cand] = current_col_values_cand
            else: 
                df_candidatos_processado[col_name_cand] = df_candidatos_processado[col_name_cand] | current_col_values_cand
        else:
            print(f"Aviso (Candidatos): Padrão regex não definido para a tecnologia: {tech_original_name_cand}")
    print(f"Extração de habilidades/tecnologias para candidatos (skill_*) concluída.")
else:
    print("Coluna 'texto_completo_candidato' não encontrada ou df vazio. Extração de skills ignorada.")

# Consolidar colunas duplicadas (ex: skill_nodejs e skill_node.js)
if 'skill_node.js' in df_candidatos_processado.columns and 'skill_nodejs' in df_candidatos_processado.columns:
    df_candidatos_processado['skill_nodejs'] = df_candidatos_processado['skill_nodejs'] | df_candidatos_processado['skill_node.js']
    df_candidatos_processado.drop(columns=['skill_node.js'], inplace=True, errors='ignore')
if 'skill_power_bi' in df_candidatos_processado.columns and 'skill_powerbi' in df_candidatos_processado.columns: # Se power_bi existir como nome limpo
    df_candidatos_processado['skill_powerbi'] = df_candidatos_processado['skill_powerbi'] | df_candidatos_processado['skill_power_bi']
    df_candidatos_processado.drop(columns=['skill_power_bi'], inplace=True, errors='ignore')


# 3.5.3. Generalização do Título Profissional do Candidato (categoria_profissional)
def generalizar_titulo_profissional_candidato(titulo):
    if not isinstance(titulo, str): titulo = str(titulo)
    titulo_lower = titulo.lower() 
    if titulo_lower == "não informado": return "Outros/Não Especificado"
    if "consultor sap" in titulo_lower or "consultora sap" in titulo_lower or ("sap" in titulo_lower and ("consultant" in titulo_lower or "especialista" in titulo_lower or "consultor(a)" in titulo_lower)): return "Consultoria SAP"
    return "Outros/Não Especificado"

if 'titulo_profissional' in df_candidatos_processado.columns and not df_candidatos_processado.empty:
    df_candidatos_processado["categoria_profissional"] = df_candidatos_processado["titulo_profissional"].apply(generalizar_titulo_profissional_candidato)
    print("Coluna 'categoria_profissional' criada.")
else:
    df_candidatos_processado["categoria_profissional"] = "Não Informado"
    print("Coluna 'titulo_profissional' não encontrada ou df vazio. 'categoria_profissional' definida como 'Não Informado'.")

# 3.5.4. Níveis de Idioma Ordinais do Candidato (usar mapa_nivel_idioma da seção de vagas)
if not df_candidatos_processado.empty:
    if 'nivel_ingles' in df_candidatos_processado.columns:
        df_candidatos_processado["nivel_ingles_ordinal"] = df_candidatos_processado["nivel_ingles"].apply(codificar_idioma)
    else:
        df_candidatos_processado["nivel_ingles_ordinal"] = 0
        print("Aviso (Candidatos): Coluna 'nivel_ingles' não encontrada. 'nivel_ingles_ordinal' definida como 0.")

    if 'nivel_espanhol' in df_candidatos_processado.columns:
        df_candidatos_processado["nivel_espanhol_ordinal"] = df_candidatos_processado["nivel_espanhol"].apply(codificar_idioma)
    else:
        df_candidatos_processado["nivel_espanhol_ordinal"] = 0
        print("Aviso (Candidatos): Coluna 'nivel_espanhol' não encontrada. 'nivel_espanhol_ordinal' definida como 0.")
    print("Colunas de nível de idioma ordinal do candidato criadas/atualizadas.")
else:
    df_candidatos_processado["nivel_ingles_ordinal"] = 0
    df_candidatos_processado["nivel_espanhol_ordinal"] = 0
    print("df_candidatos_processado vazio. Níveis de idioma ordinal definidos como 0.")


# 3.5.5. Nível Acadêmico Padronizado do Candidato
mapa_nivel_academico_candidato = {
    'ensino fundamental': 'Ensino Fundamental', 'médio': 'Ensino Médio', '2º grau': 'Ensino Médio', 'segundo grau': 'Ensino Médio',
    'técnico': 'Ensino Técnico', 'profissionalizante': 'Ensino Técnico',
    'superior incompleto': 'Superior Incompleto', 'cursando superior': 'Superior Incompleto', 'graduação em curso': 'Superior Incompleto', 'superior cursando': 'Superior Incompleto',
    'superior completo': 'Superior Completo', 'graduação': 'Superior Completo', 'bacharelado': 'Superior Completo', 'tecnólogo': 'Superior Completo',
    'pós-graduação - especialização': 'Pós-graduação', 'pós-graduação': 'Pós-graduação', 'especialização': 'Pós-graduação', 'pós graduação completo': 'Pós-graduação', 'pós graduação cursando':'Pós-graduação', 'pós graduação incompleto':'Pós-graduação',
    'mba': 'MBA', 'mestrado': 'Mestrado', 'mestrado completo':'Mestrado', 'mestrado incompleto':'Mestrado', 'mestrado cursando':'Mestrado',
    'doutorado': 'Doutorado', 'phd': 'Doutorado', 'doutorado completo':'Doutorado', 'doutorado incompleto':'Doutorado', 'doutorado cursando':'Doutorado'
}
def padronizar_nivel_academico_candidato(nivel):
    nivel_str = str(nivel).lower() 
    if nivel_str == "não informado": return "Não Informado"
    for key, value in mapa_nivel_academico_candidato.items():
        if key in nivel_str: return value
    return "Outros/Não Especificado"

if 'nivel_academico' in df_candidatos_processado.columns and not df_candidatos_processado.empty:
    df_candidatos_processado["nivel_academico_padronizado"] = df_candidatos_processado["nivel_academico"].apply(padronizar_nivel_academico_candidato)
    print("Coluna 'nivel_academico_padronizado' do candidato criada.")
else:
    df_candidatos_processado["nivel_academico_padronizado"] = "Não Informado"
    print("Coluna 'nivel_academico' do candidato não encontrada ou df vazio. 'nivel_academico_padronizado' definida como 'Não Informado'.")

# 3.5.6. Nível Profissional Padronizado do Candidato
mapa_nivel_profissional_candidato = {
    'estagiário': 'Estagiário/Trainee', 'estágio': 'Estagiário/Trainee', 'trainee': 'Estagiário/Trainee',
    'júnior': 'Júnior', 'jr': 'Júnior',
    'pleno': 'Pleno', 'pl': 'Pleno',
    'sênior': 'Sênior', 'sr': 'Sênior', 'senior': 'Sênior',
    'especialista': 'Especialista',
    'coordenador': 'Liderança/Coordenação', 'supervisor': 'Liderança/Coordenação', 'líder': 'Liderança/Coordenação', 'lider': 'Liderança/Coordenação',
    'gerente': 'Gerência/Diretoria', 'diretor': 'Gerência/Diretoria', 'head': 'Gerência/Diretoria', 'gestor': 'Gerência/Diretoria'
}
def padronizar_nivel_profissional_candidato(nivel):
    nivel_str = str(nivel).lower()
    if nivel_str == "não informado": return "Não Informado"
    for key, value in mapa_nivel_profissional_candidato.items():
        if key in nivel_str: return value
    if 'analista' in nivel_str and not any(k in nivel_str for k in ['júnior', 'pleno', 'sênior', 'jr', 'pl', 'sr', 'junior', 'senior']):
        return "Analista (Nível não especificado)"
    return "Outros/Não Especificado"

if 'nivel_profissional' in df_candidatos_processado.columns and not df_candidatos_processado.empty :
    df_candidatos_processado["nivel_profissional_padronizado"] = df_candidatos_processado["nivel_profissional"].apply(padronizar_nivel_profissional_candidato)
    print("Coluna 'nivel_profissional_padronizado' do candidato criada.")
else:
    df_candidatos_processado["nivel_profissional_padronizado"] = "Não Informado"
    print("Coluna 'nivel_profissional' (original) do candidato não encontrada ou df vazio. 'nivel_profissional_padronizado' definida como 'Não Informado'.")

# 3.5.7. PCD (Pessoa com Deficiência) Padronizado
def limpar_pcd_candidato(valor_pcd):
    valor_lower = str(valor_pcd).lower().strip() 
    if valor_lower in ["sim", "s", "yes", "y", "1", "true"]: return "Sim"
    if valor_lower in ["não", "nao", "n", "no", "0", "false"]: return "Não"
    return "Não Informado" 

if 'pcd' in df_candidatos_processado.columns and not df_candidatos_processado.empty:
    df_candidatos_processado["pcd_padronizado"] = df_candidatos_processado["pcd"].apply(limpar_pcd_candidato)
    print("Coluna 'pcd_padronizado' do candidato criada.")
else:
    df_candidatos_processado["pcd_padronizado"] = "Não Informado"
    print("Coluna 'pcd' do candidato não encontrada ou df vazio. 'pcd_padronizado' definida como 'Não Informado'.")

# 3.5.8. Localização Limpa do Candidato
if 'local' in df_candidatos_processado.columns and not df_candidatos_processado.empty:
    df_candidatos_processado["local_limpo_candidato"] = df_candidatos_processado["local"] # Já foi limpo na Seção 3.4
    print("Coluna 'local_limpo_candidato' do candidato criada/mantida.")
else:
    df_candidatos_processado["local_limpo_candidato"] = "Não Informado"
    print("Coluna 'local' do candidato não encontrada ou df vazio. 'local_limpo_candidato' definida como 'Não Informado'.")


# ### *3.6. Informações e visualização do DataFrame de candidatos processado*
if 'df_candidatos_processado' in globals() and not df_candidatos_processado.empty:
    print("\n--- Informações do DataFrame de Candidatos Processado (df_candidatos_processado) ---")
    df_candidatos_processado.info(verbose=False)
    print("\n--- Amostra do DataFrame de Candidatos Processado ---")
    print(df_candidatos_processado.head(2).to_string())
else:
    print("\nDataFrame df_candidatos_processado não foi criado ou está vazio após o processamento.")

# ---------------------------------------------------------------------------
# ## **4. Processamento de Dados de Prospecções**
# ---------------------------------------------------------------------------
print("\n--- Iniciando Processamento de Dados de Prospecções ---")
# ### *4.1. Download e Carregamento do arquivo prospects.json*
data_prospects_json = {}
try:
    print(f"Baixando prospects.json de ID: {file_id_prospects} para {output_path_prospects_raw}...")
    gdown.download(id=file_id_prospects, output=output_path_prospects_raw, quiet=False, fuzzy=True)
    with open(output_path_prospects_raw, 'r', encoding='utf-8') as f:
        data_prospects_json = json.load(f)
    print(f"Arquivo '{output_path_prospects_raw}' carregado com sucesso.")
except Exception as e:
    print(f"Erro Crítico ao baixar ou carregar '{os.path.basename(output_path_prospects_raw)}': {e}")
    data_prospects_json = {}

# ### *4.2. Transformação do JSON de prospecções em DataFrame*
lista_prospeccoes_flat = []
if isinstance(data_prospects_json, dict) and data_prospects_json:
    for id_vaga_origem_prospect, info_vaga_prospect in data_prospects_json.items():
        if isinstance(info_vaga_prospect, dict):
            titulo_vaga_json_prospect = info_vaga_prospect.get("titulo", "Não Informado")
            modalidade_vaga_json_prospect = info_vaga_prospect.get("modalidade", "Não Informado")
            candidatos_prospectados_prospect = info_vaga_prospect.get("prospects", [])
            if isinstance(candidatos_prospectados_prospect, list):
                for candidato_data_prospect in candidatos_prospectados_prospect:
                    if isinstance(candidato_data_prospect, dict):
                        prospect_flat_info = {
                            "id_vaga_origem": id_vaga_origem_prospect,
                            "titulo_vaga_origem_json": titulo_vaga_json_prospect,
                            "modalidade_vaga_origem_json": modalidade_vaga_json_prospect,
                            "nome_candidato": candidato_data_prospect.get("nome", "Não Informado"),
                            "id_candidato_origem": candidato_data_prospect.get("codigo", "Não Informado"),
                            "situacao_candidato": candidato_data_prospect.get("situacao_candidado", "Não Informado"), 
                            "data_candidatura": candidato_data_prospect.get("data_candidatura", "Não Informado"),
                            "data_ultima_atualizacao": candidato_data_prospect.get("ultima_atualizacao", "Não Informado"),
                            "comentario": candidato_data_prospect.get("comentario", "Não Informado"),
                            "recrutador": candidato_data_prospect.get("recrutador", "Não Informado")
                        }
                        lista_prospeccoes_flat.append(prospect_flat_info)
    df_prospects_processado = pd.DataFrame(lista_prospeccoes_flat)
    if not df_prospects_processado.empty:
        print(f"DataFrame de prospecções (df_prospects_processado) criado com {df_prospects_processado.shape[0]} linhas e {df_prospects_processado.shape[1]} colunas.")
    else:
        print("DataFrame de prospecções está vazio após tentativa de normalização.")
else:
    print("Não foi possível processar 'prospects.json' (vazio ou formato inesperado). DataFrame de prospecções estará vazio.")
    df_prospects_processado = pd.DataFrame() 
    
# ### *4.3. Limpeza e pré-processamento do DataFrame de prospecções*
if 'df_prospects_processado' in globals() and not df_prospects_processado.empty:
    cols_para_limpeza_prospects = [
        "titulo_vaga_origem_json", "modalidade_vaga_origem_json", "nome_candidato",
        "situacao_candidato", "comentario", "recrutador"
    ]
    placeholders_gerais_prospects = ['', 'nan', 'none', 'null', 'na', '<na>', 'undefined', 'nil', '-', '[]', '{}', 'não informado', '<NA>']

    for col_prospect in df_prospects_processado.columns:
        df_prospects_processado[col_prospect] = df_prospects_processado[col_prospect].fillna('').astype(str).str.strip()
        df_prospects_processado[col_prospect] = df_prospects_processado[col_prospect].replace(placeholders_gerais_prospects, "Não Informado")

    for col_prospect_lower in cols_para_limpeza_prospects:
        if col_prospect_lower in df_prospects_processado.columns:
            df_prospects_processado[col_prospect_lower] = df_prospects_processado[col_prospect_lower].str.lower()
            df_prospects_processado[col_prospect_lower] = df_prospects_processado[col_prospect_lower].replace(placeholders_gerais_prospects, "Não Informado") # Repetir para garantir após lower

    date_cols_prospects = ["data_candidatura", "data_ultima_atualizacao"]
    for col_date_prospect in date_cols_prospects:
        if col_date_prospect in df_prospects_processado.columns:
            df_prospects_processado[col_date_prospect + "_dt"] = pd.to_datetime(df_prospects_processado[col_date_prospect], format='%d-%m-%Y', errors='coerce')
        else:
            df_prospects_processado[col_date_prospect + "_dt"] = pd.NaT 
    print("Limpeza e pré-processamento de df_prospects_processado concluída.")
else:
    print("DataFrame de prospecções está vazio ou não definido. Limpeza e Pré-processamento ignorados.")
    if 'df_prospects_processado' not in globals() or not isinstance(df_prospects_processado, pd.DataFrame):
        df_prospects_processado = pd.DataFrame()


# ### *4.4. Engenharia de features para prospecções*
# 4.4.1. Duração da Etapa/Processo (em dias) - duracao_etapa_dias
if 'df_prospects_processado' in globals() and not df_prospects_processado.empty:
    if "data_ultima_atualizacao_dt" in df_prospects_processado.columns and "data_candidatura_dt" in df_prospects_processado.columns:
        df_prospects_processado["data_ultima_atualizacao_dt"] = pd.to_datetime(df_prospects_processado["data_ultima_atualizacao_dt"], errors='coerce')
        df_prospects_processado["data_candidatura_dt"] = pd.to_datetime(df_prospects_processado["data_candidatura_dt"], errors='coerce')

        df_prospects_processado["duracao_etapa_dias"] = (df_prospects_processado["data_ultima_atualizacao_dt"] - df_prospects_processado["data_candidatura_dt"]).dt.days
        df_prospects_processado["duracao_etapa_dias"] = df_prospects_processado["duracao_etapa_dias"].apply(lambda x: x if pd.notna(x) and x >= 0 else np.nan)
        print("Coluna 'duracao_etapa_dias' criada.")
    else:
        df_prospects_processado["duracao_etapa_dias"] = np.nan
        print("Colunas de data '_dt' não encontradas ou inválidas em prospecções para calcular 'duracao_etapa_dias'. Coluna criada com NaN.")
else:
    if 'df_prospects_processado' in globals() and isinstance(df_prospects_processado, pd.DataFrame): # Se o df existe mas está vazio
         df_prospects_processado["duracao_etapa_dias"] = np.nan


# 4.4.2. Padronização da Modalidade da Vaga (em Prospecções)
def padronizar_modalidade_prospect(modalidade_texto):
    if not isinstance(modalidade_texto, str): modalidade_texto = str(modalidade_texto)
    modalidade_texto = modalidade_texto.lower() 
    if modalidade_texto == "não informado" or modalidade_texto == "nan": return "Não Informado"
    if re.search(r"h[íi]brido|hibrida", modalidade_texto): return "Híbrido"
    if re.search(r"remoto|remota|home|100% remoto", modalidade_texto): return "Remoto"
    if re.search(r"presencial", modalidade_texto): return "Presencial"
    return "Outros/Não Especificado"

if 'df_prospects_processado' in globals() and not df_prospects_processado.empty:
    if 'modalidade_vaga_origem_json' in df_prospects_processado.columns:
        df_prospects_processado["modalidade_vaga_padronizada_prospect"] = df_prospects_processado["modalidade_vaga_origem_json"].apply(padronizar_modalidade_prospect)
        print("Coluna 'modalidade_vaga_padronizada_prospect' criada.")
    else:
        df_prospects_processado["modalidade_vaga_padronizada_prospect"] = "Não Informado"
        print("Coluna 'modalidade_vaga_origem_json' não encontrada em prospecções. 'modalidade_vaga_padronizada_prospect' definida como 'Não Informado'.")
else:
    if 'df_prospects_processado' in globals() and isinstance(df_prospects_processado, pd.DataFrame):
        df_prospects_processado["modalidade_vaga_padronizada_prospect"] = "Não Informado"

# 4.4.3. Agrupamento da Situação do Candidato (situacao_candidato_agrupada)
mapa_situacao_agrupada = {
    "prospect": "Em Processo", "encaminhado ao requisitante": "Em Processo", "contato inicial": "Em Processo",
    "entrevista rh": "Em Processo", "entrevista gestor": "Em Processo", "teste técnico": "Em Processo",
    "em processo": "Em Processo", "em avaliação": "Em Processo", "agendado": "Em Processo", "selecionado para entrevista": "Em Processo",
    "inscrito": "Em Processo",
    "aprovado": "Finalizado - Contratado", "contratado": "Finalizado - Contratado", "proposta aceita": "Finalizado - Contratado",
    "contratado pela decision": "Finalizado - Contratado", "contratado como hunting": "Finalizado - Contratado",
    "oferta enviada": "Finalizado - Oferta", "proposta enviada": "Finalizado - Oferta", "encaminhar proposta": "Finalizado - Oferta",
    "reprovado": "Finalizado - Rejeitado", "não aprovado": "Finalizado - Rejeitado", "desclassificado": "Finalizado - Rejeitado",
    "perfil não aderente": "Finalizado - Rejeitado", "fora do perfil": "Finalizado - Rejeitado",
    "não evoluiu": "Finalizado - Rejeitado", "não selecionado": "Finalizado - Rejeitado",
    "não aprovado pelo cliente": "Finalizado - Rejeitado", "não aprovado pelo rh": "Finalizado - Rejeitado", "não aprovado pelo requisitante": "Finalizado - Rejeitado",
    "recusado": "Finalizado - Rejeitado",
    "desistiu": "Desistiu/Standby", "stand by": "Desistiu/Standby", "standby": "Desistiu/Standby",
    "não tem interesse": "Desistiu/Standby", "sem interesse": "Desistiu/Standby", "sem interesse nesta vaga": "Desistiu/Standby",
    "desistiu da contratação": "Desistiu/Standby",
    "congelado": "Vaga Congelada/Cancelada", "pausado": "Vaga Congelada/Cancelada", "cancelado":"Vaga Congelada/Cancelada",
    "documentação": "Outros/Não Classificado"
}
def agrupar_situacao_candidato(situacao):
    if not isinstance(situacao, str): situacao = str(situacao)
    situacao_lower = situacao.lower() 
    if situacao_lower == "não informado" or situacao_lower == "nan": return "Não Informado"

    if "contratado pela decision" in situacao_lower: return "Finalizado - Contratado"
    if "contratado como hunting" in situacao_lower: return "Finalizado - Contratado"
    if "proposta aceita" in situacao_lower: return "Finalizado - Contratado"
    if "aprovado" in situacao_lower and not "não aprovado" in situacao_lower : return "Finalizado - Contratado"
    if "não aprovado pelo cliente" in situacao_lower: return "Finalizado - Rejeitado"
    for key, value in mapa_situacao_agrupada.items():
        if key in situacao_lower:
            return value
    return "Outros/Não Classificado"

if 'df_prospects_processado' in globals() and not df_prospects_processado.empty:
    if 'situacao_candidato' in df_prospects_processado.columns:
        df_prospects_processado["situacao_candidato_agrupada"] = df_prospects_processado["situacao_candidato"].apply(agrupar_situacao_candidato)
        print("Coluna 'situacao_candidato_agrupada' criada/atualizada.")
    else:
        df_prospects_processado["situacao_candidato_agrupada"] = "Não Informado"
        print("Coluna 'situacao_candidato' não encontrada em prospecções. 'situacao_candidato_agrupada' definida como 'Não Informado'.")
else:
     if 'df_prospects_processado' in globals() and isinstance(df_prospects_processado, pd.DataFrame):
        df_prospects_processado["situacao_candidato_agrupada"] = "Não Informado"


# 4.4.4. Análise de Comentário por Valor Monetário
if 'df_prospects_processado' in globals() and not df_prospects_processado.empty:
    if 'comentario' in df_prospects_processado.columns:
        df_prospects_processado['comentario_tem_valor_monetario'] = df_prospects_processado['comentario'].apply(
            lambda x: 1 if isinstance(x, str) and (re.search(r'r\$|\bsal[áa]rio\b|\bremunera[çc][ãa]o\b|pretens[ãa]o', x, re.IGNORECASE)) else 0
        )
        print("Coluna 'comentario_tem_valor_monetario' criada.")
    else:
        df_prospects_processado['comentario_tem_valor_monetario'] = 0
        print("Coluna 'comentario' não encontrada em prospecções. 'comentario_tem_valor_monetario' definida como 0.")
else:
    if 'df_prospects_processado' in globals() and isinstance(df_prospects_processado, pd.DataFrame):
        df_prospects_processado['comentario_tem_valor_monetario'] = 0

print("Engenharia de features para df_prospects_processado concluída.")

# ### *4.5. Informações e visualização do DataFrame de prospecções processado*
if 'df_prospects_processado' in globals() and not df_prospects_processado.empty:
    print("\n--- Informações do DataFrame de Prospecções Processado (df_prospects_processado) ---")
    df_prospects_processado.info(verbose=False)
    print("\n--- Amostra do DataFrame de Prospecções Processado ---")
    print(df_prospects_processado.head(2).to_string())
else:
    print("\nDataFrame df_prospects_processado não foi criado ou está vazio.")


# ---------------------------------------------------------------------------
# ## **5. Merge dos DataFrames Processados**
# ---------------------------------------------------------------------------
print("\n--- Iniciando Merge dos DataFrames ---")
if 'df_prospects_processado' not in globals() or df_prospects_processado.empty:
    print("df_prospects_processado está vazio ou não definido. Merge não pode continuar sem prospecções.")
    df_master = pd.DataFrame() # Define df_master como vazio para o script não quebrar depois
else:
    if 'df_vagas_processado' not in globals() or df_vagas_processado.empty:
        print("Aviso: df_vagas_processado está vazio ou não definido. Merge com vagas será incompleto.")
        df_vagas_processado_para_merge = pd.DataFrame(columns=['id_vaga'] + [col for col in df_vagas_processado.columns if col != 'id_vaga'])
    else:
        df_vagas_processado_para_merge = df_vagas_processado

    if 'df_candidatos_processado' not in globals() or df_candidatos_processado.empty:
        print("Aviso: df_candidatos_processado está vazio ou não definido. Merge com candidatos será incompleto.")
        df_candidatos_processado_para_merge = pd.DataFrame(columns=['id_candidato'] + [col for col in df_candidatos_processado.columns if col != 'id_candidato'])
    else:
        df_candidatos_processado_para_merge = df_candidatos_processado

    print(f"Linhas em df_prospects_processado: {len(df_prospects_processado)}")
    print(f"Linhas em df_vagas_processado_para_merge: {len(df_vagas_processado_para_merge)}")

    # ### *5.1. Merge: Prospects + Vagas*
    df_combinado_temp = pd.merge(
        df_prospects_processado,
        df_vagas_processado_para_merge,
        left_on='id_vaga_origem',
        right_on='id_vaga', 
        how='left',
        suffixes=('_prospect', '_vaga_merged') 
    )
    print(f"Linhas após merge com vagas: {len(df_combinado_temp)}")

    if 'id_vaga' in df_combinado_temp.columns and 'id_vaga_origem' in df_combinado_temp.columns and 'id_vaga' != 'id_vaga_origem':
        df_combinado_temp = df_combinado_temp.drop(columns=['id_vaga'])
        print("Coluna 'id_vaga' (duplicada do merge com vagas) removida.")

    # ### *5.2. Merge: (Prospects + Vagas) + Candidatos*
    print(f"Linhas em df_candidatos_processado_para_merge: {len(df_candidatos_processado_para_merge)}")
    df_master = pd.merge(
        df_combinado_temp,
        df_candidatos_processado_para_merge,
        left_on='id_candidato_origem',
        right_on='id_candidato', 
        how='left',
        suffixes=('_vaga', '_candidato')
    )
    print(f"Linhas após merge com candidatos (final df_master): {len(df_master)}")

    if 'id_candidato_candidato' in df_master.columns: # Se o sufixo _candidato foi adicionado a id_candidato
        df_master = df_master.drop(columns=['id_candidato_candidato'])
        print("Coluna 'id_candidato_candidato' (duplicada do merge com candidatos) removida.")
    elif 'id_candidato_y' in df_master.columns: # Outro sufixo comum do Pandas
        df_master = df_master.drop(columns=['id_candidato_y'])
        if 'id_candidato_x' in df_master.columns:
            df_master.rename(columns={'id_candidato_x': 'id_candidato'}, inplace=True)
        print("Coluna 'id_candidato_y' removida e 'id_candidato_x' renomeada para 'id_candidato'.")
    elif 'id_candidato' in df_master.columns and 'id_candidato_origem' in df_master.columns and df_master['id_candidato'].equals(df_master['id_candidato_origem']):
        pass


    print("\nMerge concluído! DataFrame 'df_master' criado.")
    print(f"Número de linhas em df_master: {len(df_master)}")
    print(f"Número de colunas em df_master: {len(df_master.columns)}")

# ### *5.3. Informações e amostra do df_master*
if 'df_master' in globals() and not df_master.empty:
    print("\n--- Informações do df_master ---")
    df_master.info(verbose=True, show_counts=True, max_cols=200) 
    print("\n--- Amostra do df_master ---")
    print(df_master.head().to_string())
else:
    print("\ndf_master está vazio ou não foi criado. Verifique as etapas de merge e carregamento de dados.")

# ---------------------------------------------------------------------------
# ## **6. Preparação final dos dados para modelagem**
# ---------------------------------------------------------------------------
df_limpo = pd.DataFrame() # Inicializar
if 'df_master' in globals() and not df_master.empty:
    print("\n--- Iniciando Preparação Final para Modelagem ---")
    # ### *6.1. Limpeza de linhas com merges incompletos*
    print(f"\nLinhas em df_master antes da limpeza de nulos do merge: {len(df_master)}")

    coluna_chave_vaga_check = 'titulo_vaga' 
    coluna_chave_candidato_check = 'texto_completo_candidato'
    if coluna_chave_vaga_check + "_vaga" in df_master.columns: 
        coluna_chave_vaga_check = coluna_chave_vaga_check + "_vaga"
    elif coluna_chave_vaga_check + "_vaga_merged" in df_master.columns:
        coluna_chave_vaga_check = coluna_chave_vaga_check + "_vaga_merged"

    if coluna_chave_candidato_check + "_candidato" in df_master.columns:
         coluna_chave_candidato_check = coluna_chave_candidato_check + "_candidato"

    colunas_para_dropna_merge = []
    if coluna_chave_vaga_check in df_master.columns:
        colunas_para_dropna_merge.append(coluna_chave_vaga_check)
    else:
        print(f"Aviso: A coluna chave para vagas '{coluna_chave_vaga_check}' (ou com sufixo) não foi encontrada em df_master.")

    if coluna_chave_candidato_check in df_master.columns:
        colunas_para_dropna_merge.append(coluna_chave_candidato_check)
    else:
        print(f"Aviso: A coluna chave para candidatos '{coluna_chave_candidato_check}' (ou com sufixo) não foi encontrada em df_master.")

    df_limpo = df_master.copy() 
    if colunas_para_dropna_merge:
        for col_check in colunas_para_dropna_merge:
            df_limpo = df_limpo[df_limpo[col_check].notna()]

        df_limpo.dropna(subset=colunas_para_dropna_merge, inplace=True) 
        print(f"Linhas em df_limpo após remover nulos de merge usando {colunas_para_dropna_merge}: {len(df_limpo)} linhas.")
    else:
        print("Aviso: Nenhuma coluna chave identificada para dropna de merges incompletos. df_limpo será uma cópia de df_master.")


    # ### *6.2. Criação da variável alvo foi_contratado*
    situacoes_sucesso_contratado = [
        'contratado pela decision', 'contratado como hunting', 'aprovado', 'proposta aceita'
    ]
    situacoes_nao_sucesso_contratado = [ 
        'não aprovado pelo cliente', 'não aprovado pelo rh', 'não aprovado pelo requisitante',
        'desistiu', 'desistiu da contratação', 'sem interesse nesta vaga', 'recusado',
        'reprovado', 'não aprovado', 'desclassificado', 'perfil não aderente', 'fora do perfil',
        'não evoluiu', 'não selecionado'
    ]

    if 'situacao_candidato' in df_limpo.columns: 
        df_limpo.loc[:, 'foi_contratado_temp'] = -1 

        cond_sucesso = df_limpo['situacao_candidato'].str.lower().isin([s.lower() for s in situacoes_sucesso_contratado])
        cond_nao_sucesso = df_limpo['situacao_candidato'].str.lower().isin([s.lower() for s in situacoes_nao_sucesso_contratado])

        df_limpo.loc[cond_sucesso, 'foi_contratado_temp'] = 1
        df_limpo.loc[cond_nao_sucesso, 'foi_contratado_temp'] = 0

        df_modelagem = df_limpo[df_limpo['foi_contratado_temp'] != -1].copy()
        df_modelagem.rename(columns={'foi_contratado_temp': 'foi_contratado'}, inplace=True)

        print(f"Coluna 'foi_contratado' criada e df_modelagem filtrado.")
        print(f"Linhas no df_modelagem final: {len(df_modelagem)}")
    else:
        print("Erro: Coluna 'situacao_candidato' não encontrada em df_limpo. Variável alvo não pôde ser criada.")
        df_modelagem = df_limpo.copy() 
        if not df_modelagem.empty:
             df_modelagem['foi_contratado'] = -1 
else:
    print("df_master está vazio. Preparação para modelagem não pode continuar.")
    df_modelagem = pd.DataFrame() 

# ### *6.3. Verificação do df_modelagem*
if 'df_modelagem' in globals() and not df_modelagem.empty and 'foi_contratado' in df_modelagem.columns:
    if (df_modelagem['foi_contratado'] != -1).any(): 
        print("\n--- Informações do df_modelagem ---")
        df_modelagem.info(verbose=True, show_counts=True, max_cols=200)
        print("\n--- Distribuição da variável alvo 'foi_contratado' no df_modelagem ---")
        print(df_modelagem['foi_contratado'].value_counts(normalize=True))
        print("\nContagem absoluta:")
        print(df_modelagem['foi_contratado'].value_counts())
        print("\n--- Amostra do df_modelagem (colunas selecionadas) ---")

        colunas_para_verificar_modelagem = ['id_vaga_origem', 'id_candidato_origem', 'situacao_candidato', 'situacao_candidato_agrupada', 'foi_contratado']
        colunas_existentes_para_verificar_modelagem = [col for col in colunas_para_verificar_modelagem if col in df_modelagem.columns]

        if colunas_existentes_para_verificar_modelagem:
            print(df_modelagem[colunas_existentes_para_verificar_modelagem].head(10).to_string())
        else:
            print("Nenhuma das colunas de verificação ('id_vaga_origem', 'id_candidato_origem', etc.) foi encontrada em df_modelagem.")
    else:
        print("df_modelagem não contém entradas válidas para 'foi_contratado' (todas são -1 ou coluna não existe). Verifique as etapas anteriores.")
else:
    print("df_modelagem está vazio ou não foi criado corretamente. Verifique as etapas de merge e criação do alvo.")

# ---------------------------------------------------------------------------
# ## **7. Análise Exploratória de Dados (EDA) no df_modelagem**
# ---------------------------------------------------------------------------
if 'df_modelagem' in globals() and not df_modelagem.empty:
    print("\n--- Iniciando Recálculo de Features de EDA no df_modelagem ---")
    # ### *7.1. Recálculo de features de EDA no df_modelagem*
    # Função auxiliar para obter nome de coluna com possíveis sufixos
    def obter_nome_coluna_eda(df, nome_base, sufixos_possiveis=['_vaga', '_candidato', '_vaga_merged', '']): 
        for sufixo in sufixos_possiveis:
            nome_col_testado = nome_base + sufixo
            if nome_col_testado in df.columns:
                return nome_col_testado
        # Se nenhum sufixo funcionar, tentar o nome base sem sufixo se ele existir
        if nome_base in df.columns:
            return nome_base
        print(f"Aviso (EDA): Coluna base '{nome_base}' (ou com sufixos comuns) não encontrada no DataFrame.")
        return None

    df_modelagem.name = 'df_modelagem' 

    col_nivel_ingles_cand_nome = obter_nome_coluna_eda(df_modelagem, 'nivel_ingles_ordinal', ['_candidato', ''])
    col_nivel_ingles_vaga_nome = obter_nome_coluna_eda(df_modelagem, 'nivel_ingles_ordinal', ['_vaga', '_vaga_merged', ''])

    if col_nivel_ingles_cand_nome and col_nivel_ingles_vaga_nome:
        # Assegurar que são numéricos antes de comparar, convertendo não numéricos para 0 (ou NaN)
        df_modelagem[col_nivel_ingles_cand_nome] = pd.to_numeric(df_modelagem[col_nivel_ingles_cand_nome], errors='coerce').fillna(0)
        df_modelagem[col_nivel_ingles_vaga_nome] = pd.to_numeric(df_modelagem[col_nivel_ingles_vaga_nome], errors='coerce').fillna(0)
        df_modelagem['compat_ingles'] = (df_modelagem[col_nivel_ingles_cand_nome] >= df_modelagem[col_nivel_ingles_vaga_nome]).astype(int)
    else:
        print("Aviso (EDA): Colunas para 'compat_ingles' não encontradas. Preenchendo com 0.")
        df_modelagem['compat_ingles'] = 0

    col_nivel_espanhol_cand_nome = obter_nome_coluna_eda(df_modelagem, 'nivel_espanhol_ordinal', ['_candidato', ''])
    col_nivel_espanhol_vaga_nome = obter_nome_coluna_eda(df_modelagem, 'nivel_espanhol_ordinal', ['_vaga', '_vaga_merged', ''])
    if col_nivel_espanhol_cand_nome and col_nivel_espanhol_vaga_nome:
        df_modelagem[col_nivel_espanhol_cand_nome] = pd.to_numeric(df_modelagem[col_nivel_espanhol_cand_nome], errors='coerce').fillna(0)
        df_modelagem[col_nivel_espanhol_vaga_nome] = pd.to_numeric(df_modelagem[col_nivel_espanhol_vaga_nome], errors='coerce').fillna(0)
        df_modelagem['compat_espanhol'] = (df_modelagem[col_nivel_espanhol_cand_nome] >= df_modelagem[col_nivel_espanhol_vaga_nome]).astype(int)
    else:
        print("Aviso (EDA): Colunas para 'compat_espanhol' não encontradas. Preenchendo com 0.")
        df_modelagem['compat_espanhol'] = 0

    tech_cols_vaga_eda = [col for col in df_modelagem.columns if col.startswith('tech_') and not col.endswith('_candidato')]
    skill_cols_candidato_eda = [col for col in df_modelagem.columns if col.startswith('skill_')]

    df_modelagem.loc[:, 'total_techs_vaga'] = 0
    if tech_cols_vaga_eda:
        for tc in tech_cols_vaga_eda:
            df_modelagem[tc] = pd.to_numeric(df_modelagem[tc], errors='coerce').fillna(0)
        df_modelagem.loc[:, 'total_techs_vaga'] = df_modelagem[tech_cols_vaga_eda].sum(axis=1)
    else:
        print("Aviso (EDA): Nenhuma coluna 'tech_*' (vaga) encontrada para calcular 'total_techs_vaga'.")

    df_modelagem.loc[:, 'skills_match_count'] = 0
    if tech_cols_vaga_eda and skill_cols_candidato_eda:
        mapa_nomes_skills_candidato_eda = {}
        for s_col_eda in skill_cols_candidato_eda:
            df_modelagem[s_col_eda] = pd.to_numeric(df_modelagem[s_col_eda], errors='coerce').fillna(0)
            nome_base_skill_eda = s_col_eda.replace('skill_', '').strip('_')
            mapa_nomes_skills_candidato_eda[nome_base_skill_eda] = s_col_eda

        for t_col_eda in tech_cols_vaga_eda: 
            nome_base_tech_eda = t_col_eda.replace('tech_', '').strip('_')
            if nome_base_tech_eda in mapa_nomes_skills_candidato_eda:
                s_col_correspondente_eda = mapa_nomes_skills_candidato_eda[nome_base_tech_eda]
                df_modelagem.loc[:, 'skills_match_count'] += (df_modelagem[s_col_correspondente_eda].astype(int) * df_modelagem[t_col_eda].astype(int))
    else:
         print("Aviso (EDA): Colunas 'tech_*' (vaga) ou 'skill_*' (candidato) não encontradas para 'skills_match_count'.")

    df_modelagem.loc[:, 'skills_faltantes_vaga'] = df_modelagem['total_techs_vaga'] - df_modelagem['skills_match_count']
    df_modelagem.loc[:, 'skills_faltantes_vaga'] = df_modelagem['skills_faltantes_vaga'].clip(lower=0)
    print("Features de EDA ('compat_ingles', 'compat_espanhol', 'skills_faltantes_vaga', 'total_techs_vaga', 'skills_match_count') recalculadas/garantidas em df_modelagem.")
else:
    print("df_modelagem está vazio. EDA ignorada.")


# ---------------------------------------------------------------------------
# ## **8. Preparação das features para modelagem**
# ---------------------------------------------------------------------------
if 'df_modelagem' in globals() and not df_modelagem.empty and 'foi_contratado' in df_modelagem.columns and (df_modelagem['foi_contratado'] != -1).any():
    print("\n--- Iniciando Preparação de Features para Modelagem ---")

    # Usar a mesma função obter_nome_coluna_eda para consistência
    # Features numéricas diretas (já existem ou são criadas e precisam ser selecionadas)
    features_numericas_selecionadas_base = []

    nivel_ingles_vaga_col = obter_nome_coluna_eda(df_modelagem, 'nivel_ingles_ordinal', ['_vaga', '_vaga_merged', ''])
    if nivel_ingles_vaga_col: features_numericas_selecionadas_base.append(nivel_ingles_vaga_col)
    nivel_espanhol_vaga_col = obter_nome_coluna_eda(df_modelagem, 'nivel_espanhol_ordinal', ['_vaga', '_vaga_merged', ''])
    if nivel_espanhol_vaga_col: features_numericas_selecionadas_base.append(nivel_espanhol_vaga_col)
    vaga_sap_bool_col = obter_nome_coluna_eda(df_modelagem, 'vaga_sap_bool', ['_vaga', '_vaga_merged', '']) 
    if vaga_sap_bool_col is None and 'vaga_sap_bool' in df_modelagem.columns: 
        vaga_sap_bool_col = 'vaga_sap_bool'
    if vaga_sap_bool_col and vaga_sap_bool_col in df_modelagem.columns:
         features_numericas_selecionadas_base.append(vaga_sap_bool_col)
    nivel_ingles_cand_col = obter_nome_coluna_eda(df_modelagem, 'nivel_ingles_ordinal', ['_candidato', ''])
    if nivel_ingles_cand_col: features_numericas_selecionadas_base.append(nivel_ingles_cand_col)
    nivel_espanhol_cand_col = obter_nome_coluna_eda(df_modelagem, 'nivel_espanhol_ordinal', ['_candidato', ''])
    if nivel_espanhol_cand_col: features_numericas_selecionadas_base.append(nivel_espanhol_cand_col)

    features_numericas_ja_criadas_eda = ['compat_ingles', 'compat_espanhol', 'skills_faltantes_vaga', 'total_techs_vaga', 'skills_match_count']
    for fn_criada_eda in features_numericas_ja_criadas_eda:
        if fn_criada_eda in df_modelagem.columns:
            features_numericas_selecionadas_base.append(fn_criada_eda)

    if 'comentario_tem_valor_monetario' in df_modelagem.columns:
        features_numericas_selecionadas_base.append('comentario_tem_valor_monetario')

    features_numericas_selecionadas_final = sorted(list(set(col for col in features_numericas_selecionadas_base if col in df_modelagem.columns and (df_modelagem[col].dtype == 'int64' or df_modelagem[col].dtype == 'float64' or df_modelagem[col].dtype == 'int32'))))
    print(f"\nFeatures numéricas selecionadas: {features_numericas_selecionadas_final}")

    # Features categóricas para One-Hot Encoding
    features_categoricas_selecionadas_base = []
    # Da VAGA
    for fc_vaga_base in ['nivel_profissional_vaga', 'nivel_academico', 'modalidade_trabalho', 'categoria_vaga', 'vaga_sap']: 
        col_fc_vaga = obter_nome_coluna_eda(df_modelagem, fc_vaga_base, ['_vaga', '_vaga_merged', ''])
        if col_fc_vaga is None and fc_vaga_base in df_modelagem.columns: 
            col_fc_vaga = fc_vaga_base
        if col_fc_vaga and col_fc_vaga in df_modelagem.columns and df_modelagem[col_fc_vaga].dtype == 'object':
            features_categoricas_selecionadas_base.append(col_fc_vaga)
            df_modelagem[col_fc_vaga] = df_modelagem[col_fc_vaga].astype(str).fillna("Não Informado") 

    # Do CANDIDATO
    for fc_cand_base in ['nivel_academico_padronizado', 'categoria_profissional', 'pcd_padronizado', 'nivel_profissional_padronizado']:
        col_fc_cand = obter_nome_coluna_eda(df_modelagem, fc_cand_base, ['_candidato', ''])
        if col_fc_cand and col_fc_cand in df_modelagem.columns and df_modelagem[col_fc_cand].dtype == 'object':
            features_categoricas_selecionadas_base.append(col_fc_cand)
            df_modelagem[col_fc_cand] = df_modelagem[col_fc_cand].astype(str).fillna("Não Informado")


    features_categoricas_selecionadas_final = sorted(list(set(col for col in features_categoricas_selecionadas_base if col in df_modelagem.columns)))
    print(f"Features categóricas selecionadas para One-Hot Encoding: {features_categoricas_selecionadas_final}")
    df_modelagem_encoded = pd.get_dummies(df_modelagem,
                                          columns=features_categoricas_selecionadas_final,
                                          dummy_na=False, 
                                          drop_first=True) 
    print(f"\nNúmero de colunas em df_modelagem_encoded após One-Hot Encoding: {len(df_modelagem_encoded.columns)}")

    colunas_features_finais_modelo = features_numericas_selecionadas_final.copy()
    for col_base_cat in features_categoricas_selecionadas_final:
        cols_from_dummy_cat = [col for col in df_modelagem_encoded.columns if col.startswith(str(col_base_cat) + '_')]
        colunas_features_finais_modelo.extend(cols_from_dummy_cat)

    colunas_features_finais_modelo = sorted(list(set(col for col in colunas_features_finais_modelo if col in df_modelagem_encoded.columns)))

    colunas_a_remover_de_features = ['foi_contratado', 'duracao_etapa_dias', 'id_vaga_origem', 'id_candidato_origem', 'id_vaga', 'id_candidato'] 
    colunas_features_finais_modelo = [col for col in colunas_features_finais_modelo if col not in colunas_a_remover_de_features]

    colunas_tech_vaga_finais = [col for col in df_modelagem_encoded.columns if col.startswith('tech_') and col in df_vagas_processado.columns] 
    colunas_skill_candidato_finais = [col for col in df_modelagem_encoded.columns if col.startswith('skill_') and col in df_candidatos_processado.columns] 
    
    colunas_features_finais_modelo.extend(colunas_tech_vaga_finais)
    colunas_features_finais_modelo.extend(colunas_skill_candidato_finais)
    colunas_features_finais_modelo = sorted(list(set(colunas_features_finais_modelo))) 

    print(f"\nNúmero de features finais para o modelo: {len(colunas_features_finais_modelo)}")


    X = df_modelagem_encoded[colunas_features_finais_modelo]
    y = df_modelagem_encoded['foi_contratado']
    X = X.fillna(0)


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print("\nFormatos dos dataframes de treino e teste:")
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")
    if not y_train.empty:
        print("\nDistribuição do alvo no conjunto de treino:")
        print(y_train.value_counts(normalize=True))
    if not y_test.empty:
        print("\nDistribuição do alvo no conjunto de teste:")
        print(y_test.value_counts(normalize=True))

    print("\nAlgumas colunas de X_train (primeiras 5 linhas, primeiras 5 colunas) para verificação:")
    if not X_train.empty:
        print(X_train.iloc[:5, :min(5, X_train.shape[1])])
    else:
        print("X_train está vazio.")
else:
    print("Modelagem não pode prosseguir: df_modelagem está vazio, sem alvo 'foi_contratado' ou sem alvos válidos.")
    X_train, X_test, y_train, y_test = pd.DataFrame(), pd.DataFrame(), pd.Series(dtype='int'), pd.Series(dtype='int')


# ---------------------------------------------------------------------------
# ## **9. Treinamento e avaliação de modelos baseline**
# ---------------------------------------------------------------------------
if not X_train.empty and not y_train.empty:
    print("\n--- Iniciando Treinamento de Modelos Baseline ---")
    # ### *9.1. Regressão Logística*
    log_reg_model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000, solver='liblinear')
    log_reg_model.fit(X_train, y_train)

    y_pred_lr = log_reg_model.predict(X_test)
    y_pred_proba_lr = log_reg_model.predict_proba(X_test)[:, 1]

    accuracy_lr = accuracy_score(y_test, y_pred_lr)
    roc_auc_lr = roc_auc_score(y_test, y_pred_proba_lr) if len(np.unique(y_test)) > 1 else 0.5
    conf_matrix_lr = confusion_matrix(y_test, y_pred_lr)
    class_report_lr = classification_report(y_test, y_pred_lr, target_names=['Não Contratado (0)', 'Contratado (1)'], zero_division=0)

    print("\n--- Resultados da Regressão Logística ---")
    print(f"Acurácia: {accuracy_lr:.4f}")
    print(f"AUC-ROC: {roc_auc_lr:.4f}")
    print("\nMatriz de Confusão:")
    print(conf_matrix_lr)
    print("\nRelatório de Classificação:")
    print(class_report_lr)

    # ### *9.2. Random Forest Classifier*
    rf_clf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1, min_samples_split=10, min_samples_leaf=5)
    rf_clf_model.fit(X_train, y_train)

    y_pred_rf = rf_clf_model.predict(X_test)
    y_pred_proba_rf = rf_clf_model.predict_proba(X_test)[:, 1]

    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    roc_auc_rf = roc_auc_score(y_test, y_pred_proba_rf) if len(np.unique(y_test)) > 1 else 0.5
    conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
    class_report_rf = classification_report(y_test, y_pred_rf, target_names=['Não Contratado (0)', 'Contratado (1)'], zero_division=0)

    print("\n--- Resultados do Random Forest Classifier ---")
    print(f"Acurácia: {accuracy_rf:.4f}")
    print(f"AUC-ROC: {roc_auc_rf:.4f}")
    print("\nMatriz de Confusão:")
    print(conf_matrix_rf)
    print("\nRelatório de Classificação:")
    print(class_report_rf)

    if hasattr(rf_clf_model, 'feature_importances_') and not X_train.empty:
        feature_importances_rf = pd.Series(rf_clf_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
        print("\nTop 20 Features Mais Importantes (Random Forest Baseline):")
        print(feature_importances_rf.head(20))
    else:
        print("Não foi possível calcular a importância das features para rf_clf_model.")
else:
    print("Treinamento de modelos não pode prosseguir: X_train ou y_train estão vazios.")
    rf_clf_model = None 

# ---------------------------------------------------------------------------
# ## **10. Análise dos resultados e modelo**
# ---------------------------------------------------------------------------
if rf_clf_model is not None and not X_test.empty and not y_test.empty: # Checar se o modelo foi treinado
    print("\n--- Iniciando Análise de Resultados e Otimização ---")
    # ### *10.1. Ajuste do Threshold de Decisão do Random Forest*
    y_pred_proba_rf_classe1_test = rf_clf_model.predict_proba(X_test)[:, 1]
    print("\n--- Ajuste de Threshold para Random Forest ---")
    thresholds_para_testar_rf = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
    resultados_threshold_rf = []
    for thr_rf in thresholds_para_testar_rf:
        y_pred_rf_novo_thr = (y_pred_proba_rf_classe1_test >= thr_rf).astype(int)
        precision_rf, recall_rf, f1_rf, _ = precision_recall_fscore_support(y_test, y_pred_rf_novo_thr, labels=[0,1], zero_division=0)
        conf_matrix_thr_rf = confusion_matrix(y_test, y_pred_rf_novo_thr)
        accuracy_geral_thr_rf = accuracy_score(y_test, y_pred_rf_novo_thr)
        resultados_threshold_rf.append({
            'threshold': thr_rf, 'accuracy': accuracy_geral_thr_rf,
            'recall_classe1': recall_rf[1], 'precision_classe1': precision_rf[1], 'f1_classe1': f1_rf[1],
            'TN': conf_matrix_thr_rf[0,0], 'FP': conf_matrix_thr_rf[0,1],
            'FN': conf_matrix_thr_rf[1,0], 'TP': conf_matrix_thr_rf[1,1]
        })
    df_resultados_threshold_rf = pd.DataFrame(resultados_threshold_rf)
    print("\n--- Resumo dos Resultados por Threshold (Random Forest Baseline) ---")
    print(df_resultados_threshold_rf[['threshold', 'accuracy', 'recall_classe1', 'precision_classe1', 'f1_classe1', 'TP', 'FN']])


    # ### *10.2. Otimização de hiperparâmetros do Random Forest*
    param_dist_rf_opt = {
        'n_estimators': randint(100, 400), 'max_depth': [None, 10, 20, 30],
        'min_samples_split': randint(5, 30), 'min_samples_leaf': randint(2, 15),
        'class_weight': ['balanced', 'balanced_subsample'], 'max_features': ['sqrt', 'log2', 0.4, None]
    }
    scorer_f1_classe1_opt = make_scorer(f1_score, pos_label=1, zero_division=0)
    random_search_rf_opt = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
        param_distributions=param_dist_rf_opt, n_iter=15, cv=3, 
        scoring=scorer_f1_classe1_opt, random_state=42, n_jobs=-1, verbose=1
    )
    print("\nIniciando RandomizedSearchCV para Random Forest...")
    random_search_rf_opt.fit(X_train, y_train)

    print("\nMelhores Hiperparâmetros encontrados para Random Forest:")
    print(random_search_rf_opt.best_params_)
    print(f"\nMelhor F1-score (classe 1) na validação cruzada: {random_search_rf_opt.best_score_:.4f}")

    best_rf_clf_otimizado = random_search_rf_opt.best_estimator_

    y_pred_best_rf_otimizado = best_rf_clf_otimizado.predict(X_test)
    y_pred_proba_best_rf_otimizado = best_rf_clf_otimizado.predict_proba(X_test)[:, 1]

    accuracy_best_rf_otimizado = accuracy_score(y_test, y_pred_best_rf_otimizado)
    roc_auc_best_rf_otimizado = roc_auc_score(y_test, y_pred_proba_best_rf_otimizado) if len(np.unique(y_test)) > 1 else 0.5
    conf_matrix_best_rf_otimizado = confusion_matrix(y_test, y_pred_best_rf_otimizado)
    class_report_best_rf_otimizado = classification_report(y_test, y_pred_best_rf_otimizado, target_names=['Não Contratado (0)', 'Contratado (1)'], zero_division=0)

    print("\n--- Resultados do Random Forest Otimizado ---")
    print(f"Acurácia: {accuracy_best_rf_otimizado:.4f}")
    print(f"AUC-ROC: {roc_auc_best_rf_otimizado:.4f}")
    print("\nMatriz de Confusão:")
    print(conf_matrix_best_rf_otimizado)
    print("\nRelatório de Classificação:")
    print(class_report_best_rf_otimizado)

    # ### *10.3. Importância das Features do Modelo Final*
    if hasattr(best_rf_clf_otimizado, 'feature_importances_'):
        feature_importances_final_model = pd.Series(best_rf_clf_otimizado.feature_importances_, index=X_train.columns).sort_values(ascending=False)
        print("\nTop 20 Features Mais Importantes (Modelo Random Forest Otimizado - FINAL):")
        print(feature_importances_final_model.head(20))
        plt.figure(figsize=(10,12))
        sns.barplot(x=feature_importances_final_model.head(20).values, y=feature_importances_final_model.head(20).index)
        plt.title('Top 20 Features Mais Importantes - Random Forest Otimizado')
        plt.xlabel('Importância da Feature')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig(os.path.join(path_artifacts, "feature_importances.png")) # Salvar figura
        print(f"Gráfico de importância das features salvo em {os.path.join(path_artifacts, 'feature_importances.png')}")

    else:
        print("Não foi possível calcular a importância das features do modelo otimizado.")
else:
    print("Análise de resultados e otimização não pode prosseguir: modelo RF baseline não treinado ou dados de teste vazios.")
    best_rf_clf_otimizado = None 


# ---------------------------------------------------------------------------
# ## **11. Demonstração do Modelo com Exemplos REAIS do Conjunto de Teste**
# ---------------------------------------------------------------------------
exemplo_tp_df, exemplo_tn_df = None, None # Inicializar
if best_rf_clf_otimizado is not None and 'X_test' in globals() and not X_test.empty and 'y_test' in globals() and not y_test.empty:
    print("\n--- Iniciando Demonstração com Exemplos do Conjunto de Teste ---")
    # ### *11.1. Selecionando exemplos significativos do conjunto de teste*
    y_pred_proba_test_demo = best_rf_clf_otimizado.predict_proba(X_test)[:, 1]
    prediction_threshold_final_demo = 0.50 
    y_pred_test_binario_demo = (y_pred_proba_test_demo >= prediction_threshold_final_demo).astype(int)

    df_resultados_teste_demo = pd.DataFrame({
        'y_real': y_test.values,
        'y_predito_binario': y_pred_test_binario_demo,
        'probabilidade_contratado': y_pred_proba_test_demo
    }, index=X_test.index)

    tps_demo = df_resultados_teste_demo[
        (df_resultados_teste_demo['y_real'] == 1) & (df_resultados_teste_demo['y_predito_binario'] == 1)
    ].sort_values(by='probabilidade_contratado', ascending=False)

    tns_demo = df_resultados_teste_demo[
        (df_resultados_teste_demo['y_real'] == 0) & (df_resultados_teste_demo['y_predito_binario'] == 0)
    ].sort_values(by='probabilidade_contratado', ascending=True)

    indice_tp_exemplo_demo, indice_tn_exemplo_demo = None, None
    if not tps_demo.empty:
        indice_tp_exemplo_demo = tps_demo.index[0]
        exemplo_tp_df = X_test.loc[[indice_tp_exemplo_demo]] 
        print(f"\n--- Exemplo de Verdadeiro Positivo (TP) Selecionado (Índice Original: {indice_tp_exemplo_demo}) ---")
        print(f"Probabilidade de Contratação (prevista): {tps_demo.loc[indice_tp_exemplo_demo, 'probabilidade_contratado']:.4f}")
    else:
        print("Não foram encontrados Verdadeiros Positivos com o threshold atual para demonstração.")

    if not tns_demo.empty:
        indice_tn_exemplo_demo = tns_demo.index[0]
        exemplo_tn_df = X_test.loc[[indice_tn_exemplo_demo]] 
        print(f"\n--- Exemplo de Verdadeiro Negativo (TN) Selecionado (Índice Original: {indice_tn_exemplo_demo}) ---")
        print(f"Probabilidade de Contratação (prevista): {tns_demo.loc[indice_tn_exemplo_demo, 'probabilidade_contratado']:.4f}")
    else:
        print("Não foram encontrados Verdadeiros Negativos com o threshold atual para demonstração.")

    # ### *11.2. Analisando as Features mais importantes para os exemplos selecionados*
    if 'feature_importances_final_model' in locals() and isinstance(feature_importances_final_model, pd.Series):
        top_10_features_demo = feature_importances_final_model.head(10).index.tolist()
        if exemplo_tp_df is not None:
            print(f"\n--- Valores das Top 10 Features para o Exemplo TP (Índice Original: {indice_tp_exemplo_demo}) ---")
            print(exemplo_tp_df[top_10_features_demo].transpose().rename(columns={exemplo_tp_df.index[0]: 'Valor no Exemplo TP'}))
        if exemplo_tn_df is not None:
            print(f"\n--- Valores das Top 10 Features para o Exemplo TN (Índice Original: {indice_tn_exemplo_demo}) ---")
            print(exemplo_tn_df[top_10_features_demo].transpose().rename(columns={exemplo_tn_df.index[0]: 'Valor no Exemplo TN'}))
    else:
        print("Importância das features ('feature_importances_final_model') não calculada ou não é uma Series.")
else:
    print("Demonstração com exemplos não pode prosseguir: modelo otimizado não treinado ou dados de teste X_test/y_test vazios.")

# ---------------------------------------------------------------------------
# ## **12. Preparação dos arquivos para o streamlit**
# ---------------------------------------------------------------------------

print("\n--- Iniciando Preparação de Arquivos para Streamlit ---")

if 'df_vagas_processado' in globals() and isinstance(df_vagas_processado, pd.DataFrame) and not df_vagas_processado.empty:
    try:
        df_vagas_processado.to_csv(os.path.join(path_data_processed, 'vagas_processadas.csv'), index=False)
        print(f"DataFrame 'df_vagas_processado' (processado) salvo em '{os.path.join(path_data_processed, 'vagas_processadas.csv')}'")
    except Exception as e:
        print(f"Erro ao salvar df_vagas_processado (processado): {e}")
else:
    print("DataFrame 'df_vagas_processado' não encontrado, não é DataFrame ou vazio. Não foi salvo (processado).")

if 'df_candidatos_processado' in globals() and isinstance(df_candidatos_processado, pd.DataFrame) and not df_candidatos_processado.empty:
    try:
        df_candidatos_processado.to_csv(os.path.join(path_data_processed, 'candidatos_processados.csv'), index=False)
        print(f"DataFrame 'df_candidatos_processado' (processado) salvo em '{os.path.join(path_data_processed, 'candidatos_processados.csv')}'")
    except Exception as e:
        print(f"Erro ao salvar df_candidatos_processado (processado): {e}")
else:
    print("DataFrame 'df_candidatos_processado' não encontrado, não é DataFrame ou vazio. Não foi salvo (processado).")

# Salvar o modelo otimizado final
if 'best_rf_clf_otimizado' in locals() and best_rf_clf_otimizado is not None:
    joblib.dump(best_rf_clf_otimizado, os.path.join(path_artifacts, 'modelo_recrutamento_rf.joblib'))
    print(f"Modelo 'best_rf_clf_otimizado' salvo em '{os.path.join(path_artifacts, 'modelo_recrutamento_rf.joblib')}'")
else:
    print("Variável 'best_rf_clf_otimizado' não encontrada ou é None. Nenhum modelo salvo.")
'''
 # Salvar as colunas do modelo
'''
colunas_para_salvar_joblib = None
if 'X' in globals() and isinstance(X, pd.DataFrame) and not X.empty:
    colunas_para_salvar_joblib = X.columns.tolist()
elif 'X_train' in globals() and isinstance(X_train, pd.DataFrame) and not X_train.empty:
    colunas_para_salvar_joblib = X_train.columns.tolist()

if colunas_para_salvar_joblib:
    joblib.dump(colunas_para_salvar_joblib, os.path.join(path_artifacts, 'colunas_modelo.joblib'))
    print(f"{len(colunas_para_salvar_joblib)} colunas do modelo salvas em '{os.path.join(path_artifacts, 'colunas_modelo.joblib')}'")
else:
    print("Nenhuma lista de colunas (X ou X_train) encontrada para salvar.")


artefatos_para_streamlit = {
    'mapa_nivel_idioma': mapa_nivel_idioma, # Definido na Seção 2.5.3
    'mapa_nivel_academico_candidato': mapa_nivel_academico_candidato, # Definido na Seção 3.5.5
    'mapa_nivel_profissional_candidato': mapa_nivel_profissional_candidato, # Definido na Seção 3.5.6
    'tecnologias_lista_vagas': tecnologias_lista_vagas, # Definido na Seção 2.5.7
    'tecnologias_lista_candidatos': tecnologias_lista_candidatos, # Definido na Seção 3.5.2
}
joblib.dump(artefatos_para_streamlit, os.path.join(path_artifacts, 'artefatos_engenharia.joblib'))
print(f"Artefatos de engenharia de features salvos em '{os.path.join(path_artifacts, 'artefatos_engenharia.joblib')}'")

'''
Salvar os DataFrames de exemplo TP e TN
'''
if exemplo_tp_df is not None and not exemplo_tp_df.empty:
    exemplo_tp_df.to_csv(os.path.join(path_artifacts, 'exemplo_tp_streamlit.csv'), index=False)
    print(f"Exemplo TP salvo em '{os.path.join(path_artifacts, 'exemplo_tp_streamlit.csv')}'")
else:
    print("Exemplo TP (exemplo_tp_df) não disponível ou vazio. Não foi salvo.")

if exemplo_tn_df is not None and not exemplo_tn_df.empty:
    exemplo_tn_df.to_csv(os.path.join(path_artifacts, 'exemplo_tn_streamlit.csv'), index=False)
    print(f"Exemplo TN salvo em '{os.path.join(path_artifacts, 'exemplo_tn_streamlit.csv')}'")
else:
    print("Exemplo TN (exemplo_tn_df) não disponível ou vazio. Não foi salvo.")

print("\n--- Processo de salvamento de artefatos concluído. ---")
print(f"Verifique as pastas '{path_data_raw}', '{path_data_processed}' e '{path_artifacts}' no diretório onde o script foi executado.")