import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import re

#  Configuração da Página 
st.set_page_config(layout="wide", page_title="Painel de Otimização de Recrutamento")

#  Funções de Carregamento com Cache 
@st.cache_resource
def carregar_modelo(caminho_modelo):
    try:
        modelo = joblib.load(caminho_modelo)
        return modelo
    except FileNotFoundError:
        st.error(f"Erro: Arquivo do modelo não encontrado em '{caminho_modelo}'. Verifique o caminho.")
        return None
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        return None

@st.cache_data
def carregar_dados_csv(caminho_arquivo, nome_arquivo):
    caminho_completo = os.path.join(caminho_arquivo, nome_arquivo)
    try:
        df = pd.read_csv(caminho_completo)
        if 'id_vaga' in df.columns:
            df['id_vaga'] = df['id_vaga'].astype(str)
        if 'id_candidato' in df.columns:
            df['id_candidato'] = df['id_candidato'].astype(str)
        return df
    except FileNotFoundError:
        st.error(f"Erro: Arquivo de dados '{nome_arquivo}' não encontrado em '{caminho_completo}'. Verifique o caminho.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Erro ao carregar o arquivo '{nome_arquivo}': {e}")
        return pd.DataFrame()

@st.cache_data
def carregar_artefatos_joblib(caminho_artefatos, nome_artefato):
    caminho_completo = os.path.join(caminho_artefatos, nome_artefato)
    try:
        artefato = joblib.load(caminho_completo)
        return artefato
    except FileNotFoundError:
        st.error(f"Erro: Arquivo de artefato '{nome_artefato}' não encontrado em '{caminho_completo}'. Verifique o caminho.")
        return None
    except Exception as e:
        st.error(f"Erro ao carregar o artefato '{nome_artefato}': {e}")
        return None

#  Caminhos para os Arquivos 
PATH_DATA = 'data/'
PATH_ARTIFACTS = 'artifacts/'

#  Funções de Engenharia de Features (Definidas Globalmente) 
def extrair_modalidade(texto):
    texto_lower = str(texto).lower()
    if re.search(r"100% remoto|totalmente remoto|home office|trabalho remoto|remoto", texto_lower):
        if re.search(r"h[íi]brido", texto_lower): return "Híbrido"
        return "Remoto"
    elif re.search(r"h[íi]brido", texto_lower): return "Híbrido"
    elif re.search(r"presencial|no escrit[óo]rio|na planta|loca[çl][ãa]o", texto_lower) and not re.search(r"remoto|h[íi]brido", texto_lower): return "Presencial"
    return "Não Informado"

def codificar_idioma(nivel, mapa_nivel_idioma_local):
    return mapa_nivel_idioma_local.get(str(nivel).lower(), 0)

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

def generalizar_titulo_profissional_candidato(titulo, mapa_nivel_profissional_cand_local):
    if not isinstance(titulo, str): titulo = str(titulo)
    titulo_lower = titulo.lower()
    if titulo_lower == "não informado": return "Outros/Não Especificado"
    if "consultor sap" in titulo_lower or "consultora sap" in titulo_lower or ("sap" in titulo_lower and ("consultant" in titulo_lower or "especialista" in titulo_lower or "consultor(a)" in titulo_lower)): return "Consultoria SAP"
    if "arquiteto sap" in titulo_lower or "architect sap" in titulo_lower: return "Arquitetura SAP"
    if "architect" in titulo_lower or "arquiteto" in titulo_lower or "tech lead" in titulo_lower or "líder técnico" in titulo_lower or "lider técnico" in titulo_lower: return "Liderança Técnica & Arquitetura"
    dev_keywords = ["desenvolvedor", "developer", "programador", "software engineer", "dev", "desenvolvimento", "development", "fullstack", "frontend", "backend", "mobile", "abap", "programadora"]
    if any(keyword in titulo_lower for keyword in dev_keywords): return "Desenvolvimento de Software"
    data_keywords = ["dados", "data", "bi", "business intelligence", "analytics", "cientista", "scientist", "engenheiro de dados", "data engineer", "analista de dados"]
    if any(keyword in titulo_lower for keyword in data_keywords): return "Dados & BI"
    infra_keywords = ["infraestrutura", "infrastructure", "cloud", "aws", "azure", "gcp", "devops", "sysadmin", "rede", "network", "segurança", "security", "sre", "suporte ti", "analista de infraestrutura"]
    if any(keyword in titulo_lower for keyword in infra_keywords): return "Infra, Cloud & DevOps"
    pm_po_keywords = ["gerente de projetos", "project manager", "pm", "gpm", "product owner", "po", "product manager", "coordenador de projetos", "project coordinator", "scrum master", "agile coach", "gestor de projetos", "analista de projetos"]
    if any(keyword in titulo_lower for keyword in pm_po_keywords): return "Gestão de Projetos & Produtos"
    qa_keywords = ["qa", "quality assurance", "testes", "tester", "analista de testes", "automação de testes"]
    if any(keyword in titulo_lower for keyword in qa_keywords): return "Qualidade & Testes"
    support_keywords = ["suporte", "support", "analista de suporte", "service desk", "operações", "operations", "sustentação", "atendimento ao cliente", "help desk"]
    if any(keyword in titulo_lower for keyword in support_keywords): return "Suporte & Operações"
    design_keywords = ["design", "designer", "ux", "ui", "product designer", "web designer"]
    if any(keyword in titulo_lower for keyword in design_keywords): return "Design (UX/UI)"
    analyst_keywords = ["analista", "analyst", "especialista", "specialist"]
    if any(keyword in titulo_lower for keyword in analyst_keywords):
        if "negócios" in titulo_lower or "business" in titulo_lower: return "Funcional & Negócios"
        if "sistemas" in titulo_lower or "systems" in titulo_lower: return "Análise de Sistemas"
        if "processos" in titulo_lower: return "Análise de Processos"
        if "requisitos" in titulo_lower: return "Análise de Requisitos"
        if "financeiro" in titulo_lower or "contábil" in titulo_lower or "fiscal" in titulo_lower : return "Financeiro & Contábil"
        if "rh" in titulo_lower or "recursos humanos" in titulo_lower or "recrutamento" in titulo_lower : return "Recursos Humanos"
        if "marketing" in titulo_lower or "comunicação" in titulo_lower : return "Marketing & Comunicação"
        return "Analista (Genérico)"
    lead_coord_keywords = ["líder", "lider", "coordenador", "supervisor", "gerente", "manager", "diretor", "director", "head", "gestor", "gestora"]
    if any(keyword in titulo_lower for keyword in lead_coord_keywords): return "Liderança & Gestão"
    return "Outros/Não Especificado"

def padronizar_nivel_academico_candidato(nivel, mapa_nivel_academico_cand_local):
    nivel_str = str(nivel).lower()
    if nivel_str == "não informado": return "Não Informado"
    for key, value in mapa_nivel_academico_cand_local.items():
        if key in nivel_str: return value
    return "Outros/Não Especificado"

def padronizar_nivel_profissional_candidato(nivel, mapa_nivel_prof_cand_local):
    nivel_str = str(nivel).lower()
    if nivel_str == "não informado": return "Não Informado"
    for key, value in mapa_nivel_prof_cand_local.items():
        if key in nivel_str: return value
    if 'analista' in nivel_str and not any(k in nivel_str for k in ['júnior', 'pleno', 'sênior', 'jr', 'pl', 'sr', 'junior', 'senior']):
        return "Analista (Nível não especificado)"
    return "Outros/Não Especificado"

def limpar_pcd_candidato(valor_pcd):
    valor_lower = str(valor_pcd).lower().strip()
    if valor_lower in ["sim", "s", "yes", "y", "1", "true", "true", "SIM", "S"]: return "Sim"
    if valor_lower in ["não", "nao", "n", "no", "0", "false", "false", "NAO", "NÃO"]: return "Não"
    return "Não Informado"

def safe_to_numeric_scalar(value, default=0):
    num_val = pd.to_numeric(value, errors='coerce')
    return default if pd.isna(num_val) else num_val

def preparar_features_para_predicao(vaga_selecionada_serie, candidato_serie, colunas_modelo_esperadas_lista, mapas_locais):
    input_data = {}
    mapa_nivel_idioma_local = mapas_locais['mapa_nivel_idioma']
    
    input_data['nivel_ingles_ordinal_vaga'] = codificar_idioma(vaga_selecionada_serie.get('nivel_ingles', "Não Informado"), mapa_nivel_idioma_local)
    input_data['nivel_espanhol_ordinal_vaga'] = codificar_idioma(vaga_selecionada_serie.get('nivel_espanhol', "Não Informado"), mapa_nivel_idioma_local)
    input_data['vaga_sap_bool'] = 1 if str(vaga_selecionada_serie.get('vaga_sap', "Não")).lower() == "sim" else 0
    input_data['nivel_ingles_ordinal_candidato'] = codificar_idioma(candidato_serie.get('nivel_ingles', "Não Informado"), mapa_nivel_idioma_local)
    input_data['nivel_espanhol_ordinal_candidato'] = codificar_idioma(candidato_serie.get('nivel_espanhol', "Não Informado"), mapa_nivel_idioma_local)
    input_data['compat_ingles'] = 1 if input_data['nivel_ingles_ordinal_candidato'] >= input_data['nivel_ingles_ordinal_vaga'] else 0
    input_data['compat_espanhol'] = 1 if input_data['nivel_espanhol_ordinal_candidato'] >= input_data['nivel_espanhol_ordinal_vaga'] else 0
    
    tech_cols_vaga = [col for col in vaga_selecionada_serie.index if col.startswith('tech_')]
    skill_cols_candidato = [col for col in candidato_serie.index if col.startswith('skill_')]
    
    total_techs_vaga_val = 0
    if tech_cols_vaga:
        tech_values_series = vaga_selecionada_serie[tech_cols_vaga]
        total_techs_vaga_val = pd.to_numeric(tech_values_series, errors='coerce').fillna(0).sum()
    input_data['total_techs_vaga'] = total_techs_vaga_val
    
    skills_match_count = 0
    if tech_cols_vaga and skill_cols_candidato:
        mapa_skills_cand_temp = {s.replace('skill_', ''): s for s in skill_cols_candidato}
        for t_col_v in tech_cols_vaga:
            nome_base_t = t_col_v.replace('tech_', '')
            if nome_base_t in mapa_skills_cand_temp:
                s_col_c = mapa_skills_cand_temp[nome_base_t]
                vaga_tech_val = safe_to_numeric_scalar(vaga_selecionada_serie.get(t_col_v, 0))
                cand_skill_val = safe_to_numeric_scalar(candidato_serie.get(s_col_c, 0))
                if vaga_tech_val == 1 and cand_skill_val == 1:
                    skills_match_count += 1
    input_data['skills_match_count'] = skills_match_count
    input_data['skills_faltantes_vaga'] = max(0, input_data['total_techs_vaga'] - skills_match_count)
    input_data['comentario_tem_valor_monetario'] = 0

    input_data['nivel_profissional_vaga'] = str(vaga_selecionada_serie.get('nivel_profissional_vaga', "Não Informado"))
    input_data['nivel_academico_vaga'] = str(vaga_selecionada_serie.get('nivel_academico', "Não Informado"))
    input_data['modalidade_trabalho'] = str(vaga_selecionada_serie.get('modalidade_trabalho', "Não Informado"))
    input_data['categoria_vaga'] = str(vaga_selecionada_serie.get('categoria_vaga', "Não Informado"))
    input_data['vaga_sap'] = str(vaga_selecionada_serie.get('vaga_sap', "Não Informado"))
    input_data['nivel_academico_padronizado_candidato'] = str(candidato_serie.get('nivel_academico_padronizado', "Não Informado"))
    input_data['categoria_profissional_candidato'] = str(candidato_serie.get('categoria_profissional', "Não Informado"))
    input_data['pcd_padronizado_candidato'] = str(candidato_serie.get('pcd_padronizado', "Não Informado"))
    input_data['nivel_profissional_padronizado_candidato'] = str(candidato_serie.get('nivel_profissional_padronizado_candidato', "Não Informado"))
    
    features_categoricas_para_onehot = [
        'nivel_profissional_vaga', 'nivel_academico_vaga', 'modalidade_trabalho',
        'categoria_vaga', 'vaga_sap',
        'nivel_academico_padronizado_candidato', 'categoria_profissional_candidato',
        'pcd_padronizado_candidato', 'nivel_profissional_padronizado_candidato'
    ]
    for col_cat_check in features_categoricas_para_onehot:
        if col_cat_check not in input_data: input_data[col_cat_check] = "Não Informado"
    
    df_input_predict = pd.DataFrame([input_data])
    try:
        df_input_encoded = pd.get_dummies(df_input_predict, columns=features_categoricas_para_onehot, dummy_na=False)
    except Exception as e:
        print(f"DEBUG: Erro one-hot encoding: {e} para dados {input_data}")
        return None

    for tech_col in tech_cols_vaga:
        df_input_encoded[tech_col] = safe_to_numeric_scalar(vaga_selecionada_serie.get(tech_col, 0))
    for skill_col in skill_cols_candidato:
        df_input_encoded[skill_col] = safe_to_numeric_scalar(candidato_serie.get(skill_col, 0))
        
    if colunas_modelo_esperadas_lista is None: return None
    try:
        df_final_predict = df_input_encoded.reindex(columns=colunas_modelo_esperadas_lista, fill_value=0)
        for col in df_final_predict.columns:
            if not pd.api.types.is_numeric_dtype(df_final_predict[col]):
                df_final_predict[col] = pd.to_numeric(df_final_predict[col], errors='coerce').fillna(0)
            else:
                df_final_predict[col] = df_final_predict[col].fillna(0)
        return df_final_predict
    except Exception as e:
        print(f"DEBUG: Erro ao alinhar colunas: {e}. Colunas codificadas: {df_input_encoded.columns.tolist()}")
        return None

def calcular_scores_para_vaga_com_pre_filtro(
    _vaga_id,
    _df_vagas_completo_cache,
    _df_candidatos_completo_cache,
    _colunas_modelo_cache,
    _modelo_obj_cache,
    _mapas_eng_cache,
    _num_min_tech_match=1,
    _max_candidatos_para_score_completo=500,
    _randomize_subset=True 
    ):
    
    if _modelo_obj_cache is None or _colunas_modelo_cache is None or _mapas_eng_cache is None:
        print("Modelo, colunas do modelo ou mapas de engenharia não carregados.")
        return []

    vaga_serie_df_cache = _df_vagas_completo_cache[_df_vagas_completo_cache['id_vaga'] == _vaga_id]
    if vaga_serie_df_cache.empty:
        print(f"Vaga com ID {_vaga_id} não encontrada.")
        return []
    vaga_serie_cache = vaga_serie_df_cache.iloc[0]

    df_candidatos_filtrados_cache = _df_candidatos_completo_cache.copy()
    
    # Filtro 1: Categoria Profissional
    categoria_vaga_cache = vaga_serie_cache.get('categoria_vaga', "Não Informado")
    if categoria_vaga_cache != "Não Informado" and 'categoria_profissional_candidato' in df_candidatos_filtrados_cache.columns:
        df_candidatos_filtrados_cache = df_candidatos_filtrados_cache[
            df_candidatos_filtrados_cache['categoria_profissional_candidato'] == categoria_vaga_cache
        ]
    
    # Filtro 2: Nível Profissional
    nivel_vaga_str_cache = str(vaga_serie_cache.get('nivel_profissional_vaga', "Não Informado")).lower()
    if nivel_vaga_str_cache != "não informado" and 'nivel_profissional_padronizado_candidato' in df_candidatos_filtrados_cache.columns:
        niveis_compativeis_cache = []
        if "sênior" in nivel_vaga_str_cache or "senior" in nivel_vaga_str_cache: niveis_compativeis_cache = ["Sênior", "Especialista", "Liderança/Coordenação", "Gerência/Diretoria"]
        elif "pleno" in nivel_vaga_str_cache: niveis_compativeis_cache = ["Pleno", "Sênior", "Especialista", "Liderança/Coordenação", "Gerência/Diretoria"]
        elif "júnior" in nivel_vaga_str_cache or "jr" in nivel_vaga_str_cache: niveis_compativeis_cache = ["Júnior", "Pleno", "Sênior", "Especialista"]
        if niveis_compativeis_cache:
            df_candidatos_filtrados_cache = df_candidatos_filtrados_cache[
                df_candidatos_filtrados_cache['nivel_profissional_padronizado_candidato'].isin(niveis_compativeis_cache)
            ]
    
    # Filtro 3: Mínimo de Tecnologias em Comum
    techs_requeridas_pela_vaga = [col.replace('tech_', '') for col in vaga_serie_cache.index if col.startswith('tech_') and safe_to_numeric_scalar(vaga_serie_cache.get(col, 0)) == 1]
    
    if techs_requeridas_pela_vaga and _num_min_tech_match > 0:
        indices_filtrados = []
        for idx, cand_row in df_candidatos_filtrados_cache.iterrows():
            count = sum(1 for tech in techs_requeridas_pela_vaga if f"skill_{tech}" in cand_row and safe_to_numeric_scalar(cand_row.get(f"skill_{tech}", 0)) == 1)
            if count >= _num_min_tech_match: indices_filtrados.append(idx)
        df_candidatos_filtrados_cache = df_candidatos_filtrados_cache.loc[indices_filtrados]

    num_candidatos_apos_filtro = len(df_candidatos_filtrados_cache)
    if df_candidatos_filtrados_cache.empty: return []

    # Aplicar limite e randomização
    if num_candidatos_apos_filtro > _max_candidatos_para_score_completo:
        print(f"INFO: Vaga {_vaga_id}: {num_candidatos_apos_filtro} candidatos pós-filtro. ")
        if _randomize_subset:
            print(f"Selecionando aleatoriamente {_max_candidatos_para_score_completo} candidatos.")
            df_candidatos_para_scoring = df_candidatos_filtrados_cache.sample(n=_max_candidatos_para_score_completo, random_state=None) 
        else:
            print(f"Selecionando os primeiros {_max_candidatos_para_score_completo} candidatos.")
            df_candidatos_para_scoring = df_candidatos_filtrados_cache.head(_max_candidatos_para_score_completo)
    else:
        df_candidatos_para_scoring = df_candidatos_filtrados_cache
    
    resultados_candidatos_internos_cache = []
    for _, candidato_serie_atual_cache in df_candidatos_para_scoring.iterrows():
        df_features_predicao_cache = preparar_features_para_predicao(vaga_serie_cache, candidato_serie_atual_cache, _colunas_modelo_cache, _mapas_eng_cache)
        if df_features_predicao_cache is not None and not df_features_predicao_cache.empty:
            try:
                prob_match_cache = _modelo_obj_cache.predict_proba(df_features_predicao_cache)[0, 1]
                resultados_candidatos_internos_cache.append({
                    'ID Candidato': candidato_serie_atual_cache.get('id_candidato', 'N/A'),
                    'Nome': candidato_serie_atual_cache.get('nome', 'N/A'),
                    'Título Profissional': candidato_serie_atual_cache.get('titulo_profissional', 'N/A'), # Mantido caso precise em outro lugar, mas não será exibido na tabela principal
                    'Pontuação de Match': prob_match_cache
                })
            except Exception as e_pred_cache_loop: print(f"Erro ao prever (sem cache) cand {candidato_serie_atual_cache.get('id_candidato', '')} vaga {_vaga_id}: {e_pred_cache_loop}")
    return resultados_candidatos_internos_cache

#  Lógica Principal da Aplicação 
def main():
    st.title("🎯 Painel de Otimização de Recrutamento")
    
    modelo_obj = carregar_modelo(os.path.join(PATH_ARTIFACTS, 'modelo_recrutamento_rf.joblib'))
    colunas_modelo_obj = carregar_artefatos_joblib(PATH_ARTIFACTS, 'colunas_modelo.joblib')
    artefatos_eng_obj = carregar_artefatos_joblib(PATH_ARTIFACTS, 'artefatos_engenharia.joblib')
    df_vagas = carregar_dados_csv(PATH_DATA, 'vagas_processadas.csv')
    df_candidatos = carregar_dados_csv(PATH_DATA, 'candidatos_processados.csv')

    if modelo_obj is None or colunas_modelo_obj is None or artefatos_eng_obj is None or df_vagas.empty or df_candidatos.empty:
        st.error("Um ou mais arquivos essenciais não puderam ser carregados ou estão vazios. A aplicação não pode continuar.")
        st.stop()

    mapas_para_engenharia = {
        'mapa_nivel_idioma': artefatos_eng_obj.get('mapa_nivel_idioma', {}),
        'mapa_nivel_academico_candidato': artefatos_eng_obj.get('mapa_nivel_academico_candidato', {}),
        'mapa_nivel_profissional_candidato': artefatos_eng_obj.get('mapa_nivel_profissional_candidato', {})
    }

    st.markdown("Selecione filtros para vagas, escolha uma vaga e clique em 'Buscar Candidatos'.")
    st.sidebar.header("Filtrar Vagas")
    
    categorias_vaga_unicas = sorted(df_vagas['categoria_vaga'].unique()) if 'categoria_vaga' in df_vagas.columns else []
    cat_selecionada = st.sidebar.multiselect("Categoria da Vaga:", options=categorias_vaga_unicas, default=[])

    modalidades_unicas = sorted(df_vagas['modalidade_trabalho'].unique()) if 'modalidade_trabalho' in df_vagas.columns else []
    mod_selecionada = st.sidebar.multiselect("Modalidade de Trabalho:", options=modalidades_unicas, default=[])

    niveis_vaga_unicos = sorted(df_vagas['nivel_profissional_vaga'].unique()) if 'nivel_profissional_vaga' in df_vagas.columns else []
    nivel_vaga_selecionado = st.sidebar.multiselect("Nível Profissional da Vaga:", options=niveis_vaga_unicos, default=[])
    
    df_vagas_filtrado_display = df_vagas.copy()
    if cat_selecionada and 'categoria_vaga' in df_vagas_filtrado_display.columns:
        df_vagas_filtrado_display = df_vagas_filtrado_display[df_vagas_filtrado_display['categoria_vaga'].isin(cat_selecionada)]
    if mod_selecionada and 'modalidade_trabalho' in df_vagas_filtrado_display.columns:
        df_vagas_filtrado_display = df_vagas_filtrado_display[df_vagas_filtrado_display['modalidade_trabalho'].isin(mod_selecionada)]
    if nivel_vaga_selecionado and 'nivel_profissional_vaga' in df_vagas_filtrado_display.columns:
        df_vagas_filtrado_display = df_vagas_filtrado_display[df_vagas_filtrado_display['nivel_profissional_vaga'].isin(nivel_vaga_selecionado)]

    st.header("Vagas Disponíveis")
    if df_vagas_filtrado_display.empty:
        st.info("Nenhuma vaga encontrada com os filtros selecionados.")
    else:
        cols_display_vagas = ['id_vaga', 'titulo_vaga', 'cliente', 'categoria_vaga', 'modalidade_trabalho', 'nivel_profissional_vaga']
        cols_display_vagas_existentes = [col for col in cols_display_vagas if col in df_vagas_filtrado_display.columns]
        st.dataframe(df_vagas_filtrado_display[cols_display_vagas_existentes], height=300, use_container_width=True)

        if not df_vagas_filtrado_display.empty and 'id_vaga' in df_vagas_filtrado_display.columns and 'titulo_vaga' in df_vagas_filtrado_display.columns:
            df_vagas_selectbox_options = df_vagas_filtrado_display.dropna(subset=['id_vaga', 'titulo_vaga'])
            lista_opcoes_vagas_selectbox = [f"{row['id_vaga']} - {row['titulo_vaga']}" for _, row in df_vagas_selectbox_options.iterrows()]
            
            if not lista_opcoes_vagas_selectbox:
                st.warning("Nenhuma vaga disponível para seleção após a filtragem.")
            else:
                selectbox_key = "vaga_selecionada_selectbox_main"
                if selectbox_key not in st.session_state:
                    st.session_state[selectbox_key] = lista_opcoes_vagas_selectbox[0] if lista_opcoes_vagas_selectbox else None
                
                current_selection = st.session_state[selectbox_key]
                if current_selection not in lista_opcoes_vagas_selectbox:
                    current_selection = lista_opcoes_vagas_selectbox[0] if lista_opcoes_vagas_selectbox else None
                    st.session_state[selectbox_key] = current_selection
                
                current_index = lista_opcoes_vagas_selectbox.index(current_selection) if current_selection in lista_opcoes_vagas_selectbox else 0

                vaga_display_selecionada_selectbox = st.selectbox(
                    "Selecione uma vaga:",
                    options=lista_opcoes_vagas_selectbox,
                    index=current_index,
                    key=selectbox_key
                )
                
                id_vaga_para_widgets = str(vaga_display_selecionada_selectbox.split(" - ")[0]) if vaga_display_selecionada_selectbox else "default_vaga_key"

                num_min_tech_match_param = st.sidebar.slider("Número Mínimo de Tecnologias em Comum (Pré-filtro):", 0, 5, 1, key=f"slider_tech_match_main_{id_vaga_para_widgets}")
                max_candidatos_param = st.sidebar.select_slider("Máximo de Candidatos para Score Detalhado:", options=[50, 100, 200, 500, 1000, len(df_candidatos)], value=500, key=f"slider_max_cand_main_{id_vaga_para_widgets}")
                randomize_subset_param = st.sidebar.checkbox("Randomizar subconjunto de candidatos (se limitado)?", value=True, key=f"cb_random_{id_vaga_para_widgets}")


                if 'last_searched_vaga_id' not in st.session_state:
                    st.session_state.last_searched_vaga_id = None
                if 'last_search_results' not in st.session_state:
                    st.session_state.last_search_results = None
                
                titulo_vaga_para_botao = ""
                if vaga_display_selecionada_selectbox and " - " in vaga_display_selecionada_selectbox:
                    parts = vaga_display_selecionada_selectbox.split(" - ", 1) 
                    titulo_vaga_para_botao = parts[1] if len(parts) > 1 else parts[0]
                else:
                    titulo_vaga_para_botao = "Vaga Selecionada"


                if st.button(f"🚀 Buscar Candidatos para '{titulo_vaga_para_botao}'", key=f"search_button_{id_vaga_para_widgets}"):
                    if vaga_display_selecionada_selectbox:
                        id_vaga_escolhida_atual_btn = str(vaga_display_selecionada_selectbox.split(" - ")[0])
                        
                        with st.spinner(f"Pré-filtrando e calculando compatibilidade dos candidatos para vaga ID {id_vaga_escolhida_atual_btn}..."):
                            resultados_candidatos = calcular_scores_para_vaga_com_pre_filtro(
                                id_vaga_escolhida_atual_btn,
                                df_vagas,
                                df_candidatos,
                                colunas_modelo_obj,
                                modelo_obj,
                                mapas_para_engenharia,
                                _num_min_tech_match=num_min_tech_match_param,
                                _max_candidatos_para_score_completo=max_candidatos_param,
                                _randomize_subset=randomize_subset_param
                            )
                        st.session_state.last_searched_vaga_id = id_vaga_escolhida_atual_btn
                        st.session_state.last_search_results = resultados_candidatos
                    else:
                        st.session_state.last_searched_vaga_id = None
                        st.session_state.last_search_results = None
                
                if vaga_display_selecionada_selectbox:
                    id_vaga_corrente_selectbox = str(vaga_display_selecionada_selectbox.split(" - ")[0])
                    if st.session_state.last_searched_vaga_id == id_vaga_corrente_selectbox and \
                       st.session_state.last_search_results is not None:
                        
                        vaga_original_serie_df_disp = df_vagas[df_vagas['id_vaga'] == st.session_state.last_searched_vaga_id]
                        if not vaga_original_serie_df_disp.empty:
                            titulo_vaga_display_res = vaga_original_serie_df_disp.iloc[0].get('titulo_vaga', 'Vaga Desconhecida')
                            st.subheader(f"Top 5 Candidatos para: {titulo_vaga_display_res}")
                            
                            resultados_para_display = st.session_state.last_search_results
                            if resultados_para_display:
                                df_scores_candidatos_disp = pd.DataFrame(resultados_para_display)
                                if not df_scores_candidatos_disp.empty:
                                    df_top_5_candidatos_disp = df_scores_candidatos_disp.sort_values(by='Pontuação de Match', ascending=False).head(5)
                                    df_top_5_candidatos_disp['Pontuação de Match'] = df_top_5_candidatos_disp['Pontuação de Match'].apply(lambda x: f"{x:.2%}")
                                    
                                    st.dataframe(df_top_5_candidatos_disp[['ID Candidato', 'Nome', 'Pontuação de Match']], use_container_width=True)

                                else:
                                    st.info(f"Nenhum candidato retornado pela função de score para a vaga '{titulo_vaga_display_res}'.")
                            else:
                                st.info(f"Nenhum candidato passou na pré-filtragem ou não há candidatos compatíveis para a vaga '{titulo_vaga_display_res}'.")
                        else:
                             st.error(f"Vaga com ID '{st.session_state.last_searched_vaga_id}' não encontrada para exibir resultados.")
        else:
            st.info("Nenhuma vaga para selecionar (verifique filtros e dados).")

    st.sidebar.markdown("---")
    st.sidebar.markdown("Desenvolvido como parte do Datathon de RH.")

# --- Ponto de Entrada Principal ---
if __name__ == '__main__':
    main()