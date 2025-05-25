# Importação das bibliotecas necessárias para a aplicação
import streamlit as st
import pandas as pd
import numpy as np

# Importações para avaliação de modelos
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score

# Importar classes de modelos referenciadas nas secções de análise
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import plot_tree # Utilizado para visualizar árvores de decisão
import matplotlib.pyplot as plt # Utilizado para exibir gráficos da matplotlib, como plot_tree
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate

# Bibliotecas para controlo de tempo e sistema
import time
import os

# Bibliotecas para visualização interativa (Plotly) e navegação (streamlit-option-menu)
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu

# Biblioteca para carregar e guardar modelos e pré-processadores
import joblib

# --- Configuração da Página Streamlit ---
# Define as configurações iniciais da página, como título, ícone e layout.
st.set_page_config(
    page_title="Sistema de Intervenção de Estudantes", # Título que aparece no separador do navegador
    page_icon="📊", # Ícone que aparece no separador do navegador
    layout="wide", # Usa a largura total da página
    initial_sidebar_state="expanded" # Sidebar começa expandida
)

# --- Estilo CSS Personalizado ---
# Injeta código CSS para personalizar a aparência dos elementos da aplicação.
st.markdown("""
<style>
    /* Estilos para cabeçalhos principais */
    .main-header {
        font-size: 2.8rem;
        color: #1A237E;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    /* Estilos para sub-cabeçalhos */
    .sub-header {
        font-size: 2rem;
        color: #283593;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: bold;
        border-bottom: 2px solid #C5CAE9;
        padding-bottom: 0.5rem;
    }
    /* Estilos para texto informativo */
    .info-text {
        font-size: 1rem;
        color: #424242;
        margin-bottom: 1rem;
        line-height: 1.6;
    }
    /* Estilos para cartões de métricas */
    .metric-card {
        background-color: #E8EAF6;
        border-left: 6px solid #3F51B5;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
     /* Ajusta o estilo do valor exibido pelos widgets st.metric */
    div[data-testid="stMetric"] label div {
        font-size: 1rem !important;
        color: #555 !important;
    }
     div[data-testid="stMetric"] div[data-testid="stMetricDelta"] div {
         font-size: 1.8rem !important;
         font-weight: bold !important;
         color: #1A237E !important;
     }
    /* Estilos para botões */
    .stButton > button {
        background-color: #3F51B5;
        color: white;
        font-weight: bold;
        padding: 0.75rem 1.5rem;
        border-radius: 5px;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s ease;
        font-size: 1.1rem;
    }
    .stButton > button:hover {
        background-color: #303F9F;
    }
     /* Ajusta a largura da sidebar */
    section[data-testid="stSidebar"] {
        width: 300px !important;
        background-color: #f1f3f4;
    }
    /* Estilos para as tabs (separadores) */
    .stTabs [data-baseweb="tab-list"] {
		gap: 24px;
    }

    .stTabs [data-baseweb="tab"] {
		height: 50px;
		white-space: pre-wrap;
		background-color: #E8EAF6;
		border-radius: 4px 4px 0 0;
		gap: 10px;
		padding: 10px 20px;
        font-size: 1rem;
        font-weight: bold;
    }

    .stTabs [data-baseweb="tab"] svg {
		color: #3F51B5;
    }

    .stTabs [data-baseweb="tab"]:hover {
		background-color: #C5CAE9;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
		background-color: #3F51B5;
		color: white;
		border-bottom: 3px solid #FFC107;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] svg {
		color: white;
    }
     /* Estilos para os widgets de alerta (info, warning, error) */
    div[data-testid="stAlert"] {
        font-size: 1rem;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1.5rem;
    }

    /* CSS para o Grid Layout na secção de Previsão Individual */
    .input-grid-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); /* Define colunas responsivas */
        gap: 20px; /* Espaçamento entre os itens do grid */
        margin-bottom: 20px;
    }
    .grid-item {
        display: flex; /* Usa flexbox para organizar conteúdo verticalmente */
        flex-direction: column;
    }
     /* Ajusta o estilo dos alertas dentro dos itens do grid */
    .grid-item div[data-testid="stAlert"] {
        margin-top: 5px; margin-bottom: 5px; padding: 10px;
        font-size: 0.9rem; line-height: 1.4;
    }

    /* Opcional: Ajustar o tamanho dos inputs numéricos e selectboxes dentro do grid item */
     .grid-item div[data-testid="stNumberInput"],
     .grid-item div[data-testid="stSelectbox"] {
         margin-top: 5px;
     }
     /* Estilo para o container que segura o nome da característica e a descrição curta */
     .feature-label-container {
         margin-bottom: 5px;
         display: flex;
         align-items: baseline; /* Alinha o texto base (nome e descrição) */
         flex-wrap: wrap; /* Permite quebrar linha em ecrãs pequenos */
     }
     .feature-label-container strong {
          margin-right: 5px; /* Espaço entre o nome em negrito e a descrição */
     }
     /* Estilo para a descrição pequena entre parêntesis */
     .small-description {
         font-size: 0.8em; /* Tamanho menor */
         font-weight: normal; /* Sem negrito */
         color: #555; /* Cor ligeiramente mais clara */
     }
     /* Garante que o nome da característica em negrito se comporte como um elemento inline-block */
     .grid-item strong {
         display: inline-block;
     }

</style>
""", unsafe_allow_html=True)

# --- Mapeamentos para características Ordinais Numéricas ---
# Dicionário que mapeia valores numéricos de características ordinais para rótulos descritivos em Português de Portugal.
ORDINAL_MAPPINGS = {
    'Medu': {0: 'Nenhuma', 1: 'Ensino Básico (até ao 4º ano)', 2: 'Ensino Básico (5º ao 9º ano)', 3: 'Ensino Secundário', 4: 'Ensino Superior'},
    'Fedu': {0: 'Nenhuma', 1: 'Ensino Básico (até ao 4º ano)', 2: 'Ensino Básico (5º ao 9º ano)', 3: 'Ensino Secundário', 4: 'Ensino Superior'},
    'traveltime': {1: '<15 min', 2: '15-30 min', 3: '30-60 min', 4: '>60 min'},
    'studytime': {1: '<2 horas', 2: '2 a 5 horas', 3: '5 a 10 horas', 4: '>10 horas'},
    'failures': {0: '0 reprovações', 1: '1 reprovação', 2: '2 reprovações', 3: '3+ reprovações'},
    'famrel': {1: 'Muito Mau', 2: 'Mau', 3: 'Regular', 4: 'Bom', 5: 'Excelente'},
    'freetime': {1: 'Muito Pouco', 2: 'Pouco', 3: 'Médio', 4: 'Muito', 5: 'Muito Muito'},
    'goout': {1: 'Muito Raramente', 2: 'Raramente', 3: 'Ocasionalmente', 4: 'Frequentemente', 5: 'Muito Frequentemente'},
    'Dalc': {1: 'Muito Baixo', 2: 'Baixo', 3: 'Médio', 4: 'Alto', 5: 'Muito Alto'},
    'Walc': {1: 'Muito Baixo', 2: 'Baixo', 3: 'Médio', 4: 'Alto', 5: 'Muito Alto'},
    'health': {1: 'Muito Mau', 2: 'Mau', 3: 'Regular', 4: 'Bom', 5: 'Excelente'},
}

# Lista das chaves do dicionário de mapeamentos para identificar as colunas ordinais numéricas.
ordinal_numeric_features_to_map = list(ORDINAL_MAPPINGS.keys())

# --- Descrição Curta das Características ---
# Dicionário usado para exibir descrições curtas na secção de Previsão Individual.
feature_descriptions_short = {
    "school": "Escola", "sex": "Género", "age": "Idade", "address": "Residência",
    "famsize": "Tamanho da família", "Pstatus": "Estado dos pais", "Medu": "Escolaridade da mãe",
    "Fedu": "Escolaridade do pai", "Mjob": "Ocupação da mãe", "Fjob": "Ocupação do pai",
    "reason": "Motivo da escola", "guardian": "Guardião", "traveltime": "Tempo de viagem",
    "studytime": "Tempo de estudo", "failures": "Reprovações", "schoolsup": "Apoio da escola",
    "famsup": "Apoio da família", "paid": "Aulas pagas", "activities": "Atividades extra",
    "nursery": "Frequentou creche", "higher": "Deseja ensino superior", "internet": "Acesso à internet",
    "romantic": "Relacionamento", "famrel": "Qualidade das relações familiares", "freetime": "Tempo livre",
    "goout": "Sair com amigos", "Dalc": "Álcool durante a semana", "Walc": "Álcool ao fim de semana",
    "health": "Estado de saúde", "absences": "Faltas", "passed": "Aprovado"
}

# --- Descrição Completa das Características ---
# Dicionário usado para fornecer descrições detalhadas na secção de Documentação.
full_feature_descriptions = {
    "school": "Escola do estudante (GP ou MS)",
    "sex": "Género do estudante (F ou M)",
    "age": "Idade do estudante",
    "address": "Localização da residência (Urbana ou Rural)",
    "famsize": "Tamanho da família (Maior que 3 ou Menor/Igual a 3)",
    "Pstatus": "Estado de coabitação dos pais (Moram juntos ou Separados)",
    "Medu": "Nível de escolaridade da mãe (0: Nenhuma a 4: Ensino Superior)",
    "Fedu": "Nível de escolaridade do pai (0: Nenhuma a 4: Ensino Superior)",
    "Mjob": "Ocupação da mãe (teacher, health, services, at_home, other)",
    "Fjob": "Ocupação do pai (teacher, health, services, at_home, other)",
    "reason": "Motivo pela escolha da escola (home, reputation, course, other)",
    "guardian": "Guardião do estudante (mother, father, other)",
    "traveltime": "Tempo de viagem para a escola (1: <15 min a 4: >60 min)",
    "studytime": "Tempo de estudo semanal (1: <2 horas a 4: >10 horas)",
    "failures": "Número de reprovações anteriores (0 a 3+)",
    "schoolsup": "Apoio educacional extra da escola (yes ou no)",
    "famsup": "Apoio educacional familiar (yes ou no)",
    "paid": "Fez aulas pagas extra (yes ou no)",
    "activities": "Participa de atividades extracurriculares (yes ou no)",
    "nursery": "Frequentou creche/pré-escola (yes ou no)",
    "higher": "Deseja frequentar ensino superior (yes ou no)",
    "internet": "Tem acesso à internet em casa (yes ou no)",
    "romantic": "Está num relacionamento romântico (yes ou no)",
    "famrel": "Qualidade das relações familiares (1: muito mau a 5: excelente)",
    "freetime": "Tempo livre após a escola (1: muito pouco a 5: muito)",
    "goout": "Frequência com que sai com amigos (1: muito raramente a 5: muito frequentemente)",
    "Dalc": "Consumo de álcool durante a semana (1: muito baixo a 5: muito alto)",
    "Walc": "Consumo de álcool ao fim de semana (1: muito baixo a 5: muito alto)",
    "health": "Estado de saúde atual (1: muito mau a 5: muito bom)",
    "absences": "Número de faltas escolares",
    "passed": "O estudante foi aprovado (yes ou no) - Variável Alvo"
}

# Função para carregar um modelo específico de um ficheiro .joblib, com caching para desempenho.
@st.cache_resource
def load_specific_model(model_filename):
    """
    Carrega um modelo de Machine Learning a partir de um ficheiro .joblib na pasta 'artefacts'.

    Args:
        model_filename (str): O nome do ficheiro do modelo na pasta 'artefacts/'.

    Returns:
        O modelo carregado se for bem sucedido, None caso contrário.
    """
    artefacts_path = 'artefacts/'
    model_path = os.path.join(artefacts_path, model_filename)
    try:
        loaded_model = joblib.load(model_path)
        # st.success(f"✅ Modelo '{model_filename}' carregado com sucesso!") # Comentado pois pode gerar muitas mensagens
        return loaded_model
    except FileNotFoundError:
        st.error(f"❌ Erro: O ficheiro do modelo '{model_filename}' não foi encontrado na pasta 'artefacts/'.")
        return None
    except Exception as e:
        st.error(f"❌ Ocorreu um erro ao carregar o modelo '{model_filename}': {e}")
        return None


# Função para exibir uma animação de carregamento/processamento no Streamlit.
def loading_animation(text="A processar..."):
    """
    Exibe uma barra de progresso animada no Streamlit.

    Args:
        text (str): O texto a ser exibido junto à barra de progresso.
    """
    progress_text = text
    my_bar = st.progress(0, text=progress_text)

    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1, text=progress_text)
    time.sleep(0.5) # Mantém a barra por um pequeno período após completar
    my_bar.empty() # Remove a barra de progresso

# Função para gerar uma matriz de confusão interativa utilizando Plotly.
def plot_confusion_matrix_interactive(y_true, y_pred, class_names=None):
    """
    Gera uma matriz de confusão interativa a partir dos valores reais e previstos.

    Args:
        y_true (array-like): Valores reais das classes.
        y_pred (array-like): Valores previstos das classes.
        class_names (list): Lista dos nomes das classes (e.g., ['no', 'yes']).

    Returns:
        tuple: Uma tupla contendo o objeto Figure do Plotly e a matriz de confusão NumPy.
    """
    cm = confusion_matrix(y_true, y_pred)

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=class_names, # Rótulos para o eixo x (previstos)
        y=class_names, # Rótulos para o eixo y (reais)
        colorscale='Blues', # Esquema de cores
        showscale=True,
        text=cm, # Mostra os números na matriz
        texttemplate="%{text}",
        textfont={"size": 20},
        hoverinfo="x+y+z", # Informação ao passar o rato por cima
    ))

    fig.update_layout(
        title='Matriz de Confusão',
        xaxis_title='Valores Previstos',
        yaxis_title='Valores Reais',
        xaxis=dict(side='top'), # Eixo X no topo
        yaxis=dict(autorange="reversed"), # Inverte o eixo Y para corresponder à estrutura da matriz
        margin=dict(t=50, b=50, l=50, r=50), # Margens
    )

    return fig, cm

# Função para visualizar uma matriz de correlação utilizando Plotly Express.
def plot_correlation_matrix_px(df):
    """
    Gera um mapa de calor da matriz de correlação para as colunas numéricas de um DataFrame.

    Args:
        df (pd.DataFrame): O DataFrame contendo os dados.

    Returns:
        tuple: Uma tupla contendo o objeto Figure do Plotly Express e o DataFrame da matriz de correlação,
               ou (None, None) se não houver colunas numéricas.
    """
    # Seleciona apenas colunas com tipos de dados numéricos.
    df_numeric = df.select_dtypes(include=np.number)

    # Retorna None se não houver colunas numéricas para calcular a correlação.
    if df_numeric.empty:
         return None, None

    # Calcula a matriz de correlação.
    corr = df_numeric.corr()

    # Cria o mapa de calor interativo usando plotly express.
    fig = px.imshow(
        corr,
        labels=dict(color="Correlação"), # Rótulo da barra de cor
        x=corr.columns, # Rótulos do eixo x (nomes das colunas)
        y=corr.columns, # Rótulos do eixo y (nomes das colunas)
        color_continuous_scale='RdBu_r', # Esquema de cores (vermelho/azul divergente)
        range_color=[-1, 1], # Define o intervalo da escala de cor
        aspect="auto", # Ajusta o aspeto
        text_auto=".2f", # Mostra o valor da correlação em cada célula
    )

    fig.update_layout(
        title="Matriz de Correlação", # Título do gráfico
        margin=dict(t=50, b=50, l=50, r=50), # Margens
    )

    return fig, corr

# --- Carregar Artefactos Treinados de forma segura ---
# Utiliza caching para evitar recarregar os ficheiros em cada interação do Streamlit.
@st.cache_resource
def load_pipeline_artefacts_safe():
    """
    Carrega os artefactos essenciais do pipeline de Machine Learning (pré-processador,
    modelo principal e nomes das colunas) a partir da pasta 'artefacts/'.

    Retorna:
        tuple: (True, (preprocessor, model, original_cols, processed_cols)) se for bem sucedido,
               ou (False, mensagem_de_erro) caso contrário.
    """
    artefacts_path = 'artefacts/'
    # Define os caminhos esperados para os ficheiros de artefactos.
    preprocessor_path = os.path.join(artefacts_path, 'preprocessor.joblib')
    model_path = os.path.join(artefacts_path, 'best_model.joblib') # Assume 'best_model.joblib' é o modelo principal
    original_cols_path = os.path.join(artefacts_path, 'original_input_columns.joblib')
    processed_cols_path = os.path.join(artefacts_path, 'processed_feature_names.joblib')

    try:
        # Tenta carregar cada um dos artefactos.
        preprocessor = joblib.load(preprocessor_path)
        model = joblib.load(model_path)
        original_cols = joblib.load(original_cols_path)
        processed_cols = joblib.load(processed_cols_path)

        # Exibe mensagem de sucesso ao carregar.
        # st.success("✅ Artefactos do pipeline (pré-processador, modelo e nomes de colunas) carregados com sucesso!")
        return True, (preprocessor, model, original_cols, processed_cols)

    except FileNotFoundError as e:
        # Exibe mensagem de erro específica para ficheiros não encontrados.
        error_msg = f"❌ Erro ao carregar artefactos essenciais: {e}. Certifique-se de que todos os ficheiros .joblib estão na pasta '{artefacts_path}' e têm os nomes corretos."
        return False, error_msg
    except Exception as e:
        # Exibe mensagem de erro para outras exceções durante o carregamento.
        error_msg = f"❌ Ocorreu um erro inesperado ao carregar artefactos: {e}"
        return False, error_msg

# Chama a função de carregamento dos artefactos assim que a aplicação inicia.
success_artefacts, loaded_artefacts_result = load_pipeline_artefacts_safe()

# Verifica se o carregamento dos artefactos falhou e para a execução da aplicação se necessário.
if not success_artefacts:
    st.error(loaded_artefacts_result) # Exibe a mensagem de erro retornada
    st.stop() # Para a execução da aplicação Streamlit
else:
    # Se o carregamento foi bem sucedido, desempacota os artefactos carregados.
    preprocessor, model, original_cols, processed_cols = loaded_artefacts_result


# --- Carregar o Dataset Original para Exploração de Dados (EDA) ---
# Utiliza caching para evitar recarregar o dataset em cada interação.
@st.cache_data
def load_student_data():
    """
    Carrega o dataset original 'student-data.csv'.

    Retorna:
        pd.DataFrame: O DataFrame carregado se for bem sucedido.
    """
    data_path = 'student-data.csv'
    try:
        df = pd.read_csv(data_path)
        # st.success(f"✅ Dataset '{data_path}' carregado com sucesso ({df.shape[0]} linhas, {df.shape[1]} colunas).")
        return df
    except FileNotFoundError:
        st.error(f"❌ Erro: O ficheiro '{data_path}' não foi encontrado. Certifique-se de que o dataset está no local correto.")
        st.stop() # Para a execução se o dataset principal não for encontrado
    except Exception as e:
        st.error(f"❌ Ocorreu um erro ao carregar o dataset: {e}")
        st.stop() # Para a execução se ocorrer outro erro

# Carrega o dataset original ao iniciar a aplicação.
student_df_original = load_student_data()

# Identifica o nome da coluna alvo no dataset original.
TARGET_ORIGINAL_NAME = 'passed'
# Verifica se a coluna alvo original está presente e exibe um aviso se não estiver.
if TARGET_ORIGINAL_NAME not in student_df_original.columns:
    st.error(f"❌ Coluna alvo original '{TARGET_ORIGINAL_NAME}' não encontrada no dataset. A aplicação pode não funcionar corretamente.")
    # st.stop() # Opcional: Parar se a coluna alvo não existir, mas pode ser útil permitir a EDA mesmo assim.

# Define os nomes das classes de saída do modelo (usado para rótulos nos gráficos e métricas).
CLASS_NAMES = ['no', 'yes']

# Define o nome da coluna alvo após o pré-processamento (usado nos dados processados).
TARGET_PROCESSED_NAME = 'passed_mapped' # Assume que o pipeline de pré-processamento mapeia a coluna 'passed' para 'passed_mapped'.


# --- Função para carregar os conjuntos de dados processados (treino e teste) ---
# Utiliza caching para evitar recarregar os dados processados em cada interação.
@st.cache_data
def load_processed_data(target_col_name):
    """
    Carrega os conjuntos de dados de treino e teste processados.

    Args:
        target_col_name (str): O nome da coluna alvo nos dados processados.

    Returns:
        tuple: Uma tupla contendo os DataFrames de treino e teste processados,
               ou (None, None) se o carregamento falhar.
    """
    processed_train_path = 'data/processed/train_processed.csv'
    processed_test_path = 'data/processed/test_processed.csv'

    train_df_processed = None
    test_df_processed = None
    errors = [] # Lista para armazenar mensagens de erro/aviso durante o carregamento.

    # Tenta carregar o conjunto de treino processado.
    train_df_processed = pd.read_csv(processed_train_path)
        # Verifica se a coluna alvo processada existe no DataFrame de treino.
    if target_col_name not in train_df_processed.columns:
             errors.append(f"❌ Erro: A coluna alvo processada '{target_col_name}' não foi encontrada no ficheiro '{processed_train_path}'.")
             train_df_processed = None # Define como None se a coluna alvo não for encontrada.
   


    # Tenta carregar o conjunto de teste processado.
    
    test_df_processed = pd.read_csv(processed_test_path)
    # Verifica se a coluna alvo processada existe no DataFrame de teste.
    if target_col_name not in test_df_processed.columns:
            errors.append(f"❌ Erro: A coluna alvo processada '{target_col_name}' não foi encontrada no ficheiro '{processed_test_path}'.")
            test_df_processed = None # Define como None se a coluna alvo não for encontrada.
    
    # Exibe quaisquer mensagens de erro/aviso recolhidas.
    for err in errors:
        st.markdown(err)

    return train_df_processed, test_df_processed

# Carrega os conjuntos de treino e teste processados ao iniciar a aplicação.
train_df_processed_global, test_df_processed_global = load_processed_data(TARGET_PROCESSED_NAME)


# --- Dicionário de modelos disponíveis para a secção "Avaliação e Comparação de Modelos" ---
# Mapeia um nome amigável (chave) para uma instância do modelo scikit-learn (valor).
# Estas instâncias serão treinadas temporariamente na secção de comparação.
AVAILABLE_MODELS_FOR_ANALYSIS = {
    "Regressão Logística": LogisticRegression(random_state=42, max_iter=1000),
    "KNN": KNeighborsClassifier(),
    "Árvore de Decisão": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM (Kernel RBF)": SVC(probability=True, random_state=42), # probability=True necessário para predict_proba e AUC ROC
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42)
}


# --- Configuração da Sidebar para navegação ---
with st.sidebar:
    # --- Espaço para o Logotipo (instruções comentadas) ---
    # Para adicionar o seu logotipo, descomente e ajuste a linha abaixo:
    # st.image("caminho/para/o/seu/logotipo.png", width=250)

    # Título principal na sidebar.
    st.markdown('<h1 class="sub-header" style="text-align: center;">Sistema de Intervenção de Estudantes</h1>', unsafe_allow_html=True)

    # Menu de navegação principal utilizando streamlit-option-menu.
    menu = option_menu(
        menu_title=None,  # Não exibe título do menu
        options=["Início", "Exploração de Dados", "Previsão Individual", "Análise do Modelo Treinado Principal", "Avaliação de Modelos (CM)", "Documentação"],
        icons=["house-door", "bar-chart-line", "clipboard-data", "robot", "grid-3x3", "book"], # Ícones Bootstrap para cada opção
        menu_icon="cast", # Ícone do menu recolhido
        default_index=0, # Índice da opção selecionada por defeito (Início)
    )

    st.markdown("---") # Separador visual

    # --- Informação sobre a Aplicação na Sidebar ---
    st.markdown("### Sobre a Aplicação")
    st.info("""
    Ferramenta interativa para explorar o conjunto de dados de estudantes, fazer previsões
    individuais e analisar o modelo de Machine Learning treinado e as suas propriedades.
    """)

    # --- Detalhes do Projeto Académico ---
    st.markdown("---")
    st.markdown("### Projeto Académico")
    st.write("Desenvolvido por:")
    st.write("- Afonso Marcos")
    st.write("- Afonso Salgado")
    st.write("- Pedro Afonso")

    # --- Detalhes Técnicos ---
    st.markdown("---")
    st.markdown("### Detalhes Técnicos")
    st.write("Framework: Streamlit")
    st.write("Linguagem: Python")
    st.write("Bibliotecas: scikit-learn, pandas, numpy, plotly, joblib")

if menu == "Início":
    # --- INÍCIO PROFISSIONAL DO SISTEMA ---

# Cabeçalho institucional elegante
    st.markdown("""
        <div style="
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            padding: 2.5rem 2rem;
            border-radius: 8px;
            margin-bottom: 2rem;
            border-bottom: 4px solid #3498db;
        ">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <h1 style="
                        color: white;
                        font-size: 2.2rem;
                        font-weight: 600;
                        margin-bottom: 0.5rem;
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    ">
                        Sistema de Intervenção Estudantil
                    </h1>
                    <p style="
                        color: #bdc3c7;
                        font-size: 1.1rem;
                        margin: 0;
                        font-weight: 300;
                    ">
                        Plataforma de Análise Preditiva para Gestão Académica
                    </p>
                </div>
                <div style="
                    background: rgba(52, 152, 219, 0.2);
                    padding: 1rem;
                    border-radius: 50%;
                    border: 2px solid #3498db;
                ">
                    <span style="font-size: 2rem; color: #3498db;">📊</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Resumo executivo
    st.markdown("""
        <div style="
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            padding: 1.8rem;
            border-radius: 6px;
            margin-bottom: 2rem;
            border-left: 4px solid #3498db;
        ">
            <h3 style="color: #2c3e50; margin-bottom: 1rem; font-size: 1.3rem;">A nossa Plataforma</h3>
            <p style="
                font-size: 1rem; 
                line-height: 1.6; 
                margin: 0; 
                color: #495057;
                text-align: justify;
            ">
                Esta plataforma integra uma pipeline de <strong>Machine Learning</strong> com dados do 
                dataset "student-data.csv" para fornecer insights estratégicos sobre o desempenho 
                estudantil. eO sistema prmite a identificação precoce de estudantes em risco, facilitando 
                intervenções pedagógicas direcionadas e melhoria dos resultados académicos institucionais.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Indicadores-chave de performance (KPIs)
    st.markdown("""
        <div style="margin-bottom: 2rem;">
            <h3 style="
                color: #2c3e50; 
                font-size: 1.4rem; 
                margin-bottom: 1.5rem;
                padding-bottom: 0.5rem;
                border-bottom: 2px solid #ecf0f1;
            ">Indicadores-Chave do Sistema</h3>
        </div>
        """, unsafe_allow_html=True)

    col1_kpi, col2_kpi, col3_kpi = st.columns(3)

    with col1_kpi:
            st.markdown("""
            <div style="
                background: white;
                border: 1px solid #dee2e6;
                padding: 1.5rem;
                border-radius: 6px;
                text-align: center;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            ">
                <div style="
                    background: #ecf0f1;
                    width: 60px;
                    height: 60px;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    margin: 0 auto 1rem auto;
                ">
                    <span style="font-size: 1.5rem; color: #34495e;">📈</span>
                </div>
                <div style="font-size: 1.8rem; font-weight: 600; color: #2c3e50; margin-bottom: 0.5rem;">
            """, unsafe_allow_html=True)
            st.write(f"{student_df_original.shape[0]:,}")
            st.markdown("""
                </div>
                <div style="font-size: 0.9rem; color: #6c757d; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px;">
                    Registos Estudantis
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col2_kpi:
            st.markdown("""
            <div style="
                background: white;
                border: 1px solid #dee2e6;
                padding: 1.5rem;
                border-radius: 6px;
                text-align: center;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            ">
                <div style="
                    background: #ecf0f1;
                    width: 60px;
                    height: 60px;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    margin: 0 auto 1rem auto;
                ">
                    <span style="font-size: 1.5rem; color: #34495e;">🔍</span>
                </div>
                <div style="font-size: 1.8rem; font-weight: 600; color: #2c3e50; margin-bottom: 0.5rem;">
            """, unsafe_allow_html=True)
            
            if 'original_cols' in locals() and original_cols is not None:
                st.write(f"{len(original_cols)}")
            else:
                num_features_fallback = student_df_original.shape[1] - (1 if TARGET_ORIGINAL_NAME in student_df_original.columns else 0)
                st.write(f"{num_features_fallback}")
            
            st.markdown("""
                </div>
                <div style="font-size: 0.9rem; color: #6c757d; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px;">
                    Variáveis Analisadas
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col3_kpi:
            # Determinar status do sistema
            if success_artefacts and preprocessor is not None and model is not None and processed_cols is not None:
                status_text = "OPERACIONAL"
                status_icon = "✓"
                status_color = "#27ae60"
                icon_bg = "#d5f4e6"
            else:
                status_text = "MANUTENÇÃO"
                status_icon = "⚠"
                status_color = "#e74c3c"
                icon_bg = "#fdeaea"
            
            st.markdown(f"""
            <div style="
                background: white;
                border: 1px solid #dee2e6;
                padding: 1.5rem;
                border-radius: 6px;
                text-align: center;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            ">
                <div style="
                    background: {icon_bg};
                    width: 60px;
                    height: 60px;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    margin: 0 auto 1rem auto;
                ">
                    <span style="font-size: 1.5rem; color: {status_color};">{status_icon}</span>
                </div>
                <div style="font-size: 1.2rem; font-weight: 600; color: {status_color}; margin-bottom: 0.5rem;">
                    {status_text}
                </div>
                <div style="font-size: 0.9rem; color: #6c757d; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px;">
                    Estado do Sistema
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Separador profissional
    st.markdown("""
        <hr style="
            margin: 2.5rem 0;
            border: none;
            height: 1px;
            background: linear-gradient(90deg, transparent, #bdc3c7, transparent);
        ">
        """, unsafe_allow_html=True)

        # Módulos do sistema
    st.markdown("""
        <h3 style="
            color: #2c3e50; 
            font-size: 1.4rem; 
            margin-bottom: 1.5rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #ecf0f1;
        ">Módulos Disponíveis</h3>
        """, unsafe_allow_html=True)

        # Grid de módulos em layout corporativo
    module_col1, module_col2 = st.columns(2)

    with module_col1:
            st.markdown("""
            <div style="
                background: white;
                border: 1px solid #dee2e6;
                padding: 1.5rem;
                border-radius: 6px;
                margin-bottom: 1.5rem;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            ">
                <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                    <div style="
                        background: #3498db;
                        width: 40px;
                        height: 40px;
                        border-radius: 4px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        margin-right: 1rem;
                    ">
                        <span style="color: white; font-size: 1.2rem;">📊</span>
                    </div>
                    <h4 style="margin: 0; color: #2c3e50; font-size: 1.1rem;">Análise Exploratória</h4>
                </div>
                <p style="margin: 0; color: #6c757d; font-size: 0.95rem; line-height: 1.5;">
                    Visualização de dados, estatísticas descritivas e análise de correlações 
                    para compreensão profunda dos padrões estudantis.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style="
                background: white;
                border: 1px solid #dee2e6;
                padding: 1.5rem;
                border-radius: 6px;
                margin-bottom: 1.5rem;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            ">
                <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                    <div style="
                        background: #e67e22;
                        width: 40px;
                        height: 40px;
                        border-radius: 4px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        margin-right: 1rem;
                    ">
                        <span style="color: white; font-size: 1.2rem;">📋</span>
                    </div>
                    <h4 style="margin: 0; color: #2c3e50; font-size: 1.1rem;">Avaliação de Performance</h4>
                </div>
                <p style="margin: 0; color: #6c757d; font-size: 0.95rem; line-height: 1.5;">
                    Métricas de precisão, matrizes de confusão e análise de interpretabilidade 
                    do modelo preditivo implementado.
                </p>
            </div>
            """, unsafe_allow_html=True)

    with module_col2:
            st.markdown("""
            <div style="
                background: white;
                border: 1px solid #dee2e6;
                padding: 1.5rem;
                border-radius: 6px;
                margin-bottom: 1.5rem;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            ">
                <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                    <div style="
                        background: #27ae60;
                        width: 40px;
                        height: 40px;
                        border-radius: 4px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        margin-right: 1rem;
                    ">
                        <span style="color: white; font-size: 1.2rem;">🎯</span>
                    </div>
                    <h4 style="margin: 0; color: #2c3e50; font-size: 1.1rem;">Previsão Individualizada</h4>
                </div>
                <p style="margin: 0; color: #6c757d; font-size: 0.95rem; line-height: 1.5;">
                    Sistema de previsão em tempo real para identificação de estudantes 
                    que necessitam de intervenção pedagógica específica.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style="
                background: white;
                border: 1px solid #dee2e6;
                padding: 1.5rem;
                border-radius: 6px;
                margin-bottom: 1.5rem;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            ">
                <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                    <div style="
                        background: #8e44ad;
                        width: 40px;
                        height: 40px;
                        border-radius: 4px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        margin-right: 1rem;
                    ">
                        <span style="color: white; font-size: 1.2rem;">⚖️</span>
                    </div>
                    <h4 style="margin: 0; color: #2c3e50; font-size: 1.1rem;">Comparação de Algoritmos</h4>
                </div>
                <p style="margin: 0; color: #6c757d; font-size: 0.95rem; line-height: 1.5;">
                    Benchmarking de diferentes algoritmos de Machine Learning para 
                    otimização contínua da precisão preditiva.
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Nota informativa sobre navegação
    st.markdown("""
        <div style="
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            padding: 1.2rem;
            border-radius: 6px;
            margin-top: 2rem;
            border-left: 4px solid #6c757d;
        ">
            <p style="margin: 0; color: #495057; font-size: 0.95rem;">
                <strong>Navegação:</strong> Utilize o menu lateral para aceder aos diferentes módulos do sistema. 
                Cada módulo foi concebido para fornecer insights específicos que apoiam a tomada de decisões 
                estratégicas na gestão académica.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Alerta sobre o estado do pipeline (apenas se houver problemas)
    if not (success_artefacts and preprocessor is not None and model is not None and processed_cols is not None):
            st.markdown("""
            <div style="
                background: #fff3cd;
                border: 1px solid #ffeaa7;
                color: #856404;
                padding: 1rem;
                border-radius: 6px;
                margin-top: 1.5rem;
                border-left: 4px solid #f39c12;
            ">
                <strong>⚠️ Aviso do Sistema:</strong> O pipeline de Machine Learning não está completamente carregado. 
                Algumas funcionalidades podem apresentar limitações. Recomenda-se verificar a configuração dos artefactos.
            </div>
            """, unsafe_allow_html=True)

elif menu == "Exploração de Dados":
    st.markdown('<h1 class="main-header">Exploração do Dataset Estudantil</h1>', unsafe_allow_html=True)

    df_eda = student_df_original.copy() # Renomeei para df_eda para evitar conflitos com 'df' genérico

    st.markdown('<p class="info-text">Analise a estrutura, distribuição e relações entre as características do seu dataset de dados estudantis (`student-data.csv`).</p>', unsafe_allow_html=True)

    # Define os separadores (tabs) para organizar o conteúdo.
    tab_summary, tab_distributions, tab_relations = st.tabs(["📋 Resumo Geral", "📈 Distribuições", "🔍 Relações"])

    with tab_summary: # Resumo Geral: Dimensões, Target Info, Head, Describe.
        st.markdown('<h2 class="sub-header">Resumo Geral do Dataset</h2>', unsafe_allow_html=True)
        col_dim_summary, col_target_dist = st.columns(2)
        with col_dim_summary:
            st.write("**Dimensões do Dataset:**", df_eda.shape)
            # Certifique-se que original_cols existe se for usada aqui
            if 'original_cols' in globals() and original_cols is not None:
                    st.write(f"**Número de Características (Input Originais):** {len(original_cols)}")
            else:
                    # Fallback logic needs TARGET_ORIGINAL_NAME to be defined
                    if 'TARGET_ORIGINAL_NAME' in globals():
                        num_features_fallback = df_eda.shape[1] - (1 if TARGET_ORIGINAL_NAME in df_eda.columns else 0)
                        st.warning("Lista de nomes das características originais não carregada. Contagem baseada no dataset.")
                        st.write(f"**Número de Características (Input Originais):** {num_features_fallback}")
                    else:
                        st.warning("Lista de nomes das características originais e nome da coluna alvo não definidos.")
                        st.write(f"**Número de Características (Total no Dataset):** {df_eda.shape[1]}")


            st.write(f"**Número de Amostras:** {df_eda.shape[0]}")

            # Certifique-se que TARGET_ORIGINAL_NAME e CLASS_NAMES estão definidos
            if 'TARGET_ORIGINAL_NAME' in globals() and TARGET_ORIGINAL_NAME in df_eda.columns:
                    st.write(f"**Variável Alvo Identificada:** '{TARGET_ORIGINAL_NAME}'")
                    unique_target_values = df_eda[TARGET_ORIGINAL_NAME].dropna().unique().tolist()
                    st.write(f"**Classes Presentes na Variável Alvo:** {', '.join(map(str, unique_target_values))}")
            elif 'TARGET_ORIGINAL_NAME' in globals():
                    st.info(f"Coluna alvo '{TARGET_ORIGINAL_NAME}' não encontrada no dataset.")
            else:
                st.warning("Nome da coluna alvo não definido ('TARGET_ORIGINAL_NAME').")


            st.markdown('---')
            st.write("**Primeiras 5 Linhas do Dataset:**")
            st.dataframe(df_eda.head(), use_container_width=True) # Exibe dataframe head.

        with col_target_dist: # Distribuição da Variável Alvo (Pie Chart).
                # Certifique-se que TARGET_ORIGINAL_NAME e CLASS_NAMES estão definidos
                if 'TARGET_ORIGINAL_NAME' in globals() and 'CLASS_NAMES' in globals() and TARGET_ORIGINAL_NAME in df_eda.columns:
                    st.write(f"**Distribuição dos Valores na Coluna '{TARGET_ORIGINAL_NAME}':**")
                    class_counts = df_eda[TARGET_ORIGINAL_NAME].value_counts().reset_index()
                    class_counts.columns = ['Classe', 'Contagem']

                    # Mapeamento de cores. Assumimos que as classes serão aquelas em CLASS_NAMES ('não'/'sim').
                    # Adiciona uma verificação para garantir que CLASS_NAMES tem 2 elementos
                    if len(CLASS_NAMES) >= 2:
                        color_discrete_map_target = {CLASS_NAMES[0]: 'salmon', CLASS_NAMES[1]: 'lightgreen'}
                    else:
                        st.warning("CLASS_NAMES não tem pelo menos dois elementos para o mapeamento de cores. Usando cores padrão do Plotly.")
                        color_discrete_map_target = None # Usa cores padrão


                    fig_pie = px.pie(
                        data_frame=class_counts, # <<< CORREÇÃO AQUI: Usar data_frame=
                        values='Contagem',
                        names='Classe',
                        title=f"Distribuição de '{TARGET_ORIGINAL_NAME.replace('_', ' ').title()}'",
                        hole=0.3,
                        color='Classe',
                        color_discrete_map=color_discrete_map_target
                    )
                    fig_pie.update_layout(legend_title_text=TARGET_ORIGINAL_NAME.replace('_', ' ').title())
                    st.plotly_chart(fig_pie, use_container_width=True)
                    st.info(f"Este gráfico de 'donut' mostra a proporção de alunos para cada valor único na coluna '{TARGET_ORIGINAL_NAME}'.")
                elif 'TARGET_ORIGINAL_NAME' in globals():
                    st.info(f"Não é possível apresentar o gráfico de distribuição da coluna alvo '{TARGET_ORIGINAL_NAME}' (coluna não encontrada ou CLASS_NAMES não definido).") # PT-PT.
                else:
                    st.warning("Nome da coluna alvo ou CLASS_NAMES não definidos.")


        st.markdown('<h2 class="sub-header">Estatísticas Descritivas</h2>', unsafe_allow_html=True)
        st.dataframe(df_eda.describe(include='all'), use_container_width=True)


    with tab_distributions: # Distribuições Individuais das Características.
        st.markdown('<h2 class="sub-header">Distribuição Individual das Características</h2>', unsafe_allow_html=True)
        st.markdown('<p class="info-text">Visualize a distribuição de cada característica individualmente ou compare a sua distribuição consoante a situação final do aluno.</p>', unsafe_allow_html=True)

        # Seleção de características para visualização. Usa original_cols se disponíveis. Exclui a target.
        # Certifique-se que original_cols ou df_eda existem antes de criar feature_options_dist
        if 'original_cols' in globals() and original_cols is not None:
                feature_options_dist = original_cols[:] # Usar cópia para não modificar a lista original
        elif 'df_eda' in locals():
                feature_options_dist = df_eda.columns.tolist()
        else:
                feature_options_dist = []
                st.warning("Não foi possível determinar as características de entrada para a visualização.")

        # Certifique-se que TARGET_ORIGINAL_NAME está definida antes de tentar removê-la
        if 'TARGET_ORIGINAL_NAME' in globals() and TARGET_ORIGINAL_NAME in feature_options_dist:
                feature_options_dist.remove(TARGET_ORIGINAL_NAME)


        if len(feature_options_dist) == 0: # Se não houver features de input (excluindo target) para plotar
            st.warning("Não há características de entrada selecionáveis para visualizar distribuições.")
        else: # Há features para selecionar e plotar.
                selected_feature_dist = st.selectbox(
                    "Selecione uma característica para visualizar a sua distribuição:",
                    options=feature_options_dist,
                    key="selected_feature_dist_eda" # Chave única para o selectbox.
                )

                # Radio buttons para escolher o modo de visualização: Geral vs Comparado.
                # Certifique-se que TARGET_ORIGINAL_NAME está definida para o rótulo
                target_comparison_label = f"Comparar por Situação Final ('{TARGET_ORIGINAL_NAME}')" if 'TARGET_ORIGINAL_NAME' in globals() else "Comparar por Situação Final"
                view_option = st.radio(
                    f"Como visualizar a distribuição de '{selected_feature_dist.replace('_', ' ').title()}':", # PT-PT.
                    ["Distribuição Geral", target_comparison_label], # PT-PT.
                    horizontal=True,
                    key=f"view_option_{selected_feature_dist}" # Chave única.
                )

                # --- Plotagem baseada no tipo de visualização (Geral vs Comparada) e tipo de dado (Ordinal, Numérica, Categórica) ---
                # Reutilizando a lógica refatorada e traduzida, garantindo uso do `df_eda` e rótulos PT-PT.

                # df_eda é a cópia do dataframe original. Removemos NaNs apenas para a plotagem específica para não interferir com contagens/estats descritivas.
                # NaNs handling within plotting blocks for dropna.

                if view_option == "Distribuição Geral": # Plotagem de distribuição individual.
                    st.write(f"### Distribuição Geral de **{selected_feature_dist.replace('_', ' ').title()}**")
                    dtype = df_eda[selected_feature_dist].dtype # Get dtype from original copy.

                    # Casos para tipo de dado... (mantidos da refatoração anterior em PT-PT, já robustos a NaN e mapeamentos)
                    # Categórica Ordinal Mapeada (Bar Chart)
                    # Certifique-se que ordinal_numeric_features_to_map e ORDINAL_MAPPINGS estão definidos
                    if 'ordinal_numeric_features_to_map' in globals() and selected_feature_dist in ordinal_numeric_features_to_map and \
                    'ORDINAL_MAPPINGS' in globals() and selected_feature_dist in ORDINAL_MAPPINGS:
                        st.write("(_Interpretado com rótulos descritivos da escala_):")
                        mapping_dict = ORDINAL_MAPPINGS[selected_feature_dist]
                        # Plotly `values` and `names` arguments work directly with pandas Series/DataFrame columns.
                        # Criar DataFrame para a plotagem (com dropna e mapeamento)
                        df_plot_mapped = df_eda[[selected_feature_dist]].dropna().copy() # Subset e dropna
                        df_plot_mapped['Rótulo'] = df_plot_mapped[selected_feature_dist].map(mapping_dict).fillna('Valor Desconhecido') # Mapeia na cópia
                        counts_df = df_plot_mapped['Rótulo'].value_counts().reset_index() # value_counts na coluna mapeada
                        counts_df.columns = ['Rótulo', 'Contagem']

                        # Order labels for the plot axis (crucial for ordinals).
                        ordered_labels = [mapping_dict.get(k) for k in sorted(mapping_dict.keys()) if mapping_dict.get(k) in counts_df['Rótulo'].tolist()] # Ensure labels from mapping exist in data.
                        if 'Valor Desconhecido' in counts_df['Rótulo'].tolist() and 'Valor Desconhecido' not in ordered_labels:
                            # Adiciona "Valor Desconhecido" ao fim se existir e não estiver na ordem mapeada
                            ordered_labels.append('Valor Desconhecido')


                        fig_bar = px.bar(
                            data_frame=counts_df, # <<< CORREÇÃO AQUI
                            x='Rótulo',
                            y='Contagem',
                            title=f'Distribuição de "{selected_feature_dist.replace("_", " ").title()}"',
                            text_auto=True,
                            category_orders={"Rótulo": ordered_labels}
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)
                        st.info(f"Este gráfico de barras mostra a frequência de cada nível para '{selected_feature_dist.replace('_', ' ').title()}'.")

                    # Numérica (Histogram + Boxplot)
                    elif pd.api.types.is_numeric_dtype(dtype):
                        st.write("(_Dados numéricos contínuos ou de contagem_):")
                        fig_hist = px.histogram(
                            data_frame=df_eda.dropna(subset=[selected_feature_dist]), # <<< CORREÇÃO AQUI
                            x=selected_feature_dist,
                            marginal="box",
                            title=f'Distribuição de "{selected_feature_dist.replace("_", " ").title()}"'
                        ) # Drop NaNs for plotting.
                        st.plotly_chart(fig_hist, use_container_width=True)
                        st.info(f"Histograma e Box Plot para '{selected_feature_dist.replace('_', ' ').title()}'.")

                    # Categórica (Object/String or pd.Categorical) (Bar Chart)
                    elif dtype == 'object' or pd.api.types.is_categorical_dtype(dtype): # Use dtype not df_eda[feature].dtype
                        st.write("(_Dados categóricos nominais ou binários_):")
                        counts_df = df_eda[selected_feature_dist].value_counts().reset_index()
                        counts_df.columns = [selected_feature_dist, 'Contagem']
                        fig_bar = px.bar(
                            data_frame=counts_df, # <<< CORREÇÃO AQUI
                            x=selected_feature_dist,
                            y='Contagem',
                            title=f'Distribuição de "{selected_feature_dist.replace("_", " ").title()}"',
                            text_auto=True
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)
                        st.info(f"Frequência de cada categoria para '{selected_feature_dist.replace('_', ' ').title()}'.")

                    else: st.info(f"Tipo de dado ({dtype}) não suportado para esta visualização.")


                elif view_option == target_comparison_label: # Plotagem comparativa com a target.
                    # Certifique-se que TARGET_ORIGINAL_NAME e CLASS_NAMES estão definidos
                    if 'TARGET_ORIGINAL_NAME' not in globals() or TARGET_ORIGINAL_NAME not in df_eda.columns or 'CLASS_NAMES' not in globals() or len(CLASS_NAMES) < 2:
                        st.warning(f"A coluna alvo '{TARGET_ORIGINAL_NAME if 'TARGET_ORIGINAL_NAME' in globals() else 'variável alvo'}' não foi encontrada ou as classes esperadas ({CLASS_NAMES if 'CLASS_NAMES' in globals() else 'não definidas'}) não estão configuradas corretamente.")
                    else:
                        st.write(f"### Comparação da Distribuição de **{selected_feature_dist.replace('_', ' ').title()}** por **Situação Final**")
                        # Warning if target has unexpected values.
                        target_unique_values_check = df_eda[TARGET_ORIGINAL_NAME].dropna().unique().tolist()
                        # O check original está bom, mas o set(CLASS_NAMES) precisa de existir
                        if not set(target_unique_values_check).issubset(set(CLASS_NAMES)):
                                st.warning(f"A coluna alvo '{TARGET_ORIGINAL_NAME}' parece conter valores inesperados ({target_unique_values_check}) para coloração ('{CLASS_NAMES[0]}', '{CLASS_NAMES[1]}').")


                    # Cria uma CÓPIA do DataFrame, dropando NaNs nas colunas relevantes (feature, target).
                    # Crucial para os groupby/counts subsequentes e Plotly.
                    df_plot_compare = df_eda.dropna(subset=[selected_feature_dist, TARGET_ORIGINAL_NAME]).copy()

                    # Verifica se o DataFrame resultante não está vazio após dropna
                    if df_plot_compare.empty:
                            st.warning(f"Não há dados suficientes para plotar a comparação entre '{selected_feature_dist.replace('_', ' ').title()}' e '{TARGET_ORIGINAL_NAME}'. (Talvez muitos valores nulos?)")
                    else:
                        dtype = df_plot_compare[selected_feature_dist].dtype # Get dtype from the clean subset for safety in plotting

                        # Define o mapa de cores para a target (assegurando que CLASS_NAMES existe)
                        color_discrete_map_target = {CLASS_NAMES[0]: 'salmon', CLASS_NAMES[1]: 'lightgreen'} if 'CLASS_NAMES' in globals() and len(CLASS_NAMES) >= 2 else None


                        # Casos para tipo de dado + comparação... (mantidos da refatoração anterior em PT-PT)
                        # Categórica Ordinal Mapeada (Grouped Bar Chart by Target)
                        if 'ordinal_numeric_features_to_map' in globals() and selected_feature_dist in ordinal_numeric_features_to_map and \
                            'ORDINAL_MAPPINGS' in globals() and selected_feature_dist in ORDINAL_MAPPINGS:
                                st.write("(_Comparação para dados ordinais com rótulos_):")
                                mapping_dict = ORDINAL_MAPPINGS[selected_feature_dist]
                                # Mapear na CÓPIA df_plot_compare ANTES de agrupar
                                df_plot_compare['Feature_Label'] = df_plot_compare[selected_feature_dist].map(mapping_dict).fillna('Valor Desconhecido')
                                # Count occurrences of Feature_Label and Target values.
                                comparison_counts = df_plot_compare.groupby(['Feature_Label', TARGET_ORIGINAL_NAME]).size().reset_index(name='Contagem')
                                # Ensure all ordinal levels appear on the X axis even if count=0 for a target class.
                                all_labels = [mapping_dict.get(k) for k in sorted(mapping_dict.keys())] # Ordered list of possible mapped labels.
                                if 'Valor Desconhecido' in df_plot_compare['Feature_Label'].unique() and 'Valor Desconhecido' not in all_labels: all_labels.append('Valor Desconhecido')
                                # Cria multi index com TODAS as combinações de labels e target classes
                                multi_index = pd.MultiIndex.from_product([all_labels, CLASS_NAMES], names=['Feature_Label', TARGET_ORIGINAL_NAME])
                                # Reindexa a contagem para incluir combinações com 0 contagem
                                comparison_counts_full = comparison_counts.set_index(['Feature_Label', TARGET_ORIGINAL_NAME]).reindex(multi_index, fill_value=0).reset_index() # Reindex with all possible combinations, fill 0 for missing.

                                fig_comp_bar = px.bar(
                                    data_frame=comparison_counts_full, # <<< CORREÇÃO AQUI
                                    x='Feature_Label',
                                    y='Contagem',
                                    color=TARGET_ORIGINAL_NAME,
                                    title=f'Distribuição de "{selected_feature_dist.replace("_", " ").title()}" por Situação Final',
                                    barmode='group',
                                    color_discrete_map=color_discrete_map_target,
                                    category_orders={"Feature_Label": [mapping_dict.get(k) for k in sorted(mapping_dict.keys())] if selected_feature_dist in ordinal_numeric_features_to_map else None} # Use ordered labels for x axis.
                                )
                                fig_comp_bar.update_layout(legend_title_text=TARGET_ORIGINAL_NAME.replace('_', ' ').title())
                                st.plotly_chart(fig_comp_bar, use_container_width=True)
                                st.info(f"Compara contagem de Passou/Não Passou em cada nível de '{selected_feature_dist.replace('_', ' ').title()}'.")

                        # Numérica (Box Plot por grupo Target)
                        elif pd.api.types.is_numeric_dtype(dtype):
                                st.write("(_Comparação para dados numéricos_):")
                                fig_comp_box = px.box(
                                    data_frame=df_plot_compare, # <<< CORREÇÃO AQUI
                                    x=TARGET_ORIGINAL_NAME,
                                    y=selected_feature_dist,
                                    title=f'Distribuição de "{selected_feature_dist.replace("_", " ").title()}" por Situação Final',
                                    color=TARGET_ORIGINAL_NAME,
                                    color_discrete_map=color_discrete_map_target,
                                    points="all"
                                )
                                fig_comp_box.update_layout(legend_title_text=TARGET_ORIGINAL_NAME.replace('_', ' ').title())
                                st.plotly_chart(fig_comp_box, use_container_width=True)
                                st.info(f"Compara a distribuição (medianas, quartis, outliers) de '{selected_feature_dist.replace('_', ' ').title()}' para alunos que Passaram vs Não Passaram.")

                        # Categórica (Grouped Bar Chart by Target)
                        elif dtype == 'object' or pd.api.types.is_categorical_dtype(dtype):
                                st.write("(_Comparação para dados categóricos_):")
                                comparison_counts = df_plot_compare.groupby([selected_feature_dist, TARGET_ORIGINAL_NAME]).size().reset_index(name='Contagem')
                                # Ensure all combinations appear even if count=0
                                all_feature_levels = df_plot_compare[selected_feature_dist].unique().tolist()
                                multi_index_cat = pd.MultiIndex.from_product([all_feature_levels, CLASS_NAMES], names=[selected_feature_dist, TARGET_ORIGINAL_NAME])
                                comparison_counts_full_cat = comparison_counts.set_index([selected_feature_dist, TARGET_ORIGINAL_NAME]).reindex(multi_index_cat, fill_value=0).reset_index()


                                fig_comp_bar = px.bar(
                                    data_frame=comparison_counts_full_cat, # <<< CORREÇÃO AQUI (usando full para 0s)
                                    x=selected_feature_dist,
                                    y='Contagem',
                                    color=TARGET_ORIGINAL_NAME,
                                    title=f'Distribuição de "{selected_feature_dist.replace("_", " ").title()}" por Situação Final',
                                    text_auto=True,
                                    barmode='group',
                                    color_discrete_map=color_discrete_map_target
                                )
                                fig_comp_bar.update_layout(legend_title_text=TARGET_ORIGINAL_NAME.replace('_', ' ').title())
                                st.plotly_chart(fig_comp_bar, use_container_width=True)
                                st.info(f"Compara contagem de Passou/Não Passou para cada categoria de '{selected_feature_dist.replace('_', ' ').title()}'.")

                        else:
                                st.info(f"Tipo de dado ({dtype}) não suportado para comparação com a situação final neste momento.")

            # Fim da lógica de visualização de distribuições.


        with tab_relations: # Relações entre pares de características (Scatter plots, Box plots, Barras) e Correlação.
            st.markdown('<h2 class="sub-header">Relações entre Características</h2>', unsafe_allow_html=True)
            st.markdown('<p class="info-text">Analise a relação entre pares de características no seu dataset original e explore as matrizes de correlação linear.</p>', unsafe_allow_html=True)

            st.markdown('### Visualização de Relações entre Pares de Características', unsafe_allow_html=True)
            # Certifique-se que TARGET_ORIGINAL_NAME está definida para o rótulo
            target_relation_info = f' ("{TARGET_ORIGINAL_NAME.replace("_", " ").title()}")' if 'TARGET_ORIGINAL_NAME' in globals() else ""
            st.markdown(f'<p class="info-text">Selecione duas características para visualizar sua relação. Se a coluna alvo{target_relation_info} existir no dataset, o gráfico será colorido pela situação final do aluno para facilitar a identificação de padrões.</p>', unsafe_allow_html=True) # Clarificado PT-PT.


            # Obtém lista de características de input para seleção (excluindo a target original).
            # Usa `original_cols` se disponível para consistência.
            if 'original_cols' in globals() and original_cols is not None:
                 all_features_options_for_rel = original_cols[:] # Use copy
            elif 'df_eda' in locals():
                 all_features_options_for_rel = df_eda.columns.tolist()
            else:
                 all_features_options_for_rel = []
                 st.warning("Não foi possível determinar as características de entrada para a visualização de relações.")

            # Certifique-se que TARGET_ORIGINAL_NAME está definida antes de tentar removê-la
            if 'TARGET_ORIGINAL_NAME' in globals() and TARGET_ORIGINAL_NAME in all_features_options_for_rel:
                all_features_options_for_rel.remove(TARGET_ORIGINAL_NAME) # Remove target.

            if len(all_features_options_for_rel) < 2:
                 st.warning("São necessárias pelo menos 2 características (excluindo a variável alvo, se definida) para visualizar relações entre pares.")
            else:
                # Selectboxes para escolher as duas características a plotar.
                col_select_rel1, col_select_rel2 = st.columns(2)
                with col_select_rel1:
                    feature1 = st.selectbox("Selecione a Característica 1 (Eixo X):", all_features_options_for_rel, index=0, key="rel_feature1_eda")
                with col_select_rel2:
                    options_feature2 = [col for col in all_features_options_for_rel if col != feature1] # Feature 2 não pode ser a mesma que Feature 1.
                    # Default index for Feature 2 selection box.
                    default_index_feature2 = 0 # Primeiro elemento das opções restantes como default.
                    # Try to select the second original feature if available and different from feature1
                    # Certifique-se que original_cols existe antes de a usar
                    original_cols_check = original_cols if 'original_cols' in globals() and original_cols is not None else df_eda.columns.tolist()
                    if len(original_cols_check) > 1 and \
                       feature1 == original_cols_check[0] and \
                       len(options_feature2) > 0: # Only if Feature 1 is the very first original col AND there are other options left.
                        second_original_feature_candidate = original_cols_check[1]
                        if second_original_feature_candidate in options_feature2: # If the second original feature is available as option for feature 2
                             default_index_feature2 = options_feature2.index(second_original_feature_candidate)
                    # Garante que options_feature2 não está vazia antes de usar index
                    if options_feature2:
                         feature2 = st.selectbox("Selecione a Característica 2 (Eixo Y / Cor / Faceta):", options_feature2, index=min(default_index_feature2, len(options_feature2)-1), key="rel_feature2_eda")
                    else:
                         feature2 = None
                         st.warning("Não há segunda característica disponível para seleção.")


                # Só procede se feature2 foi selecionada (lista options_feature2 não estava vazia)
                if feature2:
                    # Define a coluna usada para colorir (Target Original se existir) ou None.
                    color_col = TARGET_ORIGINAL_NAME if 'TARGET_ORIGINAL_NAME' in globals() and TARGET_ORIGINAL_NAME in df_eda.columns else None

                    # Warning if target values for coloring are not standard ('não'/'sim').
                    # Certifique-se que CLASS_NAMES existe antes de verificar subset
                    if color_col and 'CLASS_NAMES' in globals() and len(CLASS_NAMES) >= 2:
                        target_unique_values_check = df_eda[color_col].dropna().unique().tolist()
                        if not set(target_unique_values_check).issubset(set(CLASS_NAMES)):
                              st.warning(f"A coluna alvo '{color_col}' usada para colorir parece conter valores inesperados ({target_unique_values_check}). A coloração no gráfico pode não ser clara ou funcionar correctamente (esperado: '{CLASS_NAMES[0]}', '{CLASS_NAMES[1]}').")
                    elif color_col: # Target existe, mas CLASS_NAMES não definido ou incompleto.
                         st.warning(f"A coluna alvo '{color_col}' existe, mas CLASS_NAMES não está definido ou incompleto. A coloração do gráfico pode usar valores padrão do Plotly.")


                    # Remove linhas com NaNs nas colunas essenciais ANTES de plotar. Cria uma CÓPIA.
                    cols_to_dropna_relation = [feature1, feature2]
                    if color_col is not None: # Se a cor for a target, também a incluímos no dropna.
                        cols_to_dropna_relation.append(color_col)
                    df_plot_relation = df_eda.dropna(subset=cols_to_dropna_relation).copy() # Limpa dados para a plotagem específica.

                    # Verifica se o DataFrame resultante não está vazio após dropna
                    if df_plot_relation.empty:
                         st.warning(f"Não há dados suficientes para plotar a relação entre '{feature1.replace('_', ' ').title()}' e '{feature2.replace('_', ' ').title()}'. (Talvez muitos valores nulos nas características selecionadas ou na coluna alvo?)")
                    else:
                        dtype1 = df_plot_relation[feature1].dtype # Tipos no DataFrame limpo.
                        dtype2 = df_plot_relation[feature2].dtype

                        # Classificação dos tipos de dado para determinar o melhor tipo de gráfico.
                        # Numérico vs Categórico/Ordinal (considerando mapeamentos).
                        # Certifique-se que ordinal_numeric_features_to_map existe
                        ordinal_map_features = ordinal_numeric_features_to_map if 'ordinal_numeric_features_to_map' in globals() else []
                        is_numeric1 = pd.api.types.is_numeric_dtype(dtype1) and feature1 not in ordinal_map_features
                        is_numeric2 = pd.api.types.is_numeric_dtype(dtype2) and feature2 not in ordinal_map_features
                        is_ordinal_or_categorical1 = (feature1 in ordinal_map_features) or (dtype1 == 'object') or pd.api.types.is_categorical_dtype(dtype1)
                        is_ordinal_or_categorical2 = (feature2 in ordinal_map_features) or (dtype2 == 'object') or pd.api.types.is_categorical_dtype(dtype2)

                        # Define o mapa de cores para a target (assegurando que CLASS_NAMES existe)
                        color_discrete_map_target = {CLASS_NAMES[0]: 'salmon', CLASS_NAMES[1]: 'lightgreen'} if 'CLASS_NAMES' in globals() and len(CLASS_NAMES) >= 2 else None


                        # --- Geração de Gráficos baseada na combinação dos tipos de dados das features ---

                        # Caso 1: Ambas Numéricas (Scatter Plot)
                        if is_numeric1 and is_numeric2:
                            st.write(f"#### Gráfico de Dispersão: {feature1.replace('_', ' ').title()} vs {feature2.replace('_', ' ').title()}")
                            st.info("Gráfico de dispersão para duas características numéricas. Pontos são alunos. Colorido pela Situação Final para ver agrupamentos.")
                            fig = px.scatter(
                                data_frame=df_plot_relation, # <<< CORREÇÃO AQUI
                                x=feature1, y=feature2,
                                color=color_col, # Usa a coluna target original se color_col não é None.
                                # Aplica o mapa de cores PT-PT SÓ SE estiver a colorir PELA TARGET e CLASS_NAMES existir e ter 2 elementos.
                                color_discrete_map=color_discrete_map_target if color_col == TARGET_ORIGINAL_NAME else None, # Correção no mapa de cores.
                                title=f"Dispersão: {feature1.replace('_', ' ').title()} vs {feature2.replace('_', ' ').title()}",
                                opacity=0.7,
                                hover_data=[feature1, feature2] + ([color_col] if color_col else []), # Adiciona target ao hover data só se usada para color.
                            )
                            if color_col: # Atualiza título da legenda se colorida pela target.
                                 fig.update_layout(legend_title_text=color_col.replace('_', ' ').title())
                            st.plotly_chart(fig, use_container_width=True)

                        # Caso 2: Numérico (Y) vs Categórico/Ordinal (X) (Box Plot por X category, com cor por Target se aplicável)
                        elif is_ordinal_or_categorical1 and is_numeric2: # AQUI X é Feature1, Y é Feature2.
                            st.write(f"#### Distribuição de {feature2.replace('_', ' ').title()} por Níveis de {feature1.replace('_', ' ').title()}")
                            st.info(f"Box plots da característica numérica **{feature2.replace('_', ' ').title()}** para cada nível de **{feature1.replace('_', ' ').title()}** (eixo X). Se colorido pela Situação Final, compara distribuições dentro de cada nível.")

                            # feature1 é o X categórico/ordinal
                            x_col_plot_label = feature1 # Nome da coluna usada no eixo X.
                            category_orders = None # Ordem dos elementos no eixo X.

                            # Certifique-se que ORDINAL_MAPPINGS existe
                            if feature1 in ordinal_map_features and 'ORDINAL_MAPPINGS' in globals() and feature1 in ORDINAL_MAPPINGS: # Se a coluna do eixo X é ordinal mapeada (número->rótulo).
                                 # Mapear valores numéricos para rótulos textuais NA CÓPIA `df_plot_relation`.
                                 df_plot_relation[feature1] = df_plot_relation[feature1].map(ORDINAL_MAPPINGS[feature1]).fillna('Valor Desconhecido') # Map, fillna handles numbers not in mapping and NaNs.
                                 # Preparar ordem para o eixo X, baseada nos rótulos mapeados em ordem ordinal esperada.
                                 ordered_labels_x = [ORDINAL_MAPPINGS[feature1].get(k) for k in sorted(ORDINAL_MAPPINGS[feature1].keys())]
                                 if 'Valor Desconhecido' in df_plot_relation[feature1].unique() and 'Valor Desconhecido' not in ordered_labels_x: ordered_labels_x.append('Valor Desconhecido') # Add NaN mapped label if needed.
                                 category_orders = {feature1: ordered_labels_x} # Order applies to the column name used in px.box x=.

                            fig = px.box(
                                data_frame=df_plot_relation, # <<< CORREÇÃO AQUI
                                x=feature1, y=feature2, # feature1 name is used on X axis, feature2 values on Y axis.
                                color=color_col, # Use target column if not None for coloring.
                                color_discrete_map=color_discrete_map_target if color_col == TARGET_ORIGINAL_NAME else None,
                                title=f'Distribuição de "{feature2.replace('_', ' ').title()}" por "{feature1.replace('_', ' ').title()}"',
                                category_orders=category_orders, # Apply explicit order for x axis if set.
                                points=False # Do not show individual points on the box plots for clarity.
                            )
                            if color_col: # Update legend title if colored.
                                 fig.update_layout(legend_title_text=color_col.replace('_', ' ').title())
                            st.plotly_chart(fig, use_container_width=True)


                        # Caso 3: Categórico/Ordinal (X) vs Numérico (Y) (Box Plot por X category, com cor por Target se aplicável)
                        elif is_numeric1 and is_ordinal_or_categorical2: # AQUI X é Feature2, Y é Feature1.
                             st.write(f"#### Distribuição de {feature1.replace('_', ' ').title()} por Níveis de {feature2.replace('_', ' ').title()}")
                             st.info(f"Box plots da característica numérica **{feature1.replace('_', ' ').title()}** para cada nível de **{feature2.replace('_', ' ').title()}** (eixo X). Se colorido pela Situação Final, compara distribuições dentro de cada nível.")

                             # feature2 é o X categórico/ordinal
                             x_col_plot_label = feature2 # Nome da coluna usada no eixo X.
                             category_orders = None
                             # Certifique-se que ORDINAL_MAPPINGS existe
                             if feature2 in ordinal_map_features and 'ORDINAL_MAPPINGS' in globals() and feature2 in ORDINAL_MAPPINGS: # Se a coluna do eixo X é ordinal mapeada.
                                  df_plot_relation[feature2] = df_plot_relation[feature2].map(ORDINAL_MAPPINGS[feature2]).fillna('Valor Desconhecido')
                                  ordered_labels_x = [ORDINAL_MAPPINGS[feature2].get(k) for k in sorted(ORDINAL_MAPPINGS[feature2].keys())]
                                  if 'Valor Desconhecido' in df_plot_relation[feature2].unique() and 'Valor Desconhecido' not in ordered_labels_x: ordered_labels_x.append('Valor Desconhecido')
                                  category_orders = {feature2: ordered_labels_x}

                             fig = px.box(
                                 data_frame=df_plot_relation, # <<< CORREÇÃO AQUI
                                 x=feature2, y=feature1, # Feature2 on X, Feature1 on Y.
                                 color=color_col,
                                 color_discrete_map=color_discrete_map_target if color_col == TARGET_ORIGINAL_NAME else None,
                                 title=f'Distribuição de "{feature1.replace('_', ' ').title()}" por "{feature2.replace('_', ' ').title()}"',
                                 category_orders=category_orders,
                                 points=False
                             )
                             if color_col:
                                 fig.update_layout(legend_title_text=color_col.replace('_', ' ').title())
                             st.plotly_chart(fig, use_container_width=True)

                        # Caso 4: Ambas Categóricas/Ordinais (Grouped Bar Chart or Faceted Bar Chart)
                        elif is_ordinal_or_categorical1 and is_ordinal_or_categorical2:
                            st.write(f"#### Contagem de Alunos por {feature1.replace('_', ' ').title()} vs {feature2.replace('_', ' ').title()}")
                            # Certifique-se que TARGET_ORIGINAL_NAME está definida para o rótulo
                            target_relation_info_title = f" pela Situação Final" if 'TARGET_ORIGINAL_NAME' in globals() and color_col == TARGET_ORIGINAL_NAME else ""
                            st.info(f"Contagem de alunos por combinação de níveis de **{feature1.replace('_', ' ').title()}** e **{feature2.replace('_', ' ').title()}**. Se a Situação Final existir e for usada para faceta, gráfico separado (facetado) para alunos que Passaram/Não Passaram.")

                            # Mapear ambas features se forem ordinais (para exibir rótulos nos eixos/legenda).
                            # Isto deve ser feito ANTES de agrupar para que os rótulos mapeados sejam as categorias.
                            # Acontece na cópia df_plot_relation.
                            category_orders_dict = {} # Store orders for X axis and color/facet axis.

                            # Feature 1 mapping (for X axis) - Certifique que ORDINAL_MAPPINGS existe
                            if feature1 in ordinal_map_features and 'ORDINAL_MAPPINGS' in globals() and feature1 in ORDINAL_MAPPINGS:
                                 df_plot_relation[feature1] = df_plot_relation[feature1].map(ORDINAL_MAPPINGS[feature1]).fillna('Valor Desconhecido')
                                 ordered_labels_x = [ORDINAL_MAPPINGS[feature1].get(k) for k in sorted(ORDINAL_MAPPINGS[feature1].keys())]
                                 if 'Valor Desconhecido' in df_plot_relation[feature1].unique() and 'Valor Desconhecido' not in ordered_labels_x: ordered_labels_x.append('Valor Desconhecido')
                                 category_orders_dict[feature1] = ordered_labels_x # Store order for Feature 1 name

                            # Feature 2 mapping (for Color/Facet) - Certifique que ORDINAL_MAPPINGS existe
                            if feature2 in ordinal_map_features and 'ORDINAL_MAPPINGS' in globals() and feature2 in ORDINAL_MAPPINGS:
                                 df_plot_relation[feature2] = df_plot_relation[feature2].map(ORDINAL_MAPPINGS[feature2]).fillna('Valor Desconhecido')
                                 ordered_labels_color_facet = [ORDINAL_MAPPINGS[feature2].get(k) for k in sorted(ORDINAL_MAPPINGS[feature2].keys())]
                                 if 'Valor Desconhecido' in df_plot_relation[feature2].unique() and 'Valor Desconhecido' not in ordered_labels_color_facet: ordered_labels_color_facet.append('Valor Desconhecido')
                                 category_orders_dict[feature2] = ordered_labels_color_facet # Store order for Feature 2 name

                            # Define quais colunas usar para color e faceta, baseado na disponibilidade da Target.
                            # Se Target (color_col) existe, usá-la-emos para faceta, e colorimos pela Feature 2.
                            # Se Target não existe (color_col is None), colorimos pela Feature 2, sem faceta.
                            color_plot_col_name = feature2 # Nome da coluna para colorir (feature 2 por defeito).
                            facet_plot_col_name = None    # Nome da coluna para facetas (None por defeito).

                            if color_col == TARGET_ORIGINAL_NAME: # Se a Target Original existe e está disponível para color (entrou no check inicial `if color_col:`).
                                facet_plot_col_name = TARGET_ORIGINAL_NAME # A faceta será pela Target.
                                # A cor agora pode ser a Feature 2 (ou remover cor). Mantemos cor pela feature 2 dentro de cada faceta da target.
                                # color_plot_col_name = feature2 # Esta já é o default acima.
                                # Certifica que CLASS_NAMES existe para a ordem da faceta
                                if 'CLASS_NAMES' in globals() and len(CLASS_NAMES) >= 2:
                                    category_orders_dict[TARGET_ORIGINAL_NAME] = CLASS_NAMES # Order facets by CLASS_NAMES


                            # Contar ocorrências por combinação das colunas relevantes (Eixo X, Coloração, Faceta se aplicável).
                            cols_for_groupby_relation = [feature1, color_plot_col_name] # Mínimo para groupby: Eixo X e a coluna usada para colorir
                            if facet_plot_col_name is not None: # Se estamos a facetar (pela Target Original).
                                cols_for_groupby_relation.append(facet_plot_col_name) # Adiciona a coluna de faceta (Target) ao groupby.
                            # Usa `df_plot_relation` (já sem NaNs).
                            counts_df_relation = df_plot_relation.groupby(cols_for_groupby_relation).size().reset_index(name='Contagem')

                            # Note: Plotly Express `color` argumento agora refere-se à coluna `color_plot_col_name` no `counts_df_relation`.
                            # Se `facet_plot_col_name` é a target, o `color` será `feature2`.
                            # Se `facet_plot_col_name` é None (target não usada para faceta), o `color` será `feature2`.
                            # A única diferença na chamada do px.bar será o `facet_col`.

                            fig = px.bar(
                                data_frame=counts_df_relation, # <<< CORREÇÃO AQUI
                                x=feature1, # Eixo X (Feature 1).
                                y='Contagem',
                                color=color_plot_col_name, # Colorir pela Feature 2 OU pela Target (se for a coluna de cor decidida)
                                facet_col=facet_plot_col_name, # Criar colunas separadas para cada nível da Faceta (Target se color_col=Target).
                                facet_col_wrap=2, # Máximo 2 colunas de facetas.
                                title=f"Contagem por {feature1.replace('_', ' ').title()} vs {feature2.replace('_', ' ').title()}" + target_relation_info_title, # Ajusta título.
                                category_orders=category_orders_dict, # Aplica ordens explicitamente para eixos e cores/facetas.
                                labels={feature1: feature1.replace('_', ' ').title(), color_plot_col_name: color_plot_col_name.replace('_', ' ').title()}, # Define rótulos amigáveis para eixos e legendas/facetas.
                                barmode='group', # Barras agrupadas (uma barra colorida por nível da coluna `color`). Use 'stack' se quiser empilhadas.
                            )

                            # Ajustar títulos das facetas (colunas separadas) se existirem.
                            if facet_plot_col_name is not None:
                                # Certifica que TARGET_ORIGINAL_NAME está definida para o rótulo da faceta
                                facet_title_name = TARGET_ORIGINAL_NAME.replace('_', ' ').title() if 'TARGET_ORIGINAL_NAME' in globals() else facet_plot_col_name.replace('_', ' ').title()
                                fig.for_each_annotation(lambda a: a.update(text=f"{facet_title_name}={a.text.split('=')[-1]}")) # Formata o título da faceta.

                            # Opcional: Ajustar título da legenda (o padrão do px.bar color é bom, mas se colorir pela target, queremos que o título da legenda seja a feature2).
                            # Se `color_plot_col_name` for a Target, a legenda por defeito do plotly será `TARGET_ORIGINAL_NAME`, o que queremos.
                            # Se `color_plot_col_name` for a Feature 2 (e sem faceta), a legenda por defeito do plotly será `feature2`. Queremos formatar isso.
                            if color_plot_col_name != TARGET_ORIGINAL_NAME:
                                 fig.update_layout(legend_title_text=color_plot_col_name.replace('_', ' ').title())

                            st.plotly_chart(fig, use_container_width=True) # Exibe o gráfico.
                            # Explicação do gráfico para o utilizador.
                            # Certifica que TARGET_ORIGINAL_NAME está definida para a explicação
                            target_explanation_name = TARGET_ORIGINAL_NAME.replace('_', ' ').title() if 'TARGET_ORIGINAL_NAME' in globals() else "Situação Final"
                            if facet_plot_col_name == TARGET_ORIGINAL_NAME: # Gráfico facetado pela Target.
                                 st.info(f"""Este gráfico de barras compara a contagem de alunos em cada combinação dos níveis de **{feature1.replace('_', ' ').title()}** (eixo X) e **{feature2.replace('_', ' ').title()}** (cores dentro das barras), separado para cada **{target_explanation_name}** (colunas - facetas).""")
                            else: # Gráfico não facetado.
                                 st.info(f"""Este gráfico de barras mostra a contagem de alunos para cada nível de **{feature1.replace('_', ' ').title()}** (eixo X), subdividido e colorido pelos níveis de **{feature2.replace('_', ' ').title()}**. Mostra a distribuição combinada destas duas características.""")

                     

            st.markdown("---") # Separador.

            st.markdown('### Matriz de Correlação', unsafe_allow_html=True)
            st.markdown('<p class="info-text">Veja a correlação linear entre as características numéricas do seu dataset.</p>', unsafe_allow_html=True)

            # Certifica-se de que usa o DataFrame 'df' da EDA (df_source)
            # Para a matriz principal, ainda é útil ver correlações APENAS entre features
            df_features_only = df_eda[original_cols] if 'original_cols' in locals() and original_cols is not None else df_eda.drop(columns=[TARGET_ORIGINAL_NAME] if TARGET_ORIGINAL_NAME in df_eda.columns else [])
            df_numeric_for_corr_matrix = df_features_only.select_dtypes(include=np.number)

            if df_numeric_for_corr_matrix.empty:
                st.info("Não há colunas numéricas entre as características usadas para calcular a matriz de correlação no seu dataset.")
            else:
                # A função plot_correlation_matrix_px precisa estar definida em outro lugar
                fig_corr, corr_matrix = plot_correlation_matrix_px(df_numeric_for_corr_matrix) # Usa o df sem a target para a matriz principal
                if fig_corr is not None and corr_matrix is not None:
                    st.plotly_chart(fig_corr, use_container_width=True)

                else:
                    st.info("Não há dados numéricos suficientes entre as características originais no seu dataset para calcular a matriz de correlação.")


            # --- Correlação com a Variável Alvo ---
            # ESTE BLOCO PERMANECE FOCADO NA CORRELAÇÃO DAS FEATURES *NUMÉRICAS/BOOLEANAS DE ENTRADA* COM A TARGET ORIGINAL.
            # Não foi alterado, pois é uma análise diferente e específica.

            target_corr_title = f'### Correlação com a Variável Alvo: "{TARGET_ORIGINAL_NAME.replace("_", " ").title()}"' if 'TARGET_ORIGINAL_NAME' in globals() else '### Correlação com a Variável Alvo'
            st.markdown(target_corr_title, unsafe_allow_html=True) # PT-PT.

            target_corr_info = f'"{TARGET_ORIGINAL_NAME.replace("_", " ").title()}"' if 'TARGET_ORIGINAL_NAME' in globals() else "variável alvo"
            class_names_info = f"'{CLASS_NAMES[0]}' → 0, '{CLASS_NAMES[1]}' → 1" if 'CLASS_NAMES' in globals() and len(CLASS_NAMES) >= 2 else "geralmente: Não=0, Sim=1"
            st.markdown(f'<p class="info-text">Veja a correlação linear (Pearson ou Ponto-Bisserial) entre cada característica de entrada numérica e a {target_corr_info}. Se a variável alvo for binária e estiver no formato de texto, será convertida para 0 e 1 para este cálculo ({class_names_info}).</p>', unsafe_allow_html=True) # PT-PT explicação da conversão 0/1.


            # Prepara o DataFrame e a série alvo para o cálculo de correlação com a target.
            # Este bloco *ainda* foca apenas nas features de entrada *numéricas originais* vs a target.
            df_for_target_corr_calc = None # DataFrame contendo as features de input numéricas e a target (em formato numérico se convertida).
            target_col_for_corr_calc = None # A Série numérica (0/1) da target.

            # Verifica se a coluna alvo original existe.
            if 'TARGET_ORIGINAL_NAME' in globals() and TARGET_ORIGINAL_NAME in df_eda.columns:
                target_col_original = df_eda[TARGET_ORIGINAL_NAME]

                # Identifica APENAS as características de input originais (sem a target) que são numéricas ou booleanas.
                # Usa original_cols se disponível para definir o universo de features de *entrada*.
                if 'original_cols' in globals() and original_cols is not None:
                    input_features = [col for col in original_cols if col != TARGET_ORIGINAL_NAME]
                elif 'df_eda' in locals():
                    input_features = [col for col in df_eda.columns.tolist() if col != TARGET_ORIGINAL_NAME]
                else:
                    input_features = [] # Não há features de entrada disponíveis.

                # Seleciona apenas as características de *entrada* que são numéricas ou booleanas no df_eda.
                numeric_or_bool_input_features = df_eda[input_features].select_dtypes(include=[np.number, bool]).columns.tolist()


                if not numeric_or_bool_input_features: # Se não há colunas numéricas/booleanas de entrada.
                    st.info("Não há características de entrada numéricas ou booleanas no dataset original para calcular a correlação com a variável alvo.")
                else: # Há características de input numéricas/booleanas.
                    # Cria um subset do DataFrame com as features de input numéricas/booleanas e a coluna target original.
                    df_subset_for_target_corr_prep = df_eda[numeric_or_bool_input_features + [TARGET_ORIGINAL_NAME]].copy()

                    # --- Lógica para converter a coluna alvo para NUMÉRICO 0/1 se necessário ---
                    is_target_numeric_or_bool = pd.api.types.is_numeric_dtype(target_col_original) or pd.api.types.is_bool_dtype(target_col_original)
                    target_unique_vals = target_col_original.dropna().unique() # Valores únicos não nulos da target.

                    if is_target_numeric_or_bool: # A target original já é numérica/booleana.
                        # Certifica-se que é float para cálculo de correlação.
                        df_for_target_corr_calc = df_subset_for_target_corr_prep.astype(float)
                        target_col_for_corr_calc = df_for_target_corr_calc[TARGET_ORIGINAL_NAME]
                    # Certifique que CLASS_NAMES existe antes de tentar a conversão binária
                    elif 'CLASS_NAMES' in globals() and len(CLASS_NAMES) >= 2: # A target original não é numérica/booleana, mas pode ser binária com strings esperadas ('não', 'sim').
                        if len(target_unique_vals) == 2 and set(target_unique_vals).issubset(set(CLASS_NAMES)): # É binária com strings esperadas.
                            try:
                                    ordered_cat_type = pd.api.types.CategoricalDtype(categories=CLASS_NAMES, ordered=True)
                                    target_col_converted_numeric = target_col_original.astype(ordered_cat_type).cat.codes.astype(float)
                                    df_for_target_corr_calc = df_subset_for_target_corr_prep.copy()
                                    df_for_target_corr_calc[TARGET_ORIGINAL_NAME] = target_col_converted_numeric
                                    target_col_for_corr_calc = target_col_converted_numeric
                            except Exception as cat_e:
                                    st.error(f"❌ Erro ao converter a coluna alvo binária de texto para numérica 0/1: {cat_e}. Não é possível calcular a correlação com a target.")
                                    df_for_target_corr_calc = None
                                    target_col_for_corr_calc = None
                        else: # Não é binária com strings esperadas.
                            st.warning(f"A coluna alvo '{TARGET_ORIGINAL_NAME}' não é numérica e não parece ser binária com os valores esperados ('{CLASS_NAMES[0]}', '{CLASS_NAMES[1]}'). Valores encontrados (não-NaN): {list(target_unique_vals)}. Não é possível calcular correlação linear com ela.")
                            df_for_target_corr_calc = None
                            target_col_for_corr_calc = None
                    else: # Target não é numérica/bool, e CLASS_NAMES não existe/incompleto
                        st.warning(f"A coluna alvo '{TARGET_ORIGINAL_NAME}' não é numérica. A conversão para calcular correlação linear falhou porque CLASS_NAMES não está definido ou completo.")
                        df_for_target_corr_calc = None
                        target_col_for_corr_calc = None


                    # --- Cálculo e Exibição da Correlação ---
                    if df_for_target_corr_calc is not None and target_col_for_corr_calc is not None:
                        try:
                            corr_matrix_with_target = df_for_target_corr_calc.corr()

                            if TARGET_ORIGINAL_NAME in corr_matrix_with_target.columns:
                                    target_correlations_series = corr_matrix_with_target[TARGET_ORIGINAL_NAME].drop(TARGET_ORIGINAL_NAME, errors='ignore')

                                    if not target_correlations_series.empty:
                                        sorted_target_corr_abs = target_correlations_series.abs().sort_values(ascending=False)
                                        ordered_target_correlations = target_correlations_series.loc[sorted_target_corr_abs.index]

                                        corr_df_display = ordered_target_correlations.reset_index()
                                        corr_df_display.columns = ['Característica de Entrada (Numérica)', f'Correlação_com_{TARGET_ORIGINAL_NAME.replace("_", " ").title()}']

                                        st.dataframe(corr_df_display.round(4), use_container_width=True)

                                        target_explanation_name_info = TARGET_ORIGINAL_NAME.replace('_', ' ').title() if 'TARGET_ORIGINAL_NAME' in globals() else "variável alvo"
                                        status_message = f"A tabela acima lista as características de entrada numéricas/booleanas com a maior **correlação linear** com a {target_explanation_name_info} (ordenado por magnitude)."
                                        if not is_target_numeric_or_bool:
                                            if 'CLASS_NAMES' in globals() and len(CLASS_NAMES) >= 2:
                                                if 0 in target_col_for_corr_calc.unique() and 1 in target_col_for_corr_calc.unique():
                                                        status_message += f" (A coluna alvo foi convertida para 0/1: '{CLASS_NAMES[0]}' → {0}, '{CLASS_NAMES[1]}' → {1})."
                                            else:
                                                st.warning("Erro na validação pós-conversão da coluna alvo para 0/1. A interpretação pode ser ambígua.")

                                        st.info(status_message)
                                        if 'CLASS_NAMES' in globals() and len(CLASS_NAMES) >= 2:
                                            st.info(f"Uma correlação positiva alta (próximo de +1) sugere que valores mais altos nesta característica estão associados a **Passar** ('{CLASS_NAMES[1]}'). Correlação negativa alta (próximo de -1) sugere associação a **Não Passar** ('{CLASS_NAMES[0]}').")
                                        else:
                                            st.info("Uma correlação positiva alta (próximo de +1) sugere que valores mais altos nesta característica estão associados ao valor que foi mapeado para 1. Correlação negativa alta (próximo de -1) sugere associação ao valor que foi mapeado para 0.")


                                    else: st.info("Não foi possível calcular correlações. Verifique se há características de entrada numéricas.")

                            else: st.warning(f"Erro interno: Coluna '{TARGET_ORIGINAL_NAME}' não encontrada na matriz de correlação subset.")
                        except Exception as e:
                            st.error(f"❌ Erro ao calcular a correlação com a variável alvo: {e}")
                            st.info("Verifique se as características de entrada numéricas/booleanas e a coluna alvo não contêm valores problemáticos.")


            else: # TARGET_ORIGINAL_NAME not in df_eda.columns
                target_name_missing = TARGET_ORIGINAL_NAME if 'TARGET_ORIGINAL_NAME' in globals() else "variável alvo"
                st.warning(f"A coluna alvo '{target_name_missing}' não foi encontrada no dataset original ou o nome da variável alvo não está definido. Não é possível calcular a correlação com a variável alvo.")

# Fim do bloco de código (dentro do tab_relations)

# --- Secção "Previsão Individual" ---
# Permite introduzir dados para prever o resultado de um aluno único.
# Refatorado com inputs PT-PT, validação básica e correcção de mapeamento de previsão e probabilidades.
elif menu == "Previsão Individual":
    st.markdown('<h1 class="main-header">Previsão de Desempenho Individual</h1>', unsafe_allow_html=True)
    st.markdown('<p class="info-text">Utilize os campos abaixo para introduzir os dados de um aluno e obter uma previsão sobre se este **Passará** ou **Não Passará** no exame final, com base num modelo treinado à sua escolha.</p>', unsafe_allow_html=True)

    # --- Verificação de Dependências Essenciais ---
    # Para esta secção, é crucial ter os nomes das colunas originais, o dataset original (para defaults),
    # o pré-processador e pelo menos um modelo carregado (o melhor, ou outro selecionável).
    # As variáveis `original_cols`, `student_df_original`, `preprocessor` são globais carregadas no início.
    # A lista de modelos selecionáveis `model_files` e `selected_model_instance` são tratadas no sub-bloco de seleção.
    can_run_prediction = True # Flag geral para controlar se a secção é funcional.

    if 'original_cols' not in locals() or original_cols is None or not isinstance(original_cols, (list, pd.Index)):
         st.error("Não foi possível carregar a lista de nomes das características originais ('original_input_columns.joblib'). Os campos de input não podem ser criados.")
         can_run_prediction = False
    elif student_df_original is None:
         st.error("Não foi possível carregar o dataset original ('student-data.csv') necessário para determinar tipos, intervalos e valores por defeito para os campos de input.")
         can_run_prediction = False
    # Verificar se o pré-processador foi carregado. É ESSENCIAL.
    elif 'preprocessor' not in globals() or preprocessor is None:
         st.error("O pré-processador ('preprocessor.joblib') não foi carregado. A aplicação não consegue preparar os dados de input para o modelo.")
         can_run_prediction = False
    # Note: O check pelo modelo selecionável e carregamento `selected_model_instance` é feito DENTRO
    # do bloco principal da secção, pois essa parte tem lógica complexa própria.
    # Se nenhum modelo for selecionável/carregado, o botão de prever estará desativado/indisponível,
    # e uma mensagem aparecerá após os inputs.


    if can_run_prediction: # Só constrói UI de inputs se pré-requisitos ok.
         # Dicionário para armazenar os valores inseridos pelo utilizador.
        input_data = {}

        # Obtém os tipos de dado de cada característica original a partir do dataset carregado.
        original_dtypes = student_df_original.dtypes[original_cols]

        # Separa características em Numéricas (incluindo booleanos como 0/1 implicitamente) e Categóricas.
        numeric_features = [col for col in original_cols if original_dtypes[col] in [np.number, 'int64', 'float64', bool]] # Bool treated numerically or via selectbox.
        categorical_features = [col for col in original_cols if original_dtypes[col] == 'object' or pd.api.types.is_categorical_dtype(original_dtypes[col])]


        # --- Seçcão de Seleção do Modelo ---
        st.markdown("### Seleção do Modelo para Previsão", unsafe_allow_html=True)
        selected_model_instance = None # Será definida aqui.
        selected_model_filename = None # Será definida aqui.
        artefacts_path = 'artefacts/'

        try: # Bloco try/except para o carregamento de modelos.
            # Lista os ficheiros .joblib na pasta artefacts/.
            all_joblib_files = [f for f in os.listdir(artefacts_path) if f.endswith('.joblib')]
            # Filtra para modelos (excluindo artefactos do pipeline fixo).
            model_files = [f for f in all_joblib_files if f not in ['preprocessor.joblib', 'original_input_columns.joblib', 'processed_feature_names.joblib']]
            model_files.sort() # Ordena.

            # Mapeamento manual de nome de ficheiro -> nome amigável para UI.
            filename_to_display_name_map = {
                'best_model.joblib': 'Modelo Principal (Recomendado)', # simplified label
                # Add others if needed, e.g. 'random_forest_final.joblib': 'Random Forest Final',
            }

            # Prepara a lista de opções para o selectbox e o mapeamento inverso.
            model_display_options = []
            model_filename_map = {}
            for f in model_files:
                display_name = filename_to_display_name_map.get(f, f.replace('.joblib', '').replace('_', ' ').title())
                model_display_options.append(display_name)
                model_filename_map[display_name] = f

            # Define a opção por defeito: o "best_model.joblib" se estiver na lista.
            default_model_filename = 'best_model.joblib'
            default_model_display_name_candidate = filename_to_display_name_map.get(default_model_filename, default_model_filename.replace('.joblib', '').replace('_', ' ').title())
            default_model_index = 0 # Default para o 1º.
            if default_model_display_name_candidate in model_display_options:
                 default_model_index = model_display_options.index(default_model_display_name_candidate)

            if len(model_display_options) > 0: # Se houver modelos para selecionar.
                 # Selectbox para escolher o modelo.
                 selected_display_name = st.selectbox(
                    "Escolha o modelo treinado a utilizar para fazer a previsão:",
                    options=model_display_options,
                    index=default_model_index,
                    key="prediction_model_selector"
                 )
                 selected_model_filename = model_filename_map.get(selected_display_name)
                 # Carrega o modelo selecionado.
                 selected_model_instance = load_specific_model(selected_model_filename)

                 if selected_model_instance is None:
                      # Load specific model já exibe mensagem de erro dentro se o ficheiro falhou.
                      st.error(f"❌ Não foi possível carregar o modelo selecionado '{selected_display_name}' ('{selected_model_filename}'). Por favor, verifique o ficheiro.")
                 # Mensagem sobre o modelo principal (global `model`).
                 if 'model' in globals() and model is not None:
                      best_model_type_name = type(model).__name__
                      st.info(f"**Nota:** O modelo guardado como 'best_model.joblib' ({best_model_type_name}) é geralmente o recomendado.")
            else: # Não encontrou nenhum ficheiro joblib válido que pareça ser um modelo.
                 st.warning("Não foram encontrados ficheiros de modelos válidos (.joblib) na pasta 'artefacts/' (excluindo preprocessor/column lists). A previsão individual não pode ser realizada sem um modelo carregado.")


        except FileNotFoundError: st.error("❌ A pasta 'artefacts/' não foi encontrada.")
        except Exception as e: st.error(f"❌ Erro ao listar/carregar modelos em 'artefacts/': {e}")


        # --- Secção de Inputs de Dados ---
        st.markdown("---")
        st.markdown('<h3 class="sub-header">Dados do Aluno</h3>', unsafe_allow_html=True)

        st.markdown("#### Características Numéricas", unsafe_allow_html=True)
        st.markdown('<div class="input-grid-container">', unsafe_allow_html=True)
        for feature in numeric_features:
             st.markdown('<div class="grid-item">', unsafe_allow_html=True)
             # Label customizada com nome e descrição curta.
             description = feature_descriptions_short.get(feature, 'Descrição não disponível')
             st.markdown(f"<div class='feature-label-container'><strong>{feature.replace('_', ' ').title()}</strong> <span class='small-description'>({description})</span></div>", unsafe_allow_html=True)

             # Configurar inputs numéricos (st.number_input). Lida com ints vs floats, defaults.
             dtype = original_dtypes[feature]
             is_integer_type = pd.api.types.is_integer_dtype(dtype) or feature in ordinal_numeric_features_to_map # Considera ordinais numéricas como input int.
             is_boolean_type = dtype == bool # Separa booleanos para selectbox.

             min_val = student_df_original[feature].min()
             max_val = student_df_original[feature].max()
             mean_val = student_df_original[feature].mean()

             # Handling for boolean feature: use selectbox 'Sim'/'Não'.
             if is_boolean_type:
                  # Define as opções e o valor padrão ('Sim'/'Não').
                  bool_options_ui = ['Não', 'Sim'] # Ordem 0, 1 para mapear para False, True.
                  default_bool_value = bool_options_ui[1] if not student_df_original[feature].mode().empty and student_df_original[feature].mode()[0] is True else bool_options_ui[0]
                  selected_bool_label = st.selectbox(label="", options=bool_options_ui, index=bool_options_ui.index(default_bool_value), key=f"input_bool_{feature}")
                  # Guarda o valor Booleano real no dicionário.
                  input_data[feature] = True if selected_bool_label == 'Sim' else False
                  if feature in ['internet', 'romantic', 'higher']: st.info(f"'Não': Não possui. 'Sim': Possui.") # Info específica para alguns booleanos. # Add others as needed

             else: # Numeric (int or float).
                  input_min = int(min_val) if pd.notna(min_val) else 0 if is_integer_type else 0.0
                  input_max = int(max_val) if pd.notna(max_val) else (None if is_integer_type else None) # None max allowed.
                  input_value = int(round(mean_val)) if pd.notna(mean_val) else (input_min if pd.notna(input_min) else (0 if is_integer_type else 0.0)) # Default to mean (rounded for int) or min.
                  input_step = 1 if is_integer_type else 0.01
                  input_format = "%d" if is_integer_type else "%.2f"

                  # Se ordinal numérico, exibe os níveis num info box.
                  if feature in ordinal_numeric_features_to_map:
                      mapping_dict = ORDINAL_MAPPINGS[feature]
                      mapping_desc = ", ".join([f"{k}: {v}" for k, v in mapping_dict.items()])
                      st.info(mapping_desc)

                  # Cria o Number Input.
                  input_data[feature] = st.number_input(label="", min_value=input_min, max_value=input_max, value=input_value, step=input_step, format=input_format, key=f"input_numeric_{feature}")

             st.markdown('</div>', unsafe_allow_html=True) # Fecha o grid item.

        st.markdown('</div>', unsafe_allow_html=True) # Fecha grid numéricos.


        st.markdown("#### Características Categóricas Nominais", unsafe_allow_html=True) # Sub-sub-cabeçalho PT-PT.
        st.markdown('<div class="input-grid-container">', unsafe_allow_html=True) # Abre grid categóricos.
        for feature in categorical_features:
             st.markdown('<div class="grid-item">', unsafe_allow_html=True)
             description = feature_descriptions_short.get(feature, 'Descrição não disponível')
             st.markdown(f"<div class='feature-label-container'><strong>{feature.replace('_', ' ').title()}</strong> <span class='small-description'>({description})</span></div>", unsafe_allow_html=True)

             # Selectbox para categóricas não booleanas. Obtém opções e default do dataset original.
             options = student_df_original[feature].dropna().unique().tolist()
             if not options:
                  st.warning(f"Sem opções disponíveis para a característica '{feature}'. Pode estar vazia ou com apenas NaNs.")
                  input_data[feature] = None
             else:
                 default_index = 0
                 try: # Tenta usar o mode como default.
                     mode_value = student_df_original[feature].mode()
                     if not mode_value.empty and mode_value[0] in options:
                          default_index = options.index(mode_value[0])
                 except: pass
                 input_data[feature] = st.selectbox(label="", options=options, index=default_index, key=f"input_categorical_{feature}")

             st.markdown('</div>', unsafe_allow_html=True) # Fecha grid item.

        st.markdown('</div>', unsafe_allow_html=True) # Fecha grid categóricos.


        st.markdown("---") # Separador antes do botão Prever.

        # --- Botão Executar Previsão ---
        # O botão aparece e está funcional SOMENTE se o modelo e pré-processador foram carregados
        # e os nomes das colunas originais estão disponíveis.
        if selected_model_instance is not None and preprocessor is not None and original_cols is not None and student_df_original is not None:
            # Última verificação se as colunas originais (de onde recolhemos input) correspondem
            # à lista de original_cols que o preprocessor espera.
            if len(input_data) != len(original_cols):
                st.warning("Número de campos de input recolhidos não corresponde ao número de características originais esperadas. Pode haver um problema na identificação de características ou na lista `original_cols`.")
                # Continua, mas com um aviso.

            if st.button("🚀 Obter Previsão para Este Aluno", key="run_prediction_button_exec"): # PT-PT botão.

                loading_animation(f"Aplicando pré-processamento e prevendo com '{selected_model_filename}'...") # Animação PT-PT

                try: # Bloco try para a execução da previsão.

                    # --- Preparar Input como DataFrame para o Pré-processador ---
                    # Cria um DataFrame com 1 linha usando `original_cols` como nome das colunas
                    # na ordem correta esperada pelo preprocessor.
                    input_df_raw = pd.DataFrame(columns=original_cols)
                    # Preenche a primeira linha com os dados recolhidos.
                    for col in original_cols: # Itera sobre original_cols para GARANTIR a ordem.
                         if col in input_data: # Pega o valor do dicionário.
                             input_df_raw.loc[0, col] = input_data[col]
                         else:
                             # Adicionar um valor placeholder se alguma original_col não foi mapeada para um input UI.
                             # Dependendo do pré-processador, isto pode causar problemas ou ser OK (fillna).
                             st.warning(f"Característica original '{col}' não foi mapeada para um campo de input UI. Adicionando pd.NA.")
                             input_df_raw.loc[0, col] = pd.NA


                    st.markdown("#### Dados de Entrada Formatados para Processamento:", unsafe_allow_html=True)
                    st.dataframe(input_df_raw, use_container_width=True)

                    # Validação final das colunas: DEVE CORRESPONDER EXACTAMENTE à lista `original_cols`.
                    if list(input_df_raw.columns) != list(original_cols):
                         error_msg_cols = "❌ Erro de compatibilidade fatal: As colunas do DataFrame de input para previsão não correspondem à lista de colunas originais carregadas ('original_input_columns.joblib'). O pré-processamento não pode ser executado."
                         st.error(error_msg_cols)
                         raise ValueError(error_msg_cols) # Stop prediction process.


                    # --- Aplicar Pré-processamento ---
                    # Utiliza o pré-processador global.
                    input_processed = preprocessor.transform(input_df_raw)
                    st.success("✅ Pré-processamento aplicado com sucesso.")

                    # --- Obter Previsão e Probabilidades ---
                    # Utiliza o modelo selecionado (`selected_model_instance`).
                    prediction_result = selected_model_instance.predict(input_processed)
                    predicted_value = prediction_result[0] # O valor bruto previsto (ex: 0, 1, ou label string).

                    y_proba_input = None
                    if hasattr(selected_model_instance, 'predict_proba'): # Verifica se o modelo suporta probabilidades.
                         try: y_proba_input = selected_model_instance.predict_proba(input_processed)
                         except Exception as proba_e: st.info(f"Não foi possível obter probabilidades ({proba_e}).")


                    # --- Interpretar a Previsão Bruta para Label Amigável ---
                    # Mapeia o valor previsto (`predicted_value`) para uma das labels amigáveis em `CLASS_NAMES` ('não', 'sim').
                    # Requer que o modelo tenha o atributo `.classes_` que mapeia os valores brutos para a ordem das classes.
                    predicted_class_label = "Valor Previsto Desconhecido" # Fallback default.

                    if hasattr(selected_model_instance, 'classes_') and len(CLASS_NAMES) == 2: # Assume modelo binário compatível com CLASS_NAMES.
                         model_classes_list = list(selected_model_instance.classes_) # Lista de classes do modelo, na ordem (tipicamente [0, 1]).

                         if predicted_value in model_classes_list: # O valor previsto está diretamente nas classes do modelo?
                             try:
                                  # Encontra o índice do valor previsto na lista de classes do modelo.
                                  pred_index_in_model = model_classes_list.index(predicted_value)
                                  # Usa este índice para aceder à label correspondente em `CLASS_NAMES`.
                                  if 0 <= pred_index_in_model < len(CLASS_NAMES):
                                       predicted_class_label = CLASS_NAMES[pred_index_in_model] # CORRECÇÃO: Usa o índice do valor previsto.
                                  else: st.warning(f"Índice de previsão ({pred_index_in_model}) fora dos limites de {CLASS_NAMES}.") # Warn, use fallback below.
                             except: pass # Keep fallback.

                         # Fallback Adicional: Se `predicted_value` não está diretamente nas classes do modelo, mas é 0 ou 1 (e o modelo é binário).
                         if predicted_class_label == "Valor Previsto Desconhecido" and predicted_value in [0, 1]:
                             if 0 <= int(predicted_value) < len(CLASS_NAMES): # Valid 0 or 1 index.
                                  predicted_class_label = CLASS_NAMES[int(predicted_value)] # Mapeia 0->CLASS_NAMES[0], 1->CLASS_NAMES[1].
                             else: st.warning(f"Valor previsto é 0 ou 1 mas fora dos limites esperados: {predicted_value}") # Warn.

                         if predicted_class_label == "Valor Previsto Desconhecido":
                             st.warning(f"Não foi possível mapear o valor previsto bruto ({predicted_value}) para as labels amigáveis '{CLASS_NAMES}' usando model.classes_ ({model_classes_list}).")
                             predicted_class_label = f"Valor Previsto Bruto: {predicted_value}" # Default final para o raw.

                    else: # Modelo não tem .classes_ ou não parece binário (compatível com 2 CLASS_NAMES).
                        st.warning(f"Modelo ({type(selected_model_instance).__name__}) não tem `.classes_` ou não é binário (esperado 2 classes). Mapeamento para '{CLASS_NAMES}' pode não ser exacto.")
                        predicted_class_label = f"Valor Previsto Bruto: {predicted_value}" # Default para o raw.

                    # --- Exibição do Resultado ---
                    st.markdown('<h2 class="sub-header">Resultado da Previsão:</h2>', unsafe_allow_html=True)

                    # Mensagens baseadas na label amigável `predicted_class_label`.
                    if predicted_class_label == CLASS_NAMES[1]: # É 'sim'.
                         st.balloons()
                         st.success(f"🎉 Previsão: O aluno **PROVAVELMENTE PASSARÁ** no exame final!")
                    elif predicted_class_label == CLASS_NAMES[0]: # É 'não'.
                         st.info(f"😟 Previsão: O aluno **PROVAVELMENTE NÃO PASSARÁ** no exame final.")
                    else: # Label desconhecida ou raw value.
                         st.warning(f"Previsão com resultado inesperado: **{predicted_class_label}**")


                    st.markdown("---")
                    st.markdown("#### Detalhes da Previsão", unsafe_allow_html=True)
                    st.write(f"- **Modelo Utilizado:** **{selected_display_name}** (`{selected_model_filename}`)")
                    st.write(f"- **Classe Prevista (Label):** **{predicted_class_label}**")

                    # --- Exibir Probabilidades (se disponíveis e para problema binário com classes mapeáveis) ---
                    if y_proba_input is not None and y_proba_input.shape[1] == len(CLASS_NAMES) and hasattr(selected_model_instance, 'classes_') : # Shape binário e classes acessíveis.
                         model_classes_list = list(selected_model_instance.classes_) # Classes do modelo na ordem.
                         if 0 in model_classes_list and 1 in model_classes_list: # Modelo prevê 0s/1s numericamente? (Formato mais comum)
                             # Encontra o índice onde 0 e 1 estão na lista model_classes. y_proba tem colunas NESTA ordem.
                             index_of_0_proba = model_classes_list.index(0)
                             index_of_1_proba = model_classes_list.index(1)
                             st.write(f"- **Probabilidade de Não Passar ('{CLASS_NAMES[0]}'):** **{y_proba_input[0][index_of_0_proba]:.2f}**") # CLASS_NAMES[0] <=> 0.
                             st.write(f"- **Probabilidade de Passar ('{CLASS_NAMES[1]}'):** **{y_proba_input[0][index_of_1_proba]:.2f}**") # CLASS_NAMES[1] <=> 1.

                         elif CLASS_NAMES[0] in model_classes_list and CLASS_NAMES[1] in model_classes_list: # Modelo prevê strings ('não'/'sim')?
                             # Encontra o índice onde as strings 'não'/'sim' estão na lista model_classes. y_proba tem colunas NESTA ordem.
                             index_of_nao_proba = model_classes_list.index(CLASS_NAMES[0])
                             index_of_sim_proba = model_classes_list.index(CLASS_NAMES[1])
                             st.write(f"- **Probabilidade de Não Passar ('{CLASS_NAMES[0]}'):** **{y_proba_input[0][index_of_nao_proba]:.2f}**")
                             st.write(f"- **Probabilidade de Passar ('{CLASS_NAMES[1]}'):** **{y_proba_input[0][index_of_sim_proba]:.2f}**")
                         else:
                            st.warning(f"Não foi possível mapear as classes do modelo ({model_classes_list}) para '{CLASS_NAMES[0]}' / '{CLASS_NAMES[1]}' para exibir probabilidades detalhadas.") # Mensagem PT-PT.
                            if y_proba_input.shape[1] == 2: # fallback to raw indices if bin
                                 st.write(f"- Probabilidade (Índice Coluna 0): {y_proba_input[0][0]:.2f}")
                                 st.write(f"- Probabilidade (Índice Coluna 1): {y_proba_input[0][1]:.2f}")

                    else: # Não há probs, ou não é binário 2 colunas, ou sem .classes_.
                         st.info("Probabilidades de previsão não disponíveis ou não aplicáveis para este modelo ou previsão.")


                    st.info("Nota: Esta é uma previsão baseada **unicamente no modelo selecionado e nos dados que introduziu**. Representa a estimativa do modelo, não uma garantia do resultado real do aluno.")


                except Exception as e: # Captura erros gerais durante a predição pipeline (prep, predict, post-predict).
                     st.error(f"❌ Ocorreu um erro durante a execução da previsão: {e}")
                     st.info("Verifique os dados introduzidos e a compatibilidade entre os artefactos carregados (pré-processador, modelo, nomes das características).")
                     st.error(f"Detalhe do erro: {e}") # Mostrar detalhe.


        else: # Se selected_model_instance is None OR preprocessor is None OR original_cols is None.
             # Mensagem genérica já foi exibida acima controlada pela flag `can_run_prediction` e load specific model errors.
             st.info("Botão de previsão não ativo. Verifique se um modelo válido foi carregado e se todos os artefactos essenciais estão disponíveis.")


# --- Secção "Análise do Modelo Treinado Principal" ---
# Apresenta as métricas e interpretabilidade para o 'best_model.joblib'
# no conjunto de teste processado.
# Refatorado para PT-PT, comentários, e robustez no manuseamento de labels (0/1 vs 'não'/'sim').
elif menu == "Análise do Modelo Treinado Principal": # Nome do menu atualizado em sidebar e aqui.
    st.markdown('<h1 class="main-header">Análise do Modelo Treinado Principal</h1>', unsafe_allow_html=True) # Título principal PT-PT.
    st.markdown('<p class="info-text">Esta secção permite visualizar as métricas de avaliação e os fatores de interpretabilidade para o **modelo principal guardado** (`best_model.joblib`) no conjunto de dados de teste processado.</p>', unsafe_allow_html=True) # PT-PT introdução.

    st.warning("⚠️ A análise aqui apresentada refere-se exclusivamente ao modelo guardado como 'best_model.joblib' e aos dados de teste processados. Para comparar diferentes **tipos de algoritmos**, use a secção 'Avaliação de Modelos (CM)'.") # Mensagem de distinção.


    # --- Verificações de Artefactos e Dados Processados Essenciais ---
    # Verifica se os artefactos críticos (`preprocessor`, `model`, `processed_cols`)
    # e os dados de teste processados (`test_df_processed_global`) foram carregados.
    # O modelo (`model`) aqui refere-se explicitamente ao best_model.joblib.
    is_eval_main_model_possible = True # Flag de controlo.

    # Verifica se os artefactos globais foram carregados e não são None.
    if not success_artefacts or preprocessor is None or model is None or processed_cols is None:
         is_eval_main_model_possible = False
         # As mensagens de erro para artefactos em falta já foram exibidas pela `load_pipeline_artefacts_safe`.
    elif test_df_processed_global is None: # Verifica o dataset de teste processado.
         st.warning("O conjunto de teste processado (`test_processed.csv`) não foi carregado. Não é possível avaliar o modelo principal.")
         is_eval_main_model_possible = False
    elif test_df_processed_global.empty: # Verifica se o dataset de teste processado não está vazio.
         st.warning("O conjunto de teste processado está vazio. Não há dados para avaliar o modelo.")
         is_eval_main_model_possible = False
    elif TARGET_PROCESSED_NAME not in test_df_processed_global.columns: # Verifica se a target está no teste processado.
         st.error(f"A coluna alvo processada '{TARGET_PROCESSED_NAME}' não foi encontrada no conjunto de teste processado.")
         is_eval_main_model_possible = False


    # --- Preparar Dados de Teste (X_test, y_test_numeric 0/1) para Scikit-learn ---
    X_test_processed = None # Features do teste processado.
    y_test_processed_numeric = None # Variável alvo do teste em formato numérico (0/1).
    y_test_processed_labels = None # Variável alvo do teste com labels ('não'/'sim').
    is_binary_classification = False # Flag se é um problema binário esperado.


    if is_eval_main_model_possible: # Só tenta preparar dados se as verificações iniciais passaram.
        try:
            X_test_processed = test_df_processed_global.drop(columns=[TARGET_PROCESSED_NAME]) # Características do teste.
            y_test_original_format = test_df_processed_global[TARGET_PROCESSED_NAME] # A coluna alvo tal como está no CSV processado.

            # Validação da compatibilidade das colunas de teste X com `processed_cols`.
            if list(X_test_processed.columns) != list(processed_cols):
                 st.error("❌ Erro de compatibilidade: As colunas de características no conjunto de teste processado não correspondem à lista de nomes de características processadas carregadas ('processed_feature_names.joblib'). A avaliação do modelo não pode ser garantida.")
                 is_eval_main_model_possible = False

            # --- Conversão/Validação da Variável Alvo para Formato Numérico (0/1) e Labels para Métricas ---
            # Necessário que y_true e y_pred sejam 0/1 para métricas como accuracy, classification_report, AUC.
            # Necessário ter as labels correctas ('não', 'sim') para plotar Matriz de Confusão e interpretar FP/FN.
            unique_y_test_vals = y_test_original_format.dropna().unique() # Valores únicos não nulos na target do teste.

            if len(unique_y_test_vals) == 2: # Confirma que é um problema binário.
                 is_binary_classification = True
                 # Tenta mapear os valores únicos para 0 e 1 numéricos se ainda não o forem, e também para as labels 'não'/'sim'.
                 try:
                     # Preferimos 0/1 numérico para a maioria das métricas sklearn.
                     if set(unique_y_test_vals).issubset({0, 1}) and pd.api.types.is_numeric_dtype(y_test_original_format):
                          y_test_processed_numeric = y_test_original_format.astype(int) # Assegura tipo int para consistência cm matrix indexing.
                          # A label para 0 será CLASS_NAMES[0], para 1 será CLASS_NAMES[1] - mapeamento padrão.
                          y_test_processed_labels = y_test_processed_numeric.map({0: CLASS_NAMES[0], 1: CLASS_NAMES[1]}).fillna(y_test_original_format) # Cria uma série com labels PT-PT.

                     # Se os valores são as strings 'não'/'sim'. Converte para 0/1 numérico e mantém as labels.
                     elif set(unique_y_test_vals).issubset(set(CLASS_NAMES)) and (pd.api.types.is_object_dtype(y_test_original_format) or pd.api.types.is_categorical_dtype(y_test_original_format)):
                         st.info(f"Coluna alvo de teste processada é texto binário ('{CLASS_NAMES[0]}', '{CLASS_NAMES[1]}'). Convertendo para numérico 0/1 ('{CLASS_NAMES[0]}' → 0, '{CLASS_NAMES[1]}' → 1) para o cálculo de métricas.")
                         # Converte usando o mapeamento ordenado de CLASS_NAMES para garantir 'não' é 0 e 'sim' é 1.
                         ordered_cat_type = pd.api.types.CategoricalDtype(categories=CLASS_NAMES, ordered=True)
                         y_test_processed_numeric = y_test_original_format.astype(ordered_cat_type).cat.codes.astype(int)
                         y_test_processed_labels = y_test_original_format # As labels já são as strings PT-PT.

                     else: # Binary, but values are not {0,1} or {'não', 'sim'} in a standard format.
                         st.error(f"A coluna alvo de teste processada ('{TARGET_PROCESSED_NAME}') é binária ({list(unique_y_test_vals)}), mas tem um formato ou valores não suportados para conversão automática para numérico 0/1 ou mapeamento de labels PT-PT.")
                         is_eval_main_model_possible = False # Não é seguro prosseguir com métricas binárias.


                 except Exception as e: # Erro na conversão/mapeamento das labels.
                      st.error(f"Falha ao processar a coluna alvo de teste para formato de métricas/plotagem: {e}")
                      is_eval_main_model_possible = False


            elif len(unique_y_test_vals) > 2: # Problema multi-classe ou com >2 classes.
                 st.warning(f"A coluna alvo de teste processada tem mais de 2 valores únicos ({len(unique_y_test_vals)}). As métricas e a matriz de confusão serão calculadas/apresentadas para multi-classe, mas as secções TP/TN/FP/FN e AUC ROC (binários) não serão aplicáveis/exibidas.")
                 y_test_processed_numeric = y_test_original_format # Usa os dados como estão para report/accuracy (skleran lida com multi-class labels).
                 y_test_processed_labels = y_test_original_format.astype(str) # Usa labels como string para plot (ou valores num originais).
                 is_binary_classification = False # Desativa análises binárias específicas.
                 # Continua com `is_eval_main_model_possible` True, mas apenas para as partes não binárias.
            else: # < 2 valores únicos (vazio ou 1 classe).
                 st.warning(f"A coluna alvo de teste processada tem menos de 2 valores únicos ({len(y_test_original_format.dropna())} não-nulos, {list(unique_y_test_vals)} únicos). Não é possível calcular métricas de classificação válidas.")
                 is_eval_main_model_possible = False # Não é possível avaliar.


        except Exception as e: # Captura erros genéricos na preparação dos dados de teste.
            st.error(f"❌ Ocorreu um erro inesperado ao preparar os dados de teste processados para avaliação: {e}")
            is_eval_main_model_possible = False


    # --- Botão para Iniciar a Avaliação do Modelo Principal ---
    # Aparece apenas se for possível executar a avaliação.
    if is_eval_main_model_possible:
        # Note: Usamos `y_test_processed_numeric` para as funções do scikit-learn que esperam 0/1.
        # Usamos `y_test_processed_labels` e `CLASS_NAMES` para a apresentação visual (CM, relatórios).

        if st.button("Avaliar o Modelo Treinado Principal no Conjunto de Teste", key="evaluate_main_model_button"):

            # Inicia animação. Usa o nome do tipo de modelo principal (`model`).
            loading_animation(f"Avaliando o modelo principal ({type(model).__name__})...")

            try: # Bloco try para a execução da avaliação.

                # --- Previsão e Probabilidades no Conjunto de Teste ---
                y_pred_main_model = model.predict(X_test_processed) # Previsões com o modelo principal.

                y_proba_main_model = None # Probabilidades.
                if hasattr(model, 'predict_proba'): # Verifica se o modelo principal suporta `predict_proba`.
                    try: y_proba_main_model = model.predict_proba(X_test_processed)
                    except Exception as proba_e: st.info(f"Não foi possível obter probabilidades para o modelo principal ({type(model).__name__}). Detalhe: {proba_e}")

                # --- Adaptação da Previsão para Formato 0/1 se Original for Label String ---
                # classification_report e outras métricas esperam y_true e y_pred no MESMO formato (preferencialmente 0/1 numérico).
                # y_test_processed_numeric JÁ É numérico 0/1 (se binary).
                # y_pred_main_model VEM do .predict(), que será numérico 0/1 SE o modelo foi treinado com target numérico 0/1.
                # Se o modelo foi treinado com target string 'não'/'sim', .predict() pode retornar strings.
                # Garante que `y_pred_for_metrics` é no mesmo formato 0/1 que `y_test_processed_numeric`.

                y_pred_for_metrics = y_pred_main_model # Começa com a previsão raw.

                if is_binary_classification and not pd.api.types.is_numeric_dtype(y_pred_main_model) and np.all(np.isin(y_pred_main_model, CLASS_NAMES)): # Se é binário e a previsão são as strings PT-PT.
                     st.info("As previsões do modelo principal são strings. Convertendo para numérico 0/1 ('não'→0, 'sim'→1) para o cálculo de métricas.")
                     try:
                         # Converte previsões string 'não'/'sim' para 0/1, usando o mesmo mapeamento que na target original (0=não, 1=sim).
                         mapping_string_to_int = {CLASS_NAMES[0]: 0, CLASS_NAMES[1]: 1}
                         y_pred_for_metrics = np.array([mapping_string_to_int.get(pred, -1) for pred in y_pred_main_model]) # Map strings to 0/1, -1 for unknown. Check for -1 later?
                         if -1 in y_pred_for_metrics: st.warning("Valores inesperados nas previsões ao converter strings para 0/1.") # Warn about unknown.

                     except Exception as e:
                          st.error(f"Falha ao converter previsões string para numérico para métricas: {e}")
                          # Pode deixar `y_pred_for_metrics` no formato string, sklearn classification_report pode lidar,
                          # mas metrics like accuracy, roc_auc_score NEED matching dtypes. Safer to cancel binary metrics if conversion fails.
                          if is_binary_classification: st.warning("Métricas de classificação binária (Accuracy, Report, etc) podem não ser calculadas devido a incompatibilidade no formato dos resultados (previsões string vs target 0/1).")
                          # For classification_report, passing strings y_true and y_pred IS SUPPORTED, as long as labels/target_names match.
                          # So, if target_original_format was strings, we can use that here directly IF y_pred is also strings.

                          # Reavaliar a necessidade de conversão rigorosa para 0/1 numérico: classification_report aceita list/array of labels.
                          # Vamos tentar usar y_test_original_format e y_pred_main_model directamente no classification_report se eles já tiverem o mesmo dtype ou tipos compatíveis.

                # Decide quais dados usar para o `classification_report` e `accuracy_score`.
                # Se é classificação binária, usaremos o formato NUMÉRICO (0/1).
                # Se é multi-classe ou formato não binário seguro, usamos o formato original.
                y_test_for_report = y_test_processed_numeric if is_binary_classification else y_test_original_format
                y_pred_for_report = y_pred_for_metrics if is_binary_classification else y_pred_main_model


                # --- Calcula e Exibe Métricas de Avaliação ---
                st.markdown('<h2 class="sub-header">Métricas de Avaliação no Conjunto de Teste</h2>', unsafe_allow_html=True)

                # Calcula Acurácia. Use os dados formatados para report.
                try:
                    accuracy = accuracy_score(y_test_for_report, y_pred_for_report)
                    st.metric("Acurácia", f"{accuracy:.2f}")
                except Exception as e: st.warning(f"Não foi possível calcular Acurácia: {e}")

                # Calcula Relatório de Classificação.
                st.markdown("#### Relatório de Classificação", unsafe_allow_html=True)
                try:
                    # Use os dados formatados para report. Se multi-classe, target_names pode ser diferente.
                    # Se binário, usamos CLASS_NAMES=['não','sim']. Se multi-classe, sklearn usa valores únicos nas labels como nomes por defeito.
                    # Usar `target_names=CLASS_NAMES` aqui SÓ faz sentido se tivermos 2 classes E estas corresponderem a CLASS_NAMES.
                    # Para multi-classe, melhor omitir target_names ou obter labels únicos de y_true/y_pred.
                    if is_binary_classification: # Use CLASS_NAMES se é binário.
                        report_dict = classification_report(y_test_for_report, y_pred_for_report,
                                                            target_names=CLASS_NAMES, output_dict=True, zero_division=0)
                    else: # Multi-classe, ou não binário detectado como tal. Let sklearn decide target_names.
                        st.info("Calculando relatório de classificação para múltiplas classes.")
                        report_dict = classification_report(y_test_for_report, y_pred_for_report, output_dict=True, zero_division=0)

                    report_df = pd.DataFrame(report_dict).transpose()
                    st.dataframe(report_df.round(2), use_container_width=True)
                except Exception as e: st.error(f"❌ Não foi possível calcular Relatório de Classificação: {e}")

                # Exibir métricas de resumo (se Report foi calculado com weighted avg).
                if 'report_df' in locals() and 'weighted avg' in report_df.index:
                     col_met1, col_met2, col_met3 = st.columns(3)
                     with col_met1: st.metric("Precisão (Méd. Ponderada)", f"{report_df.loc['weighted avg', 'precision']:.2f}")
                     with col_met2: st.metric("Recall (Méd. Ponderada)", f"{report_df.loc['weighted avg', 'recall']:.2f}")
                     with col_met3: st.metric("F1-Score (Méd. Ponderada)", f"{report_df.loc['weighted avg', 'f1-score']:.2f}")
                else:
                     st.info("Médias ponderadas (Precisão, Recall, F1-Score) do relatório de classificação não disponíveis.")


                # Calcula AUC ROC (SÓ PARA CLASSIFICAÇÃO BINÁRIA).
                roc_auc = None
                if is_binary_classification and y_proba_main_model is not None and y_proba_main_model.shape[1] == 2: # Se binário, probs disponíveis, 2 colunas.
                     try:
                          # Precisamos da probabilidade da CLASSE POSITIVA ('sim', que corresponde a 1 numérico)
                          # e os valores REAIS 0/1 numéricos (`y_test_processed_numeric`).
                          # model.classes_ dá a ordem das colunas em y_proba.
                          if hasattr(model, 'classes_') and 1 in model.classes_: # Verifica se 1 (numérico) está nas classes do modelo.
                              index_of_positive_class_proba = list(model.classes_).index(1) # Pega o índice onde 1 está na lista de classes.
                              roc_auc = roc_auc_score(y_test_processed_numeric, y_proba_main_model[:, index_of_positive_class_proba]) # Calcula AUC usando probs da coluna correta e y_test numérico.
                              st.metric("AUC ROC", f"{roc_auc:.2f}")
                              if abs(roc_auc - 0.5) < 0.05: st.warning("AUC ROC perto de 0.5.")
                          else: st.warning("Não foi possível determinar o índice da classe positiva (1) em model.classes_ para calcular AUC ROC.")
                     except ValueError as auc_ve: st.warning(f"Não foi possível calcular AUC ROC (ValueError: {auc_ve}).") # Covers case like only 1 class present in y_true or y_score after subset.
                     except Exception as auc_e: st.warning(f"Erro inesperado ao calcular AUC ROC: {auc_e}")
                elif is_binary_classification: # É binário, mas não tem probs ou shape errado.
                     st.info("AUC ROC: N/A (Probabilidades não disponíveis ou formato incorreto para 2 classes).")
                # Não exibe nada para AUC ROC se não for classificação binária detectada.

                # --- Exibe Matriz de Confusão ---
                st.markdown('<h2 class="sub-header">Matriz de Confusão</h2>', unsafe_allow_html=True)
                # Passamos os *labels* de y_test (strings 'não'/'sim') e as previsões do modelo (que esperançosamente estão no formato 0/1 numérico, compatível com o cálculo de CM padrão), e os nomes das classes em PT-PT.
                # plot_confusion_matrix_interactive lida com o cálculo da CM a partir de y_true/y_pred e a plotagem com os nomes dados.
                # É ESSENCIAL que o `y_test_original_format` e `y_pred_main_model` sejam compatíveis para `confusion_matrix` (e.g., ambos strings 'não'/'sim' OU ambos 0/1 numérico).
                # A lógica anterior na preparação dos dados de teste assegura que `y_test_processed_numeric` (int 0/1) e `y_test_processed_labels` (str 'não'/'sim') estão disponíveis se for binário.
                # Precisamos garantir que a previsão `y_pred_main_model` também está num formato correspondente ou que a confusion_matrix é chamada com os tipos de dados corretos.

                # Opção Segura: Se é binário, usa y_test_processed_numeric para CM cálculo (CM output será 0,1). E usa CLASS_NAMES para o plot rótulos.
                # Se for multi-classe, usa o formato original (assumindo que sklear lida).
                if is_binary_classification:
                    # Para cálculo da CM, usamos os dados 0/1. O array cm será 2x2 com índices 0,1.
                    # Passamos as labels 'não'/'sim' para o PLOT para rótulo dos eixos 0/1.
                    fig_cm, cm_matrix = plot_confusion_matrix_interactive(y_true=y_test_processed_numeric, y_pred=y_pred_for_metrics, class_names=CLASS_NAMES) # CM calculada com 0/1, plot rotulado 'não'/'sim'.
                    st.plotly_chart(fig_cm, use_container_width=True)

                else: # Não é binário detectado (multi-classe ou outro problema com 2+ classes)
                     # Passa os dados como estão para o cálculo da CM, e usa labels inferidas para o plot.
                     st.info("Exibindo Matriz de Confusão para classificação não binária.")
                     fig_cm, cm_matrix = plot_confusion_matrix_interactive(y_true=y_test_original_format, y_pred=y_pred_main_model, class_names=None) # Permite cm inferir labels.
                     st.plotly_chart(fig_cm, use_container_width=True)


                # --- Interpretação da Matriz de Confusão (SÓ PARA CLASSIFICAÇÃO BINÁRIA) ---
                # Só mostra a análise detalhada de TP/TN/FP/FN se o problema for binário e a CM é 2x2.
                if is_binary_classification and cm_matrix.shape == (2, 2):
                    st.markdown("---")
                    st.markdown('<h3 class="sub-header">Interpretação da Matriz de Confusão (Classificação Binária)</h3>', unsafe_allow_html=True)

                    # Precisamos mapear os índices 0/1 da matriz (que são calculados pela confusion_matrix de sklearn, geralmente 0 e 1)
                    # para as nossas labels 'não' e 'sim'. Assumimos que a ordem 0, 1 na CM corresponde à ordem em CLASS_NAMES ['não', 'sim'].
                    # Ou seja, cm[0,0] = Real 0, Previsto 0; cm[0,1] = Real 0, Previsto 1; cm[1,0] = Real 1, Previsto 0; cm[1,1] = Real 1, Previsto 1.
                    # Como `y_test_processed_numeric` é int 0/1 e `y_pred_for_metrics` é (espero) 0/1, `confusion_matrix` resultará num array onde linha/coluna 0 é 'não', linha/coluna 1 é 'sim'.

                    # Assegura que os índices da CM correspondem a 0/1 que mapeiam para 'não'/'sim'.
                    # sklearn confusion_matrix para input 0/1 retorna ordem [0,1]. CLASS_NAMES ['não','sim']. Index 0 em CM é 'não'. Index 1 é 'sim'.
                    try:
                        # TP: Real 'sim' (1), Previsto 'sim' (1) -> cm[1,1]
                        # TN: Real 'não' (0), Previsto 'não' (0) -> cm[0,0]
                        # FP: Real 'não' (0), Previsto 'sim' (1) -> cm[0,1]
                        # FN: Real 'sim' (1), Previsto 'não' (0) -> cm[1,0]

                        # Acessar a matriz CM que foi calculada com input 0/1.
                        tn, fp, fn, tp = cm_matrix[0,0], cm_matrix[0,1], cm_matrix[1,0], cm_matrix[1,1] # Access indices assuming 0='não', 1='sim'.

                        st.write(f"**Verdadeiros Positivos (TP):** {tp}")
                        st.write(f"**Verdadeiros Negativos (TN):** {tn}")
                        st.write(f"**Falsos Positivos (FP):** {fp}")
                        st.write(f"**Falsos Negativos (FN):** {fn}")

                        # Explica o significado no contexto, usando labels PT-PT.
                        st.info(f"""
                        *   **Verdadeiros Positivos (TP):** O modelo previu **Passará** ('{CLASS_NAMES[1]}'), e o aluno **realmente Passou**. Contagens na célula (Real: '{CLASS_NAMES[1]}', Previsto: '{CLASS_NAMES[1]}').
                        *   **Verdadeiros Negativos (TN):** O modelo previu **Não Passará** ('{CLASS_NAMES[0]}'), e o aluno **realmente Não Passou**. Contagens na célula (Real: '{CLASS_NAMES[0]}', Previsto: '{CLASS_NAMES[0]}').
                        *   **Falsos Positivos (FP):** O modelo previu **Passará** ('{CLASS_NAMES[1]}'), mas o aluno **realmente Não Passou**. Representa uma **oportunidade de intervenção perdida** (aluno precisava de ajuda, mas não foi identificado como risco). Contagens na célula (Real: '{CLASS_NAMES[0]}', Previsto: '{CLASS_NAMES[1]}').
                        *   **Falsos Negativos (FN):** O modelo previu **Não Passará** ('{CLASS_NAMES[0]}'), mas o aluno **realmente Passou**. Representa uma **intervenção desnecessária** (aluno não precisava de ajuda, foi identificado como risco). Contagens na célula (Real: '{CLASS_NAMES[1]}', Previsto: '{CLASS_NAMES[0]}').
                        """)
                        # Reitera a importância dos Falsos Positivos.
                        st.warning(f"💡 No contexto deste problema de intervenção estudantil, **Falsos Positivos (FP)** podem ser mais críticos que Falsos Negativos (FN) porque representam alunos que de facto precisavam de atenção mas não foram identificados pelo modelo, resultando na falta de intervenção.")

                    except IndexError: st.warning("Erro ao aceder elementos da matriz de confusão (expecting 2x2 indices).")
                    except Exception as cm_extract_e: st.warning(f"Erro inesperado ao tentar extrair TP/TN/FP/FN: {cm_extract_e}")

                elif cm_matrix.shape != (2, 2): # Binary_classification is True, but shape is not 2x2. Something unexpected happened in CM calc.
                     st.warning(f"O problema foi detetado como binário, mas a matriz de confusão não tem dimensão 2x2 ({cm_matrix.shape}). Não é possível extrair TP/TN/FP/FN de forma fiável.")
                # Note: Se is_binary_classification é False, esta secção não é mostrada.

                # --- Interpretabilidade do Modelo Principal ---
                st.markdown('---')
                st.markdown(f'<h3 class="sub-header">Interpretabilidade do Modelo: {type(model).__name__}</h3>', unsafe_allow_html=True) # Includes model type name.
# Substitui esta linha no seu código:
                st.markdown('<p class="info-text">Quais características (APÓS o pré-processamento) foram mais relevantes para as previsões do modelo principal (<code>best_model.joblib</code>).</p>', unsafe_allow_html=True) # PT-PT.

                feature_names_processed_list = processed_cols # Rename for clarity in this section.

                # Casos de Interpretabilidade baseados no tipo do modelo (global `model`).
                # A lógica usa `hasattr` para verificar se o modelo tem `feature_importances_` (árvores/ensembles) ou `coef_` (lineares).

                # Caso: Modelos com `feature_importances_`.
                if hasattr(model, 'feature_importances_'):
                    st.write("#### Importância das Características")
                    # Verifica se o número de importâncias coincide com o número de features processadas.
                    if len(model.feature_importances_) == len(feature_names_processed_list):
                        # Cria DataFrame e ordena.
                        feature_importance_df = pd.DataFrame({
                            'Característica Processada': feature_names_processed_list, # Nomes processados.
                            'Importância': model.feature_importances_
                        }).sort_values('Importância', ascending=False)

                        # Plotagem.
                        fig_importance = px.bar(feature_importance_df.head(min(30, len(feature_importance_df))), x='Importância', y='Característica Processada', orientation='h', color_continuous_scale=px.colors.sequential.Viridis, title=f"Importância das Características (Processadas) para o Modelo {type(model).__name__}")
                        fig_importance.update_layout(yaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(fig_importance, use_container_width=True)
                        st.info("Importância indica contribuição para redução de erro/impureza (modelos baseados em árvores/ensembles). Reflecte impacto APÓS pré-processamento.")

                    else: st.error("❌ O número de importâncias do modelo não corresponde ao número de características processadas.")

                # Caso: Modelos com `coef_`.
                elif hasattr(model, 'coef_'):
                     st.write("#### Coeficientes das Características")
                     # Acede aos coeficientes. Assume forma para problema binário (1D ou 2D com shape[0]=1).
                     # len(model.coef_[0]) se 2D vs len(model.coef_) se 1D.
                     if hasattr(model.coef_, 'ndim') and (model.coef_.ndim == 1 or (model.coef_.ndim == 2 and model.coef_.shape[0] == 1)):
                          coef_values = model.coef_[0] if model.coef_.ndim == 2 else model.coef_
                          # Verifica se o número de coeficientes corresponde às features processadas.
                          if len(coef_values) == len(feature_names_processed_list):
                               try:
                                    feature_coef_df = pd.DataFrame({'Característica Processada': feature_names_processed_list, 'Coeficiente': coef_values}).sort_values('Coeficiente', ascending=False)
                                    # Calcula range de cor simétrico.
                                    coef_min, coef_max = feature_coef_df['Coeficiente'].min(), feature_coef_df['Coeficiente'].max()
                                    abs_max_coef = max(abs(coef_min), abs(coef_max)) if (pd.notna(coef_min) or pd.notna(coef_max)) and (abs(coef_min) > 1e-9 or abs(coef_max) > 1e-9) else 1e-9 # Evita div by zero/small range.
                                    color_range = [-abs_max_coef, abs_max_coef] # Simétrico range for diverging colorscale.
                                    # Plotagem.
                                    fig_coef = px.bar(feature_coef_df.head(min(30, len(feature_coef_df))), x='Coeficiente', y='Característica Processada', orientation='h', color='Coeficiente', color_continuous_scale='RdBu', range_color=color_range, title=f"Coeficientes das Características (Processadas) para o Modelo {type(model).__name__}")
                                    fig_coef.update_layout(yaxis={'categoryorder':'total ascending'})
                                    st.plotly_chart(fig_coef, use_container_width=True)
                                    st.info(f"""Coeficientes indicam influência linear na probabilidade da classe positiva ('{CLASS_NAMES[1]}' / Passará) para modelos lineares ({type(model).__name__}). Magnitude = importância; sinal = direção (+ aumenta prob. Passar, - diminui). """)
                               except Exception as coef_e: st.error(f"❌ Erro ao exibir coeficientes: {coef_e}")
                          else: st.error("❌ O número de coeficientes no modelo não corresponde ao número de características processadas.")

                     else: st.warning("Atributo `.coef_` tem estrutura inesperada para modelo binário compatível com visualização por feature processada.")

                # Caso: Modelo não suporta interpretabilidade padrão aqui.
                else:
                    st.info(f"O modelo treinado principal ({type(model).__name__}) não expõe interpretabilidade padrão (como 'feature_importances_' ou 'coef_') compatível com as visualizações configuradas nesta secção.")


            # --- Captura Exceções Gerais durante a Avaliação do Modelo Principal ---
            except Exception as e:
                 st.error(f"❌ Ocorreu um erro inesperado durante a avaliação do modelo principal: {e}")
                 st.info("Verifique a compatibilidade entre o modelo guardado e os dados de teste processados (especialmente nomes de colunas e tipos).")
                 st.error(f"Detalhe do erro: {e}")

    else: # is_eval_main_model_possible é False.
         # Mensagem genérica se a avaliação não é possível. Mensagens de erro específicas já foram exibidas acima.
         st.warning("A análise do modelo treinado principal não é possível neste momento. Verifique se os artefactos essenciais (pré-processador, modelo principal, nomes de colunas processadas) e o conjunto de teste processado foram carregados com sucesso.")


# --- Secção "Avaliação de Modelos (CM)" ---
# Permite treinar temporariamente e comparar diferentes algoritmos de ML usando a Matriz de Confusão e métricas.
# O código foi refatorado e traduzido, e deve ser funcional agora com base nas correções.
elif menu == "Avaliação de Modelos (CM)": # Nome do menu atualizado.
    st.markdown('<h1 class="main-header">Avaliação e Comparação de Modelos</h1>', unsafe_allow_html=True)
    st.markdown('<p class="info-text">Explore o desempenho de diferentes tipos de algoritmos de Machine Learning para a previsão de desempenho estudantil. Selecione um algoritmo abaixo para o treinar **temporariamente** no seu conjunto de treino processado e ver a sua performance (métricas, Matriz de Confusão) no conjunto de teste processado ou via Cross-Validation.</p>', unsafe_allow_html=True) # PT-PT intro.

    # --- Verificações de Pré-requisitos ---
    # Requer dados de treino E teste processados E nomes de colunas processadas.
    # is_eval_algos_possible will now check for base data needed for *either* method.
    is_base_data_loaded = True # Flag de controlo para dados base.

    # Verifica se os dataframes processados necessários estão carregados e não vazios.
    # BOTH train and test are needed *conceptually* for the app flow, even if CV only uses train.
    # Processed cols are needed for X sets.
    if train_df_processed_global is None or train_df_processed_global.empty or \
       test_df_processed_global is None or test_df_processed_global.empty:
         is_base_data_loaded = False
         st.error("❌ Os conjuntos de dados de treino e/ou teste processados não foram carregados ou estão vazios. Não é possível realizar avaliação.")
         # Mensagens de erro/warning já podem ter sido exibidas pelas funções de carga no topo.

    # Verifica se os nomes das características processadas estão carregados.
    if 'processed_cols' not in locals() or processed_cols is None or not isinstance(processed_cols, (list, pd.Index)) or len(processed_cols) == 0:
        if is_base_data_loaded: # Only display this error if data *did* load, but cols didn't.
             st.error("❌ Os nomes das características processadas ('processed_feature_names.joblib') não foram carregados ou estão vazios. Não é possível preparar os dados de entrada para os modelos.")
        is_base_data_loaded = False


    # --- Preparação dos Dados (X_train, y_train, X_test, y_test - todos processados) ---
    # Se for possível prosseguir, prepara X/y sets, convertendo a target para numérico (0/1) para compatibilidade com scikit-learn.
    X_train_processed_alg = None; y_train_processed_numeric_alg = None; y_train_original_format_alg = None # Vars treino
    X_test_processed_alg = None; y_test_processed_numeric_alg = None; y_test_original_format_alg = None # Vars teste
    is_binary_alg = False # Flag se o problema binário foi detectado corretamente.

    # Variáveis para dados LIMPOS (sem NaNs) e no formato string/numeric final para métricas/report/CM
    # These will be populated *after* prediction/CV based on the selected method.
    # y_true_cleaned_for_metrics = None; y_pred_cleaned_for_metrics = None # Used for test set eval
    # y_true_cleaned_for_report_cm = None; y_pred_cleaned_for_report_cm = None # Used for test set eval

    # New: Variables for CLEANED training data for CV
    X_train_cleaned_for_cv = None
    y_train_cleaned_for_cv = None # Will be numeric 0/1 if binary


    if is_base_data_loaded:
        try:
             # Extrai features (X) e target (y) dos dataframes processados carregados.
             X_train_processed_alg = train_df_processed_global.drop(columns=[TARGET_PROCESSED_NAME])
             y_train_original_format_alg = train_df_processed_global[TARGET_PROCESSED_NAME]

             X_test_processed_alg = test_df_processed_global.drop(columns=[TARGET_PROCESSED_NAME])
             y_test_original_format_alg = test_df_processed_global[TARGET_PROCESSED_NAME]

             # Valida compatibilidade das colunas X: treino vs teste vs processed_cols.
             if list(X_train_processed_alg.columns) != list(X_test_processed_alg.columns) or list(X_train_processed_alg.columns) != list(processed_cols):
                 st.error("❌ Erro de compatibilidade: As colunas de características dos conjuntos processados não são consistentes entre treino, teste e a lista carregada de características processadas ('processed_feature_names.joblib'). Não é seguro prosseguir.")
                 is_base_data_loaded = False # Can't proceed with either method if cols are messed up.


             # --- Conversão/Validação da Variável Alvo (y) para Formato Numérico (0/1) e Determinação de Labels ---
             # Para ambos treino e teste. Necessário 0/1 numérico para métricas e `.fit` na maioria dos algos.
             # Necessário labels para o report e CM.
             if is_base_data_loaded: # Only attempt if column check passed
                 unique_y_train_vals = y_train_original_format_alg.dropna().unique()
                 unique_y_test_vals = y_test_original_format_alg.dropna().unique()

                 # Binary check: exactly 2 unique values in TRAIN (dropna) and TEST (dropna) and test values are a subset of train values.
                 if len(unique_y_train_vals) == 2 and len(unique_y_test_vals) == 2 and set(unique_y_test_vals).issubset(set(unique_y_train_vals)):
                      is_binary_alg = True
                      all_binary_values = sorted(list(unique_y_train_vals) + list(unique_y_test_vals)) # e.g. [0, 1] or ['não', 'sim']

                      # Conversão/atribuição para formato NUMÉRICO (0/1) para treino/métricas: y_train/test_processed_numeric_alg.
                      # Esta variável y_test_processed_numeric_alg é a que tentamos usar para métricas como Acurácia e AUC.
                      try:
                           # Check if already 0/1 numeric
                           if set(all_binary_values).issubset({0, 1}) and pd.api.types.is_numeric_dtype(y_train_original_format_alg):
                                y_train_processed_numeric_alg = y_train_original_format_alg.astype(int)
                                y_test_processed_numeric_alg = y_test_original_format_alg.astype(int) # TESTE target numeric 0/1
                           # Check if string and CLASS_NAMES matches
                           elif 'CLASS_NAMES' in globals() and len(CLASS_NAMES) == 2 and set(all_binary_values).issubset(set(CLASS_NAMES)) and pd.api.types.is_object_dtype(y_train_original_format_alg): # Strings 'não'/'sim' matching CLASS_NAMES
                                # Map to 0/1 using CLASS_NAMES order (CLASS_NAMES[0] -> 0, CLASS_NAMES[1] -> 1).
                                ordered_cat_type = pd.api.types.CategoricalDtype(categories=CLASS_NAMES, ordered=True)
                                y_train_processed_numeric_alg = y_train_original_format_alg.astype(ordered_cat_type).cat.codes.astype(int)
                                y_test_processed_numeric_alg = y_test_original_format_alg.astype(ordered_cat_type).cat.codes.astype(int) # TESTE target numeric 0/1
                           else:
                                st.error(f"Dados alvo binários processados têm um formato ou valores inesperados ({list(all_binary_values)} / {y_train_original_format_alg.dtype}) para conversão para numérico 0/1. A avaliação dos algoritmos não pode ser garantida.")
                                is_binary_alg = False # Cannot guarantee 0/1 numeric for fit/metrics.
                                is_base_data_loaded = False # Critical failure.
                      except Exception as e:
                           st.error(f"Falha na conversão de dados alvo processados para numérico 0/1: {e}")
                           is_binary_alg = False
                           is_base_data_loaded = False # Critical.


                 elif len(unique_y_train_vals) > 2 or len(unique_y_test_vals) > 2: # Multi-classe detectado
                      st.warning(f"Conjuntos processados contêm mais de 2 classes (Treino: {list(unique_y_train_vals)}, Teste: {list(unique_y_test_vals)}). A avaliação procederá para multi-classe, mas secções binárias específicas (TP/TN/FP/FN, AUC ROC) não serão exibidas.")
                      # Usa dados como estão para treino/previsão.
                      y_train_processed_numeric_alg = y_train_original_format_alg # No attempt to force 0..n-1 for training - most sklearn models handle multi-class targets directly.
                      y_test_processed_numeric_alg = y_test_original_format_alg # No attempt to force 0..n-1 for metrics (will use original for cleanup later)
                      is_binary_alg = False # Explicitly False.


                 else: # < 2 valores únicos (vazio, 1 classe) -> covered by initial check or will be caught by cleaning.
                      pass # is_base_data_loaded should be False or cleaning will fail.

        except Exception as e: # Captura erros genéricos na preparação inicial de X/Y sets.
            st.error(f"❌ Ocorreu um erro genérico durante a preparação dos conjuntos de dados processados X/Y: {e}")
            is_base_data_loaded = False

    # --- NEW: Prepare Cleaned Training Data for CV (if needed) ---
    can_do_cv = False # Assume cannot do CV initially
    if is_base_data_loaded: # Only attempt CV data prep if base data loaded successfully
        try:
            # Combine X_train and y_train (using the _numeric_alg version which is 0/1 if binary)
            # Ensure the target is added with a temporary, known name.
            temp_train_cv_df = X_train_processed_alg.copy()
            temp_train_cv_df['_temp_target_col'] = y_train_processed_numeric_alg # Use numeric target for CV data prep

            # Drop rows with *any* NaN (in features or target) for CV
            temp_train_cv_df_cleaned = temp_train_cv_df.dropna()

            X_train_cleaned_for_cv = temp_train_cv_df_cleaned.drop(columns=['_temp_target_col'])
            y_train_cleaned_for_cv = temp_train_cv_df_cleaned['_temp_target_col']

            # Check if enough data remains after cleaning for CV (at least n_splits samples, and ideally > 0 total samples)
            # Check for minimum samples per class if binary for StratifiedKFold
            min_samples_needed_cv = 2 # Basic minimum for splitting
            if is_binary_alg:
                 # StratifiedKFold needs at least 2 samples of *each* class in each fold.
                 # This implies at least k * 2 samples total for binary. Let's just check total is > k.
                 min_samples_needed_cv = 5 # A bit more robust minimum
                 if n_splits := st.session_state.get("cv_n_splits", 5): # Get default or slider value early if possible
                      min_samples_needed_cv = max(min_samples_needed_cv, n_splits) # Ensure at least k samples

            if len(X_train_cleaned_for_cv) < min_samples_needed_cv:
                 st.warning(f"Após remover valores nulos do conjunto de treino, não há amostras suficientes ({len(X_train_cleaned_for_cv)}) para executar Cross-Validation (necessário >={min_samples_needed_cv} samples).")
                 can_do_cv = False
            elif is_binary_alg and len(y_train_cleaned_for_cv.unique()) < 2:
                 st.warning("Após remover valores nulos, o conjunto de treino para CV tem apenas uma classe. Não é possível executar Cross-Validation estratificada para classificação binária.")
                 can_do_cv = False
            else:
                 can_do_cv = True # CV data prepared and seems sufficient.


        except Exception as e:
             st.error(f"❌ Ocorreu um erro durante a limpeza dos dados de treino para Cross-Validation: {e}")
             can_do_cv = False # Prevent CV if cleaning fails.


    # --- Seleção do Método de Avaliação ---
    st.markdown('<h3 class="sub-header">Método de Avaliação</h3>', unsafe_allow_html=True)
    evaluation_method = st.radio(
        "Selecione como avaliar o modelo:",
        ["Avaliação no Conjunto de Teste", "Cross-Validation (StratifiedKFold)"],
        key="evaluation_method_selector"
    )

    # --- Configuração Específica do Método ---
    n_splits = 5 # Default value for CV folds
    if evaluation_method == "Cross-Validation (StratifiedKFold)":
         if not can_do_cv:
             st.error("Não é possível executar Cross-Validation com os dados de treino processados limpos. Verifique as mensagens de erro ou aviso acima.")
             # This prevents the button from appearing below if CV is selected and not possible.
             # We'll handle this by checking the `can_run_evaluation` flag before showing the model selection.
         else:
             st.markdown('#### Configuração de Cross-Validation')
             # Slider for number of folds, ensure min is 2. Max can be size of cleaned data, but 10 is reasonable limit.
             n_splits = st.slider("Número de Folds (k)", 2, min(10, len(X_train_cleaned_for_cv) if can_do_cv else 10), 5, key="cv_n_splits")
             # Re-check min samples needed with the selected n_splits
             min_samples_needed_cv = n_splits
             if is_binary_alg: min_samples_needed_cv = max(min_samples_needed_cv, n_splits) # Ensure at least k if binary
             if len(X_train_cleaned_for_cv) < min_samples_needed_cv:
                 st.warning(f"O número de folds selecionado ({n_splits}) é demasiado alto para o número de amostras de treino limpas disponíveis ({len(X_train_cleaned_for_cv)}). Reduza o número de folds.")
                 can_do_cv = False # Cannot proceed with CV with this setting

             # Add info about StratifiedKFold
             if can_do_cv:
                 if is_binary_alg: st.info(f"Será usada validação cruzada estratificada ({n_splits}-Fold) adequada para dados binários.")
                 else: st.info(f"Será usada validação cruzada estratificada ({n_splits}-Fold) para dados multi-classe (se possível).") # StratifiedKFold works for multi-class too.


    # --- Seleção do Algoritmo e Configuração ---
    # Only show model selection and button if base data is loaded AND the chosen method is possible.
    can_run_evaluation = is_base_data_loaded and (evaluation_method == "Avaliação no Conjunto de Teste" or (evaluation_method == "Cross-Validation (StratifiedKFold)" and can_do_cv))


    if can_run_evaluation:
        st.markdown('<h3 class="sub-header">Seleção e Configuração do Algoritmo</h3>', unsafe_allow_html=True) # Sub-cabeçalho PT-PT.

        # Certifique que AVAILABLE_MODELS_FOR_ANALYSIS está definido
        if 'AVAILABLE_MODELS_FOR_ANALYSIS' not in globals() or not AVAILABLE_MODELS_FOR_ANALYSIS:
             st.error("Erro interno: Lista de modelos disponíveis (AVAILABLE_MODELS_FOR_ANALYSIS) não definida ou vazia.")
             can_run_evaluation = False # Cannot continue.

        if can_run_evaluation: # Double check flag before proceeding
            selected_model_name_alg = st.selectbox(
                "Escolha o tipo de algoritmo para treinar temporariamente:",
                list(AVAILABLE_MODELS_FOR_ANALYSIS.keys()),
                key="evaluation_model_selector_alg" # Chave única.
            )

            # --- Configuração Opcional de Parâmetros ---
            st.markdown('#### Configuração do Algoritmo (Opcional)', unsafe_allow_html=True)
            model_params_alg = {} # Dicionário para parâmetros do modelo selecionado dinamicamente.
            current_model_base_instance = AVAILABLE_MODELS_FOR_ANALYSIS.get(selected_model_name_alg)

            # Data size limits for parameters (use training data size for KNN, min_samples_split)
            train_data_size = len(X_train_processed_alg) if X_train_processed_alg is not None else 0

            if current_model_base_instance is not None:
                 model_class_for_alg = type(current_model_base_instance) # Get class type from base instance.

                 # Display specific parameter sliders based on model name. Uses `getattr` to check default parameter value existence.
                 # (O código de configuração de parâmetros mantém-se o mesmo)
                 if selected_model_name_alg == "KNN":
                     default_n = getattr(current_model_base_instance, 'n_neighbors', 5)
                     # Garante que n_neighbors não excede o número de amostras de treino - 1 (para ter pelo menos 1 vizinho diferente da própria amostra na maioria dos casos)
                     max_n_neighbors = max(1, train_data_size - 1) # Ensure min 1 neighbor if data exists
                     model_params_alg['n_neighbors'] = st.slider(f"**{selected_model_name_alg}**: Número de Vizinhos (`n_neighbors`)", 1, max_n_neighbors, min(int(default_n), max_n_neighbors), key=f"{selected_model_name_alg}_n_neighbors_alg")

                 elif selected_model_name_alg in ["Árvore de Decisão", "Random Forest"]:
                     default_max_depth = getattr(current_model_base_instance, 'max_depth', None)
                     # Default slider value for max_depth, ensuring it's at least 1 if data exists
                     max_possible_depth = 30 # Reasonable upper limit for visualization/computation
                     default_slider_value_depth = int(default_max_depth) if default_max_depth is not None and default_max_depth > 0 else (3 if selected_model_name_alg == "Árvore de Decisão" else 5)
                     default_slider_value_depth = min(default_slider_value_depth, max_possible_depth) # Limit default to max_possible_depth
                     model_params_alg['max_depth'] = st.slider(f"**{selected_model_name_alg}**: Profundidade Máxima (`max_depth`)", 1, max_possible_depth, default_slider_value_depth, key=f"{selected_model_name_alg}_max_depth_alg")

                     default_min_samples_split = getattr(current_model_base_instance, 'min_samples_split', 2)
                     # min_samples_split cannot exceed the number of training samples.
                     max_min_samples_split = max(2, train_data_size) # Ensure min 2
                     model_params_alg['min_samples_split'] = st.slider(f"**{selected_model_name_alg}**: Mínimo de Amostras para Dividir (`min_samples_split`)", 2, min(20, max_min_samples_split), min(int(default_min_samples_split), max_min_samples_split), key=f"{selected_model_name_alg}_min_samples_split_alg")

                     if selected_model_name_alg == "Random Forest":
                         default_n_estimators = getattr(current_model_base_instance, 'n_estimators', 100)
                         model_params_alg['n_estimators'] = st.slider(f"**{selected_model_name_alg}**: Número de Árvores (`n_estimators`)", 50, 500, int(default_n_estimators), key=f"{selected_model_name_alg}_n_estimators_alg")
                 elif selected_model_name_alg in ["Regressão Logística", "SVM (Kernel RBF)", "Gradient Boosting", "AdaBoost"]:
                      st.info("Configuração de parâmetros via interface não disponível para este algoritmo neste momento.")
                 else: # Fallback for other models in the dictionary without explicit param config.
                      st.info(f"Configuração de parâmetros específica não disponível para **{selected_model_name_alg}**.")


            else: # selected_model_name_alg not found in AVAILABLE_MODELS_FOR_ANALYSIS
                 st.error(f"Erro interno: Algoritmo selecionado '{selected_model_name_alg}' não encontrado na lista AVAILABLE_MODELS_FOR_ANALYSIS.")
                 can_run_evaluation = False # Can't proceed if the base model instance isn't found.


        # --- Botão para Treinar e Avaliar o Algoritmo Selecionado ---
        # Button is active only if `can_run_evaluation` is True.
        if st.button(f"🏃‍♂️ Executar Avaliação: {selected_model_name_alg} ({evaluation_method})", key="run_alg_evaluation_button"):

            # Determine loading message based on method
            loading_msg = f"A executar {n_splits}-Fold CV para {selected_model_name_alg}..." if evaluation_method == "Cross-Validation (StratifiedKFold)" else f"A treinar {selected_model_name_alg} e avaliar no teste..."
            loading_animation(loading_msg)

            try: # Try block for the dynamic training and evaluation process.
                # --- Instanciar o Modelo com Parâmetros Selecionados ---
                if 'model_class_for_alg' not in locals(): raise ValueError("Classe do modelo não definida antes da instanciação.")
                # Add random_state for reproducibility if the model class accepts it
                if 'random_state' in model_class_for_alg().get_params().keys():
                    model_instance_to_eval = model_class_for_alg(random_state=42, **model_params_alg)
                else:
                     model_instance_to_eval = model_class_for_alg(**model_params_alg)


                # --- CONDITIONAL EVALUATION LOGIC ---

                if evaluation_method == "Avaliação no Conjunto de Teste":
                    # --- STANDARD TEST SET EVALUATION ---

                    st.write(f"A treinar modelo ({type(model_instance_to_eval).__name__}) no conjunto de treino...")
                    # Use numeric target (0/1) for binary training, original target for multi-class (sklearn handles).
                    y_train_fit = y_train_processed_numeric_alg if is_binary_alg else y_train_original_format_alg

                    # Check if training data is sufficient for fit
                    if len(X_train_processed_alg) == 0 or len(y_train_fit) == 0 or len(X_train_processed_alg) != len(y_train_fit):
                         st.error("Dados de treino processados insuficientes ou inconsistentes para treinar o modelo.")
                         # This is a critical error for this path, but should have been caught by `is_base_data_loaded`.
                         # Re-raise or handle gracefully if needed, but assuming checks above are sufficient.
                         raise ValueError("Training data insufficient/inconsistent.")


                    model_instance_to_eval.fit(X_train_processed_alg, y_train_fit)
                    st.success(f"✅ Modelo '{selected_model_name_alg}' treinado com sucesso no conjunto de treino.")


                    # --- Avaliar o Modelo no Conjunto de Teste ---
                    st.write("A avaliar no conjunto de teste processado...")
                    # Check if test data is sufficient for prediction
                    if len(X_test_processed_alg) == 0 or len(y_test_original_format_alg) == 0 or len(X_test_processed_alg) != len(y_test_original_format_alg):
                         st.error("Dados de teste processados insuficientes ou inconsistentes para avaliar o modelo.")
                         raise ValueError("Test data insufficient/inconsistent.")


                    y_pred_alg = model_instance_to_eval.predict(X_test_processed_alg) # Predict.

                    y_proba_alg = None # Get probabilities if model supports it (for AUC).
                    if hasattr(model_instance_to_eval, 'predict_proba'):
                         try: y_proba_alg = model_instance_to_eval.predict_proba(X_test_processed_alg)
                         except Exception as proba_e: st.info(f"Não foi possível obter probabilidades ({proba_e}).")

                    # --- Preparação dos Dados Limpos para Métricas/Report/CM (TESTE) ---
                    # Combine y_true (original test target) and y_pred (raw prediction) to drop NaNs consistently.
                    # Use the original format for target, as we will map it to strings for report/CM later if binary.
                    # Ensure y_pred is in a pandas Series to use .dropna() and .astype(str).
                    y_pred_series = pd.Series(y_pred_alg, index=y_test_original_format_alg.index) # Convert predictions to Series with matching index

                    # Create a temporary DataFrame to drop NaNs where *either* true or predicted is NaN.
                    temp_eval_df = pd.DataFrame({'y_true': y_test_original_format_alg, 'y_pred': y_pred_series})
                    temp_eval_df_cleaned = temp_eval_df.dropna()

                    # Extract cleaned true and predicted values
                    y_true_cleaned = temp_eval_df_cleaned['y_true']
                    y_pred_cleaned = temp_eval_df_cleaned['y_pred']

                    # Convert cleaned true and predicted values to the final format needed for classification_report and CM plotting.
                    # For binary, we want strings ('Não', 'Sim') if possible. For multi-class, original values as strings.
                    y_true_cleaned_for_report_cm = y_true_cleaned.astype(str) # Default to string
                    y_pred_cleaned_for_report_cm = y_pred_cleaned.astype(str) # Default to string

                    # If binary and the cleaned values are 0/1 numeric, map them to CLASS_NAMES strings for consistency in report/CM.
                    if is_binary_alg and 'CLASS_NAMES' in globals() and len(CLASS_NAMES) >= 2:
                         # Mapping based on the expected order [0->CLASS_NAMES[0], 1->CLASS_NAMES[1]]
                         mapping_numeric_to_string = {0: CLASS_NAMES[0], 1: CLASS_NAMES[1]}
                         try:
                             # Apply mapping only if the series is numeric (or contains 0/1 as objects/strings that can be converted)
                             if pd.api.types.is_numeric_dtype(y_true_cleaned): # If already numeric 0/1
                                  y_true_cleaned_for_report_cm = y_true_cleaned.map(mapping_numeric_to_string).fillna(y_true_cleaned.astype(str))
                             elif all(v in ['0','1'] for v in y_true_cleaned.astype(str).unique()): # If strings '0'/'1'
                                  y_true_cleaned_for_report_cm = y_true_cleaned.astype(int).map(mapping_numeric_to_string).fillna(y_true_cleaned.astype(str))
                             # Otherwise, it's already something else (strings like 'Não'/'Sim' or other unexpected), leave as is (will be .astype(str) from above)
                         except Exception as map_e: st.warning(f"Falha ao mapear y_true (0/1 ou string 0/1) para strings para CM/Relatório: {map_e}")

                         try:
                             if pd.api.types.is_numeric_dtype(y_pred_cleaned):
                                  y_pred_cleaned_for_report_cm = y_pred_cleaned.map(mapping_numeric_to_string).fillna(y_pred_cleaned.astype(str))
                             elif all(v in ['0','1'] for v in y_pred_cleaned.astype(str).unique()):
                                  y_pred_cleaned_for_report_cm = y_pred_cleaned.astype(int).map(mapping_numeric_to_string).fillna(y_pred_cleaned.astype(str))
                         except Exception as map_e: st.warning(f"Falha ao mapear y_pred (0/1 ou string 0/1) para strings para CM/Relatório: {map_e}")
                    # Else: If not binary or CLASS_NAMES not defined/incomplete, y_true/pred cleaned are just converted to strings by default above.


                    # Ensure cleaned data has enough samples for metrics
                    if len(y_true_cleaned) == 0 or len(y_true_cleaned) != len(y_pred_cleaned):
                         st.warning("Após remover valores nulos do conjunto de teste, não há amostras suficientes ou consistentes para calcular métricas de avaliação. Não é possível gerar o relatório.")
                         # Skip metric/report/CM calculations.
                         # Jump to interpretability section if applicable.
                         metrics_calculated = False
                    else:
                         metrics_calculated = True
                         # --- Calcula e Exibe Métricas de Avaliação (TESTE) ---
                         st.markdown('<h3 class="sub-header">Resultados da Avaliação no Conjunto de Teste</h3>', unsafe_allow_html=True)

                         # Acurácia: Needs y_true and y_pred in compatible formats.
                         # Use the cleaned data. If binary, try to convert to 0/1 numeric for accuracy_score if needed.
                         y_true_cleaned_for_metrics = y_true_cleaned # Start with cleaned data
                         y_pred_cleaned_for_metrics = y_pred_cleaned # Start with cleaned data

                         # If binary, ensure both are numeric 0/1 for standard metrics like accuracy_score
                         if is_binary_alg and 'CLASS_NAMES' in globals() and len(CLASS_NAMES) >= 2:
                             try:
                                  # Convert true targets to 0/1 numeric if they aren't already
                                  if not pd.api.types.is_numeric_dtype(y_true_cleaned_for_metrics) or not set(y_true_cleaned_for_metrics.dropna().unique()).issubset({0, 1}):
                                       ordered_cat_type = pd.api.types.CategoricalDtype(categories=CLASS_NAMES, ordered=True) # Map CLASS_NAMES[0]->0, CLASS_NAMES[1]->1
                                       y_true_cleaned_for_metrics = y_true_cleaned_for_report_cm.astype(ordered_cat_type).cat.codes # Use report_cm strings which are mapped

                                  # Convert predicted targets to 0/1 numeric if they aren't already
                                  if not pd.api.types.is_numeric_dtype(y_pred_cleaned_for_metrics) or not set(y_pred_cleaned_for_metrics.dropna().unique()).issubset({0, 1}):
                                       ordered_cat_type = pd.api.types.CategoricalDtype(categories=CLASS_NAMES, ordered=True) # Map CLASS_NAMES[0]->0, CLASS_NAMES[1]->1
                                       y_pred_cleaned_for_metrics = y_pred_cleaned_for_report_cm.astype(ordered_cat_type).cat.codes # Use report_cm strings which are mapped

                             except Exception as e: st.warning(f"Falha ao converter y_true/y_pred limpos para 0/1 para métricas (continuando com o formato atual): {e}")
                             # Note: If conversion fails, accuracy might be calculated on strings/other numerics if sklearn supports.

                         # Acurácia
                         try:
                              accuracy = accuracy_score(y_true_cleaned_for_metrics, y_pred_cleaned_for_metrics) # Use the cleaned, potentially converted data for accuracy.
                              st.metric("Acurácia", f"{accuracy:.2f}")
                         except Exception as e: st.warning(f"Não foi possível calcular Acurácia: {e}")


                         # Relatório de Classificação. Usa as labels string limpas.
                         st.markdown(f"#### Relatório de Classificação ({selected_model_name_alg})", unsafe_allow_html=True)
                         try:
                             # Pass the cleaned STRING labels.
                             # Explicitly specify the labels parameter using the expected CLASS_NAMES strings if binary.
                             # This forces the report to only consider these two labels, ignoring others (like 'nan').
                             report_labels_param = CLASS_NAMES if is_binary_alg and 'CLASS_NAMES' in globals() and len(CLASS_NAMES) >= 2 else None
                             # If not binary, let classification_report infer labels from the cleaned data.
                             # If binary but CLASS_NAMES is missing, it will also infer (and might still fail if 'nan' exists).
                             # Check if report_labels_param is None and data contains 'nan' strings.
                             if report_labels_param is None and ('nan' in y_true_cleaned_for_report_cm.unique() or 'nan' in y_pred_cleaned_for_report_cm.unique()):
                                  st.warning("Dados limpos contêm 'nan' como label. O relatório pode incluir esta classe inesperada.")

                             report_dict = classification_report(y_true=y_true_cleaned_for_report_cm, y_pred=y_pred_cleaned_for_report_cm,
                                                                 target_names=CLASS_NAMES if is_binary_alg and 'CLASS_NAMES' in globals() and len(CLASS_NAMES) >= 2 else None, # Use CLASS_NAMES for display names if binary
                                                                 labels=report_labels_param, # Explicitly specify the labels to consider IF binary
                                                                 output_dict=True, zero_division=0)

                             report_df = pd.DataFrame(report_dict).transpose()
                             st.dataframe(report_df.round(2), use_container_width=True)
                             report_calculated = True # Flag indicating report_df exists
                         except Exception as e:
                             st.error(f"❌ Não foi possível calcular Relatório de Classificação: {e}")
                             report_calculated = False


                         # Métricas Resumo (Médias, F1, AUC)
                         if report_calculated and 'weighted avg' in report_df.index:
                              col_met1, col_met2, col_met3 = st.columns(3)
                              with col_met1: st.metric("Precisão (Méd. Ponderada)", f"{report_df.loc['weighted avg', 'precision']:.2f}")
                              with col_met2: st.metric("Recall (Méd. Ponderada)", f"{report_df.loc['weighted avg', 'recall']:.2f}")
                              with col_met3: st.metric("F1-Score (Méd. Ponderada)", f"{report_df.loc['weighted avg', 'f1-score']:.2f}")
                         else:
                              # Check if report_df exists but 'weighted avg' isn't there (can happen if only 1 class remains after cleaning)
                              if report_calculated and not report_df.empty:
                                   st.warning("Médias ponderadas do relatório não disponíveis (dataset de teste pode ter apenas uma classe após limpeza).")
                              elif metrics_calculated: # report_df doesn't exist (error calculation above) or is empty, but metrics attempt was made.
                                   st.info("Médias ponderadas do relatório não disponíveis (não foi possível gerar o relatório de classificação).")


                         # AUC ROC (SÓ PARA CLASSIFICAÇÃO BINÁRIA E SE PROBABILIDADES DISPONÍVEIS/COMPATÍVEIS)
                         # Calculation needs y_true in 0/1 numeric and y_proba_alg[:, positive_class_index].
                         # Use y_true_cleaned_for_metrics which should be 0/1 numeric if is_binary_alg is True.
                         roc_auc = None
                         # Check if problem binary, probs obtained, and proba output shape matches expected classes.
                         if is_binary_alg and 'CLASS_NAMES' in globals() and len(CLASS_NAMES) == 2 and y_proba_alg is not None and y_proba_alg.shape[1] == len(CLASS_NAMES):
                              try:
                                   # Ensure y_true_cleaned_for_metrics is numeric 0/1 as AUC requires this.
                                   # The cleaning/conversion step above should have attempted this. Check dtype again.
                                   if not pd.api.types.is_numeric_dtype(y_true_cleaned_for_metrics) or not set(y_true_cleaned_for_metrics.dropna().unique()).issubset({0,1}):
                                        st.warning("AUC ROC: Target verdadeira não está em formato numérico 0/1 após limpeza. Não é possível calcular AUC.")
                                   # Align y_proba_alg with cleaned y_true (remove rows where y_true was NaN)
                                   # The indices of y_test_original_format_alg map to rows in X_test_processed_alg
                                   # temp_eval_df_cleaned had the index of the cleaned rows.
                                   y_proba_cleaned = y_proba_alg[temp_eval_df_cleaned.index] # Index into the numpy array

                                   if len(y_true_cleaned_for_metrics) != y_proba_cleaned.shape[0]:
                                        st.warning(f"AUC ROC: Número de amostras na target limpa ({len(y_true_cleaned_for_metrics)}) não coincide com probabilidades limpas ({y_proba_cleaned.shape[0]}). Não é possível calcular AUC.")
                                   else:
                                      # Find the index of the numeric label 1 in the model's classes list to get the correct probability column.
                                      # Assume model classes are numeric 0/1 if trained with numeric target (y_train_processed_numeric_alg).
                                      # Check if model has 'classes_' attribute.
                                      if hasattr(model_instance_to_eval, 'classes_'):
                                          # Find the index of the positive class (1).
                                          # model_instance_to_eval.classes_ should be [0, 1] if trained on numeric 0/1.
                                          # model_instance_to_eval.classes_ could be [CLASS_NAMES[0], CLASS_NAMES[1]] if trained on strings (less common, but possible)
                                          # We need the probability column corresponding to the label '1'.
                                          target_label_1_numeric = 1
                                          if target_label_1_numeric in model_instance_to_eval.classes_:
                                               index_of_positive_class = list(model_instance_to_eval.classes_).index(target_label_1_numeric)
                                               # Use y_true_cleaned_for_metrics (numeric 0/1) and the correct proba column.
                                               roc_auc = roc_auc_score(y_true=y_true_cleaned_for_metrics, y_score=y_proba_cleaned[:, index_of_positive_class])
                                               st.metric("AUC ROC", f"{roc_auc:.2f}")
                                               if abs(roc_auc - 0.5) < 0.05: st.warning("AUC ROC perto de 0.5, sugere performance próxima do aleatório.")
                                          else:
                                               st.warning("AUC ROC: Não foi possível determinar a coluna da classe positiva (1) a partir das classes do modelo.") # Cannot find 1 in model.classes_

                                      else: st.warning("AUC ROC: Modelo não expõe 'classes_' para determinar a coluna de probabilidade positiva.") # Model lacks classes_ attr.

                              except ValueError as auc_ve: st.warning(f"Não foi possível calcular AUC ROC (ValueError: {auc_ve}). Verifique labels/formatos nos dados limpos.")
                              except Exception as auc_e: st.warning(f"Erro inesperado ao calcular AUC ROC: {auc_e}")

                         elif is_binary_alg: st.info("AUC ROC: N/A (Probabilidades não disponíveis ou formato de output inesperado).") # Binary but missing proba or shape wrong.
                         # Not binary, AUC ROC not applicable.


                         # --- Exibe Matriz de Confusão (TESTE) ---
                         st.markdown('<h2 class="sub-header">Matriz de Confusão</h2>', unsafe_allow_html=True)
                         # Plotagem da CM: usa os dados STRING limpos (y_true_cleaned_for_report_cm, y_pred_cleaned_for_report_cm)
                         # e passa CLASS_NAMES se binário para garantir rótulos corretos.
                         try:
                             # plot_confusion_matrix_interactive lida com o cálculo da CM e plot.
                             # Passa as labels STRING consistentes.
                             # Pass class_names based on binary flag (CLASS_NAMES if binary, None if multi-class).
                             cm_class_names = CLASS_NAMES if is_binary_alg and 'CLASS_NAMES' in globals() and len(CLASS_NAMES) >= 2 else None
                             # If cm_class_names is None, plot_confusion_matrix_interactive should infer from data.

                             fig_cm_alg, cm_matrix_alg = plot_confusion_matrix_interactive(
                                 y_true=y_true_cleaned_for_report_cm, # String labels cleaned
                                 y_pred=y_pred_cleaned_for_report_cm, # String labels cleaned
                                 class_names=cm_class_names # Pass CLASS_NAMES if binary, else let infer
                             )

                             if fig_cm_alg: # Check if plotting was successful
                                 st.plotly_chart(fig_cm_alg, use_container_width=True)
                             else: # Error message already shown by plot_confusion_matrix_interactive
                                 cm_matrix_alg = None # Ensure it's None if plotting failed

                         except Exception as e: st.error(f"❌ Não foi possível plotar a Matriz de Confusão: {e}")


                         # --- Interpretação da Matriz de Confusão (APENAS PARA CLASSIFICAÇÃO BINÁRIA 2x2) ---
                         # Check if it's binary AND if the CM was successfully computed and is 2x2.
                         if is_binary_alg and 'CLASS_NAMES' in globals() and len(CLASS_NAMES) == 2 and 'cm_matrix_alg' in locals() and isinstance(cm_matrix_alg, np.ndarray) and cm_matrix_alg.shape == (2, 2):
                              st.markdown("---")
                              st.markdown('<h3 class="sub-header">Interpretação da Matriz de Confusão (Classificação Binária)</h3>', unsafe_allow_html=True)
                              # Access indices 0/1. Requires that plot_confusion_matrix_interactive used the order CLASS_NAMES[0], CLASS_NAMES[1].
                              # The plot function now tries to ensure this or uses inferred labels.
                              # Check the labels actually used by the plotting function/confusion_matrix calculation.
                              try:
                                  # Re-calculate CM just to be sure about the order based on the cleaned string labels
                                  # Use the unique sorted labels from the cleaned data as the basis
                                  actual_labels_used = sorted(list(set(y_true_cleaned_for_report_cm.unique()) | set(y_pred_cleaned_for_report_cm.unique())))
                                  actual_labels_used = [lbl for lbl in actual_labels_used if lbl != 'nan'] # Ensure 'nan' is not a label

                                  # Check if the expected binary labels are present and in the correct order
                                  if actual_labels_used == [CLASS_NAMES[0], CLASS_NAMES[1]]:
                                      # CM should be in the order [CLASS_NAMES[0], CLASS_NAMES[1]]
                                      # CM[0,0] = TN (Real: CLASS_NAMES[0], Pred: CLASS_NAMES[0])
                                      # CM[0,1] = FP (Real: CLASS_NAMES[0], Pred: CLASS_NAMES[1])
                                      # CM[1,0] = FN (Real: CLASS_NAMES[1], Pred: CLASS_NAMES[0])
                                      # CM[1,1] = TP (Real: CLASS_NAMES[1], Pred: CLASS_NAMES[1])
                                       tn, fp, fn, tp = cm_matrix_alg[0,0], cm_matrix_alg[0,1], cm_matrix_alg[1,0], cm_matrix_alg[1,1]

                                       st.write(f"**Verdadeiros Positivos (TP):** {tp}")
                                       st.write(f"**Verdadeiros Negativos (TN):** {tn}")
                                       st.write(f"**Falsos Positivos (FP):** {fp}")
                                       st.write(f"**Falsos Negativos (FN):** {fn}")
                                       st.info(f"""TP: Real '{CLASS_NAMES[1]}', Previsto '{CLASS_NAMES[1]}' | TN: Real '{CLASS_NAMES[0]}', Previsto '{CLASS_NAMES[0]}' | FP: Real '{CLASS_NAMES[0]}', Previsto '{CLASS_NAMES[1]}' | FN: Real '{CLASS_NAMES[1]}', Previsto '{CLASS_NAMES[0]}'.
                                       """)
                                       # Adjusted warning message for clarity.
                                       st.warning(f"💡 **Falsos Positivos (FP):** Alunos que **não** precisavam de ajuda (Real: '{CLASS_NAMES[0]}') foram incorretamente identificados como necessitando (Previsto: '{CLASS_NAMES[1]}'). Custam recursos desnecessários. **Falsos Negativos (FN):** Alunos que **precisavam** de ajuda (Real: '{CLASS_NAMES[1]}') foram incorretamente identificados como não necessitando (Previsto: '{CLASS_NAMES[0]}'). Resultam na perda de oportunidade de intervenção.")
                                  elif len(actual_labels_used) == 2: # Binary, but labels are in a different order or different names
                                        st.warning(f"A Matriz de Confusão tem 2 classes, mas os nomes ou a ordem das classes não correspondem diretamente a '{CLASS_NAMES[0]}' e '{CLASS_NAMES[1]}' (labels encontradas: {actual_labels_used}). Não é possível fornecer a interpretação padrão de TP/TN/FP/FN.")
                                  else: # Not 2 labels after cleaning, or other issue.
                                       st.warning(f"Após limpeza, a Matriz de Confusão não resultou numa matriz 2x2 com as classes binárias esperadas. (Labels encontradas: {actual_labels_used})")


                              except IndexError: st.warning("Erro ao aceder índices da matriz de confusão. A matriz calculada pode não ser 2x2 ou a ordem das classes pode ser inesperada.")
                              except Exception as cm_extract_e: st.warning(f"Erro ao extrair TP/TN/FP/FN: {cm_extract_e}")
                         # If not binary_alg or cm not 2x2, this section is skipped.


                    # --- Análise de Interpretabilidade Específica do Algoritmo Selecionado ---
                    # This part runs for the TEST evaluation method because a single model was trained.
                    st.markdown('---')
                    st.markdown(f'<h3 class="sub-header">Análise Única do Algoritmo: {selected_model_name_alg}</h3>', unsafe_allow_html=True)
                    st.markdown(f'<p class="info-text">Informações de interpretabilidade específicas para este tipo de modelo ({selected_model_name_alg}) treinado temporariamente nos dados processados.</p>', unsafe_allow_html=True) # PT-PT

                    # `processed_cols` contém os nomes das características APÓS pré-processamento.
                    feature_names_processed_list = processed_cols # Rename for consistency in this block.

                    # --- Interpretabilidade específica baseada no `selected_model_name_alg` ---
                    # Uses `model_instance_to_eval` which is the trained instance.

                    # (O código de interpretabilidade por modelo mantém-se o mesmo)
                    # Certifique-se que plot_tree e matplotlib.pyplot (plt) estão importados e que plt.close() é chamado após cada plotagem.
                    # Certifique-se que px (plotly.express) está importado.

                    if selected_model_name_alg == "Árvore de Decisão": # Plot Decision Tree (Matplotlib).
                         st.write("#### Visualização da Árvore")
                         st.info(f"A visualização mostra a árvore de decisão treinada, limitada à profundidade **{model_params_alg.get('max_depth', 'configurada ou default')}** ou até 6 níveis para clareza.")
                         # ... (código de plotagem da árvore de decisão) ...
                         tree_actual_depth = model_instance_to_eval.get_depth() if hasattr(model_instance_to_eval, 'get_depth') else None
                         max_visual_limit = 6 # Visual limit for plot.
                         # Determine the actual depth to plot based on max_depth param, actual tree depth, and visual limit
                         depth_to_plot_dt = float('inf')
                         if model_params_alg.get('max_depth') is not None: depth_to_plot_dt = min(depth_to_plot_dt, model_params_alg['max_depth'])
                         if tree_actual_depth is not None: depth_to_plot_dt = min(depth_to_plot_dt, tree_actual_depth)
                         depth_to_plot_dt = int(min(depth_to_plot_dt, max_visual_limit)) # Also limit by visual limit

                         if tree_actual_depth is not None and tree_actual_depth == 0: st.info("Árvore tem profundidade 0. Nada a visualizar.")
                         elif tree_actual_depth is not None and depth_to_plot_dt > 0 and feature_names_processed_list and not X_train_processed_alg.empty: # Only plot if tree is not trivial, plot depth > 0, feature names exist, and training data wasn't empty.
                              try:
                                   # Determine class names for plotting based on binary and CLASS_NAMES
                                   plot_class_names = [str(c) for c in CLASS_NAMES] if is_binary_alg and 'CLASS_NAMES' in globals() and len(CLASS_NAMES) >= 2 else [str(c) for c in model_instance_to_eval.classes_]

                                   fig_width_dt = max(20, len(feature_names_processed_list) * 0.3); fig_width_dt = min(fig_width_dt, 60) # Adjusted size limits.
                                   fig_height_dt = max(8, depth_to_plot_dt * 1.5); fig_height_dt = min(fig_height_dt, 60) # Adjusted size limits.

                                   fig_tree_dt, ax_tree_dt = plt.subplots(figsize=(fig_width_dt, fig_height_dt))

                                   plot_tree(model_instance_to_eval, ax=ax_tree_dt, filled=True,
                                            feature_names=feature_names_processed_list,
                                            class_names=plot_class_names,
                                            rounded=True, fontsize=9, max_depth=depth_to_plot_dt, impurity=False, node_ids=False, proportion=True)
                                   st.pyplot(fig_tree_dt)
                              except Exception as tree_e: st.error(f"❌ Não foi possível gerar a visualização da árvore: {tree_e}.")
                              finally: plt.close(fig_tree_dt) # ESSENTIAL: Close matplotlib figures.
                         else: st.info("Não foi possível plotar a árvore (verifique profundidade, features ou dados de treino).")

                    elif selected_model_name_alg == "Random Forest": # Feature Importance & 1st tree plot (Matplotlib/Plotly).
                         # Feature Importance for the ensemble.
                         st.write("#### Importância das Características (Ensemble)")
                         if hasattr(model_instance_to_eval, 'feature_importances_') and len(model_instance_to_eval.feature_importances_) == len(feature_names_processed_list):
                              try:
                                   importance_df_rf = pd.DataFrame({'Característica Processada': feature_names_processed_list, 'Importância': model_instance_to_eval.feature_importances_}).sort_values('Importância', ascending=False)
                                   fig_imp_rf = px.bar(importance_df_rf.head(min(20, len(importance_df_rf))), x='Importância', y='Característica Processada', orientation='h', color='Importância', color_continuous_scale=px.colors.sequential.Viridis, title=f"Importância das Características ({selected_model_name_alg})")
                                   fig_imp_rf.update_layout(yaxis={'categoryorder':'total ascending'})
                                   st.plotly_chart(fig_imp_rf, use_container_width=True)
                                   st.info("Importância agregada das características através das árvores do ensemble.")
                              except Exception as e: st.error(f"❌ Erro ao exibir importância de features RF: {e}")
                         else: st.warning("Importância das características RF não disponível ou incompatível.")

                         # Plot first tree (optional).
                         st.write("#### Visualização da Primeira Árvore (Exemplo)")
                         st.info(f"O Random Forest ({selected_model_name_alg}) usa {model_instance_to_eval.n_estimators if hasattr(model_instance_to_eval, 'n_estimators') else 'várias'} árvores. Visualização da primeira (`estimators_[0]`). Ajuste Profundidade Máxima nas configurações para simplificar.")
                         if hasattr(model_instance_to_eval, 'estimators_') and len(model_instance_to_eval.estimators_) > 0:
                             first_tree = model_instance_to_eval.estimators_[0]
                             tree_actual_depth = first_tree.get_depth() if hasattr(first_tree, 'get_depth') else None
                             max_visual_limit = 6
                             # Determine the actual depth to plot for this example tree
                             depth_to_plot_rf_tree = float('inf')
                             if model_params_alg.get('max_depth') is not None: depth_to_plot_rf_tree = min(depth_to_plot_rf_tree, model_params_alg['max_depth'])
                             if tree_actual_depth is not None: depth_to_plot_rf_tree = min(depth_to_plot_rf_tree, tree_actual_depth)
                             depth_to_plot_rf_tree = int(min(depth_to_plot_rf_tree, max_visual_limit))

                             if tree_actual_depth is not None and tree_actual_depth == 0: st.info("Primeira árvore tem profundidade 0.")
                             elif tree_actual_depth is not None and depth_to_plot_rf_tree > 0 and feature_names_processed_list and not X_train_processed_alg.empty:
                                  try:
                                     # Determine class names for plotting based on binary and CLASS_NAMES
                                     plot_class_names = [str(c) for c in CLASS_NAMES] if is_binary_alg and 'CLASS_NAMES' in globals() and len(CLASS_NAMES) >= 2 else [str(c) for c in first_tree.classes_]

                                     fig_width_rf_tree = max(20, len(feature_names_processed_list) * 0.3); fig_width_rf_tree = min(fig_width_rf_tree, 60)
                                     fig_height_rf_tree = max(8, depth_to_plot_rf_tree * 1.5); fig_height_rf_tree = min(fig_height_rf_tree, 60)
                                     fig_tree_rf, ax_tree_rf = plt.subplots(figsize=(fig_width_rf_tree, fig_height_rf_tree))

                                     plot_tree(first_tree, ax=ax_tree_rf, filled=True,
                                              feature_names=feature_names_processed_list,
                                              class_names=plot_class_names , # Use CLASS_NAMES if available and binary, else model's classes
                                              rounded=True, fontsize=9, max_depth=depth_to_plot_rf_tree, impurity=False, node_ids=False, proportion=True)
                                     st.pyplot(fig_tree_rf)
                                  except Exception as tree_e: st.error(f"❌ Não foi possível gerar visualização da 1ª árvore RF: {tree_e}.")
                                  finally: plt.close(fig_tree_rf)
                             else: st.info("Não foi possível plotar a 1ª árvore (verifique profundidade, features ou dados de treino).")
                         else: st.warning("Estimadores da floresta (árvores individuais) não acessíveis após treino.")

                    elif selected_model_name_alg in ["Regressão Logística"] or (selected_model_name_alg.startswith("SVM") and hasattr(model_instance_to_eval, 'coef_') ): # Linear Models (or SVM if it exposes coef_)
                         st.write("#### Coeficientes das Características")
                         if hasattr(model_instance_to_eval, 'coef_'):
                             # Try accessing coefficients (expecting 1D or 2D for binary (1, n_features)).
                             coef_values_check = model_instance_to_eval.coef_
                             # Check if the model is binary or multi-class to correctly interpret coef_ shape
                             if is_binary_alg and hasattr(coef_values_check, 'ndim') and (coef_values_check.ndim == 1 or (coef_values_check.ndim == 2 and coef_values_check.shape[0] == 1)):
                                  coef_values = coef_values_check[0] if coef_values_check.ndim == 2 else coef_values_check
                                  # Check length against processed features.
                                  if len(coef_values) == len(feature_names_processed_list):
                                     try:
                                          coef_df = pd.DataFrame({'Característica Processada': feature_names_processed_list, 'Coeficiente': coef_values}).sort_values('Coeficiente', ascending=False)
                                          # Calculate symmetric range.
                                          coef_min, coef_max = coef_df['Coeficiente'].min(), coef_df['Coeficiente'].max()
                                          abs_max_coef = max(abs(coef_min if pd.notna(coef_min) else 0), abs(coef_max if pd.notna(coef_max) else 0)) if (pd.notna(coef_min) or pd.notna(coef_max)) else 1e-9 # Avoid zero range
                                          color_range = [-abs_max_coef, abs_max_coef]
                                          fig_coef = px.bar(coef_df.head(min(30, len(coef_df))), x='Coeficiente', y='Característica Processada', orientation='h', color='Coeficiente', color_continuous_scale='RdBu', range_color=color_range, title=f"Coeficientes ({selected_model_name_alg})")
                                          fig_coef.update_layout(yaxis={'categoryorder':'total ascending'})
                                          st.plotly_chart(fig_coef, use_container_width=True)
                                          st.info(f"""Coeficientes em modelos lineares como {selected_model_name_alg} indicam a influência linear na probabilidade da classe positiva ('{CLASS_NAMES[1]}' se definido e binário). Magnitude = importância, Sinal = direção do efeito.""")
                                     except Exception as e: st.error(f"❌ Erro ao exibir coeficientes: {e}")
                                  else: st.warning("Número de coeficientes não corresponde às características processadas.")
                             elif hasattr(coef_values_check, 'ndim') and coef_values_check.ndim > 1: # Multi-class case for coef_ (ndim > 1 rows)
                                  st.warning(f"Modelo {selected_model_name_alg} é multi-classe ou tem estrutura `coef_` diferente do esperado para visualização binária por feature. Não é possível exibir coeficientes por característica individual desta forma.")
                             else: st.warning(f"Atributo `coef_` do modelo {selected_model_name_alg} tem estrutura inesperada para visualização.") # Not 1D, Not (1, n_features)
                         else: st.info(f"Modelo {selected_model_name_alg} não possui o atributo `coef_` (geralmente modelos não-lineares ou kernels SVM específicos).")


                    elif selected_model_name_alg == "KNN": # Info sobre KNN
                          st.write("#### Princípios Chave do KNN")
                          n_neighbors_val = model_instance_to_eval.n_neighbors if hasattr(model_instance_to_eval, 'n_neighbors') else 'N/A'
                          st.info(f"O algoritmo **K-Nearest Neighbors ({selected_model_name_alg})** classifica uma amostra com base na classe majoritária dos seus **{n_neighbors_val}** vizinhos mais próximos nos dados de treino processados. Não tem interpretabilidade direta via coeficientes ou importâncias de features como outros modelos.")

                    elif selected_model_name_alg in ["Gradient Boosting", "AdaBoost"]: # Info sobre Boosting + Feature Importance
                         st.write(f"#### Princípios de Modelos de Boosting ({selected_model_name_alg})")
                         st.info(f"Modelos de Boosting ({selected_model_name_alg}) constroem iterativamente ensembles de modelos fracos, corrigindo erros anteriores. São poderosos mas menos interpretáveis diretamente que modelos simples.")
                         # Feature Importance for Boosting if available and consistent.
                         if hasattr(model_instance_to_eval, 'feature_importances_') and len(model_instance_to_eval.feature_importances_) == len(feature_names_processed_list):
                             st.write(f"#### Importância das Características ({selected_model_name_alg})")
                             try:
                                 importance_df_boost = pd.DataFrame({'Característica Processada': feature_names_processed_list, 'Importância': model_instance_to_eval.feature_importances_}).sort_values('Importância', ascending=False)
                                 fig_imp_boost = px.bar(importance_df_boost.head(min(20, len(importance_df_boost))), x='Importância', y='Característica Processada', orientation='h', color='Importância', color_continuous_scale=px.colors.sequential.Viridis, title=f"Importância das Características ({selected_model_name_alg})")
                                 fig_imp_boost.update_layout(yaxis={'categoryorder':'total ascending'})
                                 st.plotly_chart(fig_imp_boost, use_container_width=True)
                                 st.info("Importância agregada das características na redução de erro durante o processo de boosting.")
                             except Exception as e: st.error(f"❌ Erro ao exibir importância de features Boosting: {e}")
                         else: st.warning("Importância das características Boosting não disponível ou incompatível.")

                    else: # Outros modelos em AVAILABLE_MODELS_FOR_ANALYSIS sem análise específica.
                          st.info(f"Análise de interpretabilidade específica para **{selected_model_name_alg}** não está configurada nesta aplicação.")


                elif evaluation_method == "Cross-Validation (StratifiedKFold)":
                    # --- CROSS-VALIDATION EVALUATION ---
                    st.write(f"A executar {n_splits}-Fold Cross-Validation para ({type(model_instance_to_eval).__name__})...")

                    # Ensure cleaned data for CV is available and sufficient (should be guaranteed by can_do_cv)
                    if X_train_cleaned_for_cv is None or y_train_cleaned_for_cv is None or len(X_train_cleaned_for_cv) < n_splits:
                         st.error("Dados de treino limpos insuficientes para executar Cross-Validation com o número de folds selecionado.")
                         raise ValueError("Insufficient cleaned data for CV.") # Stop execution gracefully


                    # Define scoring metrics
                    scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
                    # Add AUC if binary and model supports probabilities
                    if is_binary_alg:
                         if hasattr(model_instance_to_eval, 'predict_proba'):
                             # cross_validate needs the name 'roc_auc' for binary classification AUC
                             scoring.append('roc_auc')
                         else:
                             st.warning("AUC ROC não incluída na Cross-Validation: Modelo não suporta `predict_proba`.")
                    # For multi-class, roc_auc isn't directly applicable with this standard scorer name.
                    # Could add multi_class AUC ('roc_auc_ovr' or 'roc_auc_ovo') but keep it simple for now.


                    # Define the CV splitter
                    # Use StratifiedKFold if binary or multi-class classification is possible (more than 1 unique class)
                    # Otherwise, if only 1 class, KFold might be used or CV is impossible anyway.
                    # We've already checked for >= 2 unique classes in cleaned training data if binary.
                    # StratifiedKFold works fine for multi-class as well.
                    if len(y_train_cleaned_for_cv.unique()) >= 2:
                         cv_splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42) # Add random_state for reproducibility
                    else:
                         # This case should ideally be caught by can_do_cv checks earlier, but double-check.
                         st.error("Cross-Validation (StratifiedKFold) requer pelo menos 2 classes no conjunto de treino limpo.")
                         raise ValueError("Insufficient classes for StratifiedKFold.")


                    # Perform cross-validation
                    with st.spinner(f"A executar {n_splits}-fold cross-validation..."):
                        cv_results = cross_validate(
                            estimator=model_instance_to_eval,
                            X=X_train_cleaned_for_cv, # Use cleaned training data
                            y=y_train_cleaned_for_cv, # Use cleaned training data (numeric 0/1 or original multi-class)
                            cv=cv_splitter,
                            scoring=scoring,
                            return_train_score=False # Typically only test scores are reported for evaluation
                        )

                    st.success(f"✅ Cross-Validation ({n_splits}-Fold) completa para '{selected_model_name_alg}'.")

                    # Display results
                    st.markdown('<h3 class="sub-header">Resultados da Cross-Validation</h3>', unsafe_allow_html=True)

                    # Prepare results dictionary for display
                    cv_summary = {}
                    for metric in scoring:
                        # cross_validate names keys like 'test_accuracy', 'test_precision_weighted', etc.
                        test_metric_key = f'test_{metric}'
                        if test_metric_key in cv_results:
                            scores = cv_results[test_metric_key]
                            cv_summary[metric] = {
                                'Média': scores.mean(),
                                'Std Dev': scores.std()
                            }
                        # else: Metric was in scoring list but not returned? (e.g. roc_auc for non-classifier).
                        # cross_validate handles this by not returning the key if scoring fails for a metric.
                        # So, just skip if key is not in cv_results.

                    if cv_summary: # Only display if we got results
                        cv_summary_df = pd.DataFrame.from_dict(cv_summary, orient='index')
                        st.dataframe(cv_summary_df.round(4), use_container_width=True)
                    else:
                         st.warning("Nenhuma métrica calculada durante a Cross-Validation.")


                    st.info(f"""💡 **Cross-Validation ({n_splits}-Fold):** Avalia o desempenho do modelo dividindo o conjunto de treino em {n_splits} partes (folds). O modelo é treinado {n_splits} vezes, cada vez usando {n_splits-1} folds para treino e 1 fold para validação. As métricas apresentadas são a média dos resultados de cada fold de validação. Isto dá uma estimativa mais robusta do desempenho em dados não vistos comparada a um único split treino/teste.

                    **Nota:** A Matriz de Confusão e o Relatório de Classificação detalhado são tipicamente apresentados para uma única avaliação (como num conjunto de teste separado) ou de forma agregada mais complexa em CV. Estes não são exibidos diretamente aqui para a avaliação por Cross-Validation.
                    """)


                    # --- Análise de Interpretabilidade Específica do Algoritmo Selecionado (SKIPPED FOR CV) ---
                    st.markdown("---")
                    st.markdown('<h3 class="sub-header">Análise de Interpretabilidade</h3>', unsafe_allow_html=True)
                    st.info("A análise de interpretabilidade (viz. importância de features, coeficientes) para a Cross-Validation exigiria treinar um modelo separado no conjunto de treino completo ou analisar múltiplos modelos das folds. Esta secção não está disponível para este método de avaliação.")


                # --- End Conditional Evaluation Logic ---

            # --- Fim do Bloco Try para Treino e Avaliação Dinâmica ---
            except ValueError as ve:
                 st.error(f"❌ Ocorreu um erro de VALOR durante a execução ({evaluation_method}): {ve}")
                 st.info("Verifique se os dados processados são compatíveis ou se há um problema específico com a configuração do algoritmo ou CV.")

            except Exception as e:
                 # --- Existing error handling for train/eval block ---
                 st.error(f"❌ Ocorreu um erro inesperado durante a avaliação dinâmica do algoritmo {selected_model_name_alg} ({evaluation_method}): {e}")
                 st.info("Verifique se os dados processados são compatíveis com o algoritmo selecionado ou se há um problema específico com a configuração dos parâmetros.")
                 st.error(f"Detalhe do erro: {e}") # Display detail.

    else: # can_run_evaluation is False.
         # Messages already displayed by the initial checks or the can_do_cv checks above.
         if evaluation_method == "Cross-Validation (StratifiedKFold)" and not can_do_cv and is_base_data_loaded:
              # Specific message for CV not possible despite base data being loaded
              st.warning("Não é possível prosseguir com Cross-Validation devido a problemas nos dados de treino limpos ou configuração do número de folds.")
         elif not is_base_data_loaded:
              st.warning("A secção 'Avaliação de Modelos (CM)' não está completamente funcional porque os conjuntos de dados processados ou os nomes das características processadas não foram carregados com sucesso. Verifique as mensagens no topo da página.")
         # No button is shown if can_run_evaluation is False.

# --- Secção "Documentação" ---
# Fornece informação detalhada sobre o dataset, artefactos, e funcionalidades da app.
# Refatorada e totalmente em PT-PT. Inclui mapeamentos ordinais.
elif menu == "Documentação": # Menu option name as defined in the sidebar.
    st.markdown('<h1 class="main-header">Documentação e Informação Adicional</h1>', unsafe_allow_html=True) # Title in PT-PT.
    st.markdown('<p class="info-text">Bem-vindo à secção de documentação, onde poderá encontrar mais detalhes sobre a aplicação, o dataset utilizado e os modelos.</p>', unsafe_allow_html=True) # Intro text in PT-PT.

    # --- Sobre o Dataset ---
    st.markdown('<h2 class="sub-header">Sobre o Dataset Estudantil</h2>', unsafe_allow_html=True)
    st.markdown(f"""
    Esta aplicação utiliza como base o dataset **`student-data.csv`**, geralmente proveniente do repositório UCI Machine Learning. Contém características diversas sobre alunos. A tarefa focada aqui é a classificação binária para prever se um aluno Passará ou Não Passará no exame final.

    Para uma análise visual e estatística do dataset original (estrutura, distribuições, correlações), visite a secção **"Exploração de Dados"**.
    """)

    # --- Descrição Detalhada das Características Originais ---
    st.markdown('### Descrição Detalhada das Características (Features Originais)', unsafe_allow_html=True)
    st.markdown('<p class="info-text">Lista das características (colunas) presentes no dataset original `student-data.csv` e o seu significado:</p>', unsafe_allow_html=True)
    # Uses the global `full_feature_descriptions` dictionary.
    if 'full_feature_descriptions' in globals() and full_feature_descriptions is not None:
         # Iterates through the dictionary and formats each entry.
         for feature, desc in full_feature_descriptions.items():
            if feature == TARGET_ORIGINAL_NAME:
                 st.markdown(f"- **{feature.replace('_', ' ').title()}**: {desc} **(Variável Alvo)**") # Highlight Target.
            else:
                 st.markdown(f"- **{feature.replace('_', ' ').title()}**: {desc}")
    else: st.error("Não foi possível carregar a descrição completa das características originais.")

    # --- Mapeamentos de Características Ordinais Numéricas ---
    st.markdown('### Mapeamento de Níveis para Características Ordinais Numéricas', unsafe_allow_html=True)
    st.markdown('<p class="info-text">As características abaixo são armazenadas com valores numéricos no dataset, mas representam níveis ordenados. O significado de cada valor é:</p>', unsafe_allow_html=True)
    # Uses the global `ORDINAL_MAPPINGS` dictionary.
    if 'ORDINAL_MAPPINGS' in globals() and ORDINAL_MAPPINGS is not None:
         for feature, mapping in ORDINAL_MAPPINGS.items():
             st.markdown(f"**{feature.replace('_', ' ').title()}:**")
             for key, value in mapping.items():
                 st.markdown(f"- `{key}`: {value}") # List each mapping key: value.
    else: st.warning("Não foi possível carregar os mapeamentos ordinais.")

    # --- Sobre Modelos e Artefactos ---
    st.markdown('<h2 class="sub-header">Sobre os Modelos e Artefactos de Machine Learning</h2>', unsafe_allow_html=True)
    st.markdown("""
    O sistema utiliza modelos de Machine Learning previamente treinados para prever o desempenho estudantil. Estes modelos e outros objetos essenciais (artefactos) são guardados em ficheiros binários (`.joblib`) e carregados pela aplicação.

    Os artefactos principais utilizados são:
    *   **`artefacts/preprocessor.joblib`**: Contém a pipeline de pré-processamento (transformações nas características dos dados de entrada, como escalamento ou codificação de variáveis categóricas). É aplicado aos dados brutos ANTES de serem fornecidos a um modelo para previsão ou treino.
    *   **`artefacts/best_model.joblib`**: Representa o **Modelo Principal** de Machine Learning, que foi treinado nos dados processados e selecionado pelo seu desempenho na fase de desenvolvimento/teste. Este modelo é usado por defeito para fazer **previsões individuais**.
    *   **`artefacts/original_input_columns.joblib`**: Uma lista dos nomes exatos das colunas que a aplicação espera receber como input bruto (baseado no dataset original `student-data.csv`). Essencial para estruturar corretamente os dados para o pré-processador.
    *   **`artefacts/processed_feature_names.joblib`**: Uma lista dos nomes das características NO FORMATO EM QUE OS MODELOS FORAM TREINADOS, após a aplicação do pré-processamento. Essencial para associar corretamente as características aos resultados de interpretabilidade do modelo.

    Outros ficheiros `.joblib` na pasta `artefacts/` podem corresponder a modelos de outros algoritmos. Pode experimentá-los na secção **"Previsão Individual"** ou **"Avaliação de Modelos (CM)"**.

    Para uma análise aprofundada do desempenho e da interpretabilidade (quais características foram mais importantes) do **Modelo Principal (`best_model.joblib`)**, visite a secção **"Análise do Modelo Treinado Principal"**.
    """) # PT-PT explanation.

    # --- Sobre a Avaliação e Comparação de Modelos (CM) ---
    st.markdown('<h2 class="sub-header">Sobre a Avaliação e Comparação de Modelos</h2>', unsafe_allow_html=True)
    st.markdown("""
    A secção **"Avaliação de Modelos (CM)"** é uma ferramenta interativa para explorar e comparar o desempenho de diferentes tipos de algoritmos de Machine Learning no problema de classificação (prever "Passar" ou "Não Passar").

    Nesta secção, pode:
    1.  **Selecionar** um algoritmo específico (como Árvore de Decisão, Random Forest, SVM, etc.) da lista de algoritmos configurados.
    2.  **Configurar (opcionalmente)** alguns parâmetros chave desse algoritmo via sliders.
    3.  Clicar num botão para **treinar *temporariamente*** esse algoritmo nos dados de treino processados e avaliá-lo *imediatamente* nos dados de teste processados.
    4.  Visualizar as **métricas de avaliação** (Acurácia, Precision, Recall, F1-Score) e a crucial **Matriz de Confusão**, que mostra Verdadeiros Positivos/Negativos e Falsos Positivos/Negativos, para entender os erros que o modelo comete.
    5.  Ver análises de **interpretabilidade específicas** do algoritmo (Importância de Características ou Coeficientes) para ter uma ideia de quais fatores (características processadas) foram mais relevantes para o seu treino.

    É importante notar que o treino e avaliação realizados nesta secção são apenas para **comparação e demonstração *dinâmica***. **Não modificam** o `best_model.joblib` guardado que serve como o modelo principal da aplicação.
    """) # PT-PT explanation for the refactored section.

    # --- Sugestões de Próximos Passos ---
    st.markdown('<h2 class="sub-header">Sugestões de Próximos Passos e Potenciais Melhorias</h2>', unsafe_allow_html=True)
    st.markdown("""
    Este sistema oferece um ponto de partida sólido. Possíveis evoluções e melhorias incluem:
    *   **Testar Outros Modelos/Algoritmos:** Integrar mais algoritmos de ML (redes neuronais, ensembles mais avançados) nas secções de Avaliação ou guardar novos modelos otimizados.
    *   **Otimização de Hiperparâmetros:** Adicionar ferramentas de otimização (busca em grid, busca aleatória) na UI da secção de Avaliação de Modelos (CM) para ajudar a encontrar os melhores parâmetros para os algoritmos temporários.
    *   **Análise de Erros Detalhada:** Incluir visualizações focadas em analisar os casos específicos (amostras) que resultaram em Falsos Positivos ou Falsos Negativos, para identificar padrões nos erros do modelo principal.
    *   **Técnicas de Interpretabilidade Mais Avançadas:** Explorar métodos como SHAP (SHapley Additive exPlanations) ou LIME (Local Interpretable Model-agnostic Explanations) para fornecer insights mais profundos sobre as previsões, tanto a nível global (para o modelo) como individual (para uma previsão específica).
    *   **Validação Cruzada (Cross-Validation):** Adicionar a opção de realizar validação cruzada na secção de Avaliação de Modelos (CM) para obter métricas de desempenho mais robustas e menos sensíveis a um único split de treino/teste.
    *   **Feedback do Utilizador:** Capturar feedback sobre a utilidade da previsão ou sobre alunos em risco (identificados como "Não Passar") para validar/melhorar o modelo ao longo do tempo.
    """) # PT-PT suggestions.


# --- Fim das Secções do Menu ---


# --- Rodapé da Aplicação ---
# Fica no final do script, garantindo que aparece em todas as páginas por baixo do conteúdo principal.
st.markdown("---") # Separador horizontal visual.
st.markdown("© 2025 Sistema de Intervenção Estudantil. Desenvolvido com Streamlit.") # PT-PT rodapé.
