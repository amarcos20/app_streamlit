import streamlit as st
import pandas as pd
import numpy as np
# import seaborn as sns # Note: Seaborn imported but not used in the provided code for plotting
# from sklearn.metrics import ConfusionMatrixDisplay # Note: ConfusionMatrixDisplay imported but not used directly in the provided code for plotting
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score

# Importar os modelos específicos utilizados ou referenciados
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt # Para exibir a árvore de decisão


import time
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu

import joblib
import os

# --- Configuração da Página ---
st.set_page_config(
    page_title="Sistema de Intervenção Estudantil",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Estilo CSS Personalizado ---
st.markdown("""
<style>
    /* Headers */
    .main-header {
        font-size: 2.8rem;
        color: #1A237E;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 2rem;
        color: #283593;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: bold;
        border-bottom: 2px solid #C5CAE9;
        padding-bottom: 0.5rem;
    }
    .info-text {
        font-size: 1rem;
        color: #424242;
        margin-bottom: 1rem;
        line-height: 1.6;
    }
    /* Cards */
    .metric-card {
        background-color: #E8EAF6;
        border-left: 6px solid #3F51B5;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
     /* Style for st.metric value - Streamlit's built-in metric uses different classes */
    div[data-testid="stMetric"] label div {
        font-size: 1rem !important;
        color: #555 !important;
    }
     div[data-testid="stMetric"] div[data-testid="stMetricDelta"] div {
         font-size: 1.8rem !important;
         font-weight: bold !important;
         color: #1A237E !important;
     }
    /* Button */
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
     /* Adjust sidebar width */
    section[data-testid="stSidebar"] {
        width: 300px !important;
        background-color: #f1f3f4;
    }
    /* Style for tabs */
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
     /* Style for st.info, st.warning, st.error */
    div[data-testid="stAlert"] {
        font-size: 1rem;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1.5rem;
    }

    /* CSS para o Grid Layout na Previsão Individual */
    .input-grid-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); /* 4 colunas (ajustável), min 250px largura */
        gap: 20px; /* Espaçamento entre os itens do grid */
        margin-bottom: 20px;
    }
    .grid-item {
        display: flex; /* Usa flexbox dentro do item para empilhar conteúdo */
        flex-direction: column;
        /* Optional: add padding or border for visual separation */
        /* border: 1px solid #e0e0e0; */
        /* padding: 15px; */
        /* border-radius: 8px; */
    }
     /* Ajusta o estilo do st.info para caber melhor */
    .grid-item div[data-testid="stAlert"] {
        margin-top: 5px; /* Pequena margem acima do info box */
        margin-bottom: 5px; /* Pequena margem abaixo do info box */
        padding: 10px; /* Reduz padding interno */
        font-size: 0.9rem; /* Reduz tamanho da fonte */
        line-height: 1.4;
    }

    /* Opcional: Ajustar o tamanho do st.number_input e st.selectbox dentro do grid item */
     .grid-item div[data-testid="stNumberInput"],
     .grid-item div[data-testid="stselectbox"] {
         margin-top: 5px;
     }
     /* Estilo para o nome da feature e a descrição no grid item */
     .grid-item .feature-label {
         font-weight: bold; /* Nome da feature em negrito */
         margin-bottom: 5px; /* Espaço abaixo do nome */
     }
     /* CSS para a descrição pequena entre parêntesis */
     .small-description {
         font-size: 0.8em; /* Tamanho menor, 80% do tamanho normal */
         font-weight: normal; /* Não negrito */
         color: #555; /* Cor um pouco mais clara */
         margin-left: 5px; /* Espaço entre o nome e a descrição */
     }


</style>
""", unsafe_allow_html=True)

# --- Mapeamentos para features Ordinais Numéricas ---
ORDINAL_MAPPINGS = {
    'Medu': {0: 'Nenhuma', 1: 'Ensino Fund. (4ª série)', 2: 'Ensino Fund. (5ª-9ª série)', 3: 'Ensino Médio', 4: 'Ensino Superior'},
    'Fedu': {0: 'Nenhuma', 1: 'Ensino Fund. (4ª série)', 2: 'Ensino Fund. (5ª-9ª série)', 3: 'Ensino Médio', 4: 'Ensino Superior'},
    'traveltime': {1: '<15 min', 2: '15-30 min', 3: '30-60 min', 4: '>60 min'},
    'studytime': {1: '<2 horas', 2: '2 a 5 horas', 3: '5 a 10 horas', 4: '>10 horas'},
    'failures': {0: '0 falhas', 1: '1 falha', 2: '2 falhas', 3: '3+ falhas'}, # failures tem valores 0, 1, 2, 3+. O dataset pode ter mais que 3 falhas, mas 3+ é o máximo geralmente visto.
    'famrel': {1: 'Muito Ruim', 2: 'Ruim', 3: 'Regular', 4: 'Bom', 5: 'Excelente'},
    'freetime': {1: 'Muito Pouco', 2: 'Pouco', 3: 'Médio', 4: 'Muito', 5: 'Muito Muito'},
    'goout': {1: 'Muito Raramente', 2: 'Raramente', 3: 'Ocasionalmente', 4: 'Frequentemente', 5: 'Muito Frequentemente'},
    'Dalc': {1: 'Muito Baixo', 2: 'Baixo', 3: 'Médio', 4: 'Alto', 5: 'Muito Alto'},
    'Walc': {1: 'Muito Baixo', 2: 'Baixo', 3: 'Médio', 4: 'Alto', 5: 'Muito Alto'},
    'health': {1: 'Muito Ruim', 2: 'Ruim', 3: 'Regular', 4: 'Bom', 5: 'Excelente'},
    # 'age' e 'absences' são numéricas contínuas/count, não ordinais em escala
}

# Lista de features numéricas que representam escalas ordinais e para as quais temos mapeamentos
ordinal_numeric_features_to_map = list(ORDINAL_MAPPINGS.keys())

# --- Descrição Curta das Características (Para a Previsão Individual) ---
feature_descriptions_short = {
    "school": "Escola",
    "sex": "Gênero",
    "age": "Idade",
    "address": "Residência",
    "famsize": "Tamanho família",
    "Pstatus": "Status pais",
    "Medu": "Escolaridade mãe",
    "Fedu": "Escolaridade pai",
    "Mjob": "Ocupação mãe",
    "Fjob": "Ocupação pai",
    "reason": "Motivo escola",
    "guardian": "Guardião",
    "traveltime": "Tempo viagem",
    "studytime": "Tempo estudo",
    "failures": "Reprovações",
    "schoolsup": "Apoio escola",
    "famsup": "Apoio família",
    "paid": "Aulas pagas",
    "activities": "Atividades extra",
    "nursery": "Frequentou creche",
    "higher": "Deseja superior",
    "internet": "Acesso internet",
    "romantic": "Relacionamento",
    "famrel": "Qualidade relações familiares",
    "freetime": "Tempo livre",
    "goout": "Sair c/amigos",
    "Dalc": "Álcool dias semana",
    "Walc": "Álcool fins semana",
    "health": "Estado saúde",
    "absences": "Faltas",
    "passed": "Aprovado" # Descrição curta aqui para ficar bem na previsão
}

# --- Descrição Completa das Características (Para a Documentação) ---
full_feature_descriptions = {
    "school": "Escola do estudante (GP ou MS)",
    "sex": "Gênero do estudante (F ou M)",
    "age": "Idade do estudante",
    "address": "Localização da residência (Urbana ou Rural)",
    "famsize": "Tamanho da família (Maior que 3 ou Menor/Igual a 3)",
    "Pstatus": "Status de coabitação dos pais (Moram juntos ou Separados)",
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
    "higher": "Deseja cursar ensino superior (yes ou no)",
    "internet": "Tem acesso à internet em casa (yes ou no)",
    "romantic": "Está em relacionamento romântico (yes ou no)",
    "famrel": "Qualidade dos relacionamentos familiares (1: muito ruim a 5: excelente)",
    "freetime": "Tempo livre após a escola (1: muito pouco a 5: muito)",
    "goout": "Frequência com que sai com amigos (1: muito raramente a 5: muito frequentemente)",
    "Dalc": "Consumo de álcool em dias de semana (1: muito baixo a 5: muito alto)",
    "Walc": "Consumo de álcool em fins de semana (1: muito baixo a 5: muito alto)",
    "health": "Estado de saúde atual (1: muito ruim a 5: muito bom)",
    "absences": "Número de faltas escolares",
    "passed": "O estudante foi aprovado (yes ou no) - Variável Alvo" # Mantido "- Variável Alvo" aqui para clareza na documentação
}
# Função para carregar um modelo específico, com caching
@st.cache_resource
def load_specific_model(model_filename):
    artefacts_path = 'artefacts/'
    model_path = os.path.join(artefacts_path, model_filename)
    try:
        loaded_model = joblib.load(model_path)
        # st.success(f"✅ Modelo '{model_filename}' carregado com sucesso!") # Pode ser muito verboso
        return loaded_model
    except FileNotFoundError:
        st.error(f"❌ Erro: O ficheiro do modelo '{model_filename}' não foi encontrado na pasta 'artefacts/'.")
        return None
    except Exception as e:
        st.error(f"❌ Ocorreu um erro ao carregar o modelo '{model_filename}': {e}")
        return None


# Função para exibir animação de carregamento
def loading_animation(text="Processando..."):
    progress_text = text
    my_bar = st.progress(0, text=progress_text)

    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1, text=progress_text)
    time.sleep(0.5)
    my_bar.empty()

# Função para gerar matriz de confusão interativa
def plot_confusion_matrix_interactive(y_true, y_pred, class_names=None):
    cm = confusion_matrix(y_true, y_pred)

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=class_names,
        y=class_names,
        colorscale='Blues',
        showscale=True,
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 20},
        hoverinfo="x+y+z",
    ))

    fig.update_layout(
        title='Matriz de Confusão',
        xaxis_title='Valores Previstos',
        yaxis_title='Valores Reais',
        xaxis=dict(side='top'),
        yaxis=dict(autorange="reversed"),
        margin=dict(t=50, b=50, l=50, r=50),
    )

    return fig, cm

# Função para plotar matriz quadrada com mapa de calor (mais genérica, mantida)
def plot_square_matrix_heatmap(matrix, title="Matriz Quadrada", x_labels=None, y_labels=None):
    matrix_list = [[None if pd.isna(val) else float(val) for val in row] for row in matrix]
    text_matrix = [[None if pd.isna(val) else f"{val:.2f}" for val in row] for row in matrix]

    fig = go.Figure(data=go.Heatmap(
        z=matrix_list,
        x=x_labels,
        y=y_labels,
        colorscale='Viridis',
        showscale=True,
        text=text_matrix,
        texttemplate="%{text}",
        hoverinfo="x+y+z",
    ))

    fig.update_layout(
        title=title,
        margin=dict(t=50, b=50, l=50, r=50),
    )

    return fig

# Função para visualizar matriz de correlação (usando plotly express)
def plot_correlation_matrix_px(df):
    df_numeric = df.select_dtypes(include=np.number)

    if df_numeric.empty:
         return None, None

    corr = df_numeric.corr()

    fig = px.imshow(
        corr,
        labels=dict(color="Correlação"),
        x=corr.columns,
        y=corr.columns,
        color_continuous_scale='RdBu_r',
        range_color=[-1, 1],
        aspect="auto",
        text_auto=".2f",
    )

    fig.update_layout(
        title="Matriz de Correlação",
        margin=dict(t=50, b=50, l=50, r=50),
    )

    return fig, corr

# Função para analisar propriedades de uma matriz quadrada
def analyze_square_matrix(matrix, title="Análise de Matriz"):
    st.markdown(f'<h3 class="sub-header">{title}</h3>', unsafe_allow_html=True)

    if not isinstance(matrix, np.ndarray) or matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        st.error("Input inválido: A matriz deve ser um array NumPy quadrado e 2D.")
        return

    size = matrix.shape[0]
    st.write(f"**Dimensão da matriz:** {size}x{size}")

    trace = np.trace(matrix)
    st.write(f"**Traço da matriz:** {trace:.4f}")
    st.info("O traço é a soma dos elementos na diagonal principal da matriz.")

    try:
        det = np.linalg.det(matrix)
        st.write(f"**Determinante:** {det:.4e}")
        if abs(det) < 1e-9:
            st.warning("⚠️ O determinante é próximo de zero...")
        else:
             st.success("✅ O determinante sugere que a matriz não é singular.")
        st.info("O determinante indica se a matriz é invertível...")
    except np.linalg.LinAlgError:
        st.error("❌ Não foi possível calcular o determinante...")
        det = None
    except Exception as e:
        st.error(f"❌ Ocorreu um erro ao calcular o determinante: {e}")


    st.write("**Valores próprios (Eigenvalues):**")
    try:
        eigenvalues = np.linalg.eigvals(matrix)
        sorted_eigenvalues = np.sort(np.abs(eigenvalues))[::-1]

        # Exibir apenas alguns valores próprios
        max_eigen_display = min(10, len(sorted_eigenvalues)) # Display max 10 eigenvalues
        for i, val in enumerate(sorted_eigenvalues[:max_eigen_display]):
            # Find the original eigenvalue corresponding to this magnitude
            original_val = eigenvalues[np.where(np.isclose(np.abs(eigenvalues), val))[0][0]]
            st.write(f"λ{i+1} (Magnitude) = {val:.4f} (Original: {original_val:.4f})")
        if len(sorted_eigenvalues) > max_eigen_display:
            st.write("...")


        if any(abs(val) < 1e-9 for val in eigenvalues): # Verificar se ALGUM valor próprio é perto de zero
             st.warning("⚠️ Alguns valores próprios são próximos de zero...")
        else:
             st.success("✅ Os valores próprios indicam que a matriz não tem direções nulas...")
        st.info("Valores próprios representam os fatores de escala...")

    except np.linalg.LinAlgError:
        st.error("❌ Não foi possível calcular os valores próprios.")
    except Exception as e:
        st.error(f"❌ Ocorreu um erro ao calcular os valores próprios: {e}")


    try:
        condition_number = np.linalg.cond(matrix)
        st.write(f"**Número de Condição:** {condition_number:.4e}")
        if condition_number > 1000:
            st.warning("⚠️ Alto número de condição. A matriz é mal condicionada...")
        else:
            st.success("✅ Número de condição razoável. A matriz está bem condicionada.")
        st.info("O número de condição mede a sensibilidade...")
    except np.linalg.LinAlgError:
         st.error("❌ Não foi possível calcular o número de condição.")
    except Exception as e:
         st.error(f"❌ Erro ao calcular número de condição: {e}")


# --- Carregar Artefactos Treinados (Refatorado para retornar status e resultado) ---
@st.cache_resource
def load_pipeline_artefacts_safe():
    artefacts_path = 'artefacts/'
    preprocessor_path = os.path.join(artefacts_path, 'preprocessor.joblib')
    model_path = os.path.join(artefacts_path, 'best_model.joblib')
    original_cols_path = os.path.join(artefacts_path, 'original_input_columns.joblib')
    processed_cols_path = os.path.join(artefacts_path, 'processed_feature_names.joblib')

    try:
        preprocessor = joblib.load(preprocessor_path)
        model = joblib.load(model_path)
        original_cols = joblib.load(original_cols_path)
        processed_cols = joblib.load(processed_cols_path)

        st.success("✅ Artefactos do pipeline (pré-processador, modelo e nomes de colunas) carregados com sucesso!")
        return True, (preprocessor, model, original_cols, processed_cols)

    except FileNotFoundError as e:
        error_msg = f"❌ Erro ao carregar artefactos essenciais: {e}. Certifique-se de que todos os ficheiros .joblib estão na pasta '{artefacts_path}' e têm os nomes corretos."
        return False, error_msg
    except Exception as e:
        error_msg = f"❌ Ocorreu um erro inesperado ao carregar artefactos: {e}"
        return False, error_msg

# --- Chamar a função de carregamento e verificar o resultado ---
success_artefacts, loaded_artefacts_result = load_pipeline_artefacts_safe()

if not success_artefacts:
    st.error(loaded_artefacts_result)
    st.stop()
else:
    preprocessor, model, original_cols, processed_cols = loaded_artefacts_result


# --- Carregar o seu Dataset Original para EDA ---
@st.cache_data
def load_student_data():
    data_path = 'student-data.csv'
    try:
        df = pd.read_csv(data_path)
        st.success(f"✅ Dataset '{data_path}' carregado com sucesso ({df.shape[0]} linhas, {df.shape[1]} colunas).")
        return df
    except FileNotFoundError:
        st.error(f"❌ Erro: O ficheiro '{data_path}' não foi encontrado. Certifique-se de que o dataset está no local correto.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Ocorreu um erro ao carregar o dataset: {e}")
        st.stop()

# Carregar o dataset original
student_df_original = load_student_data()

# Identificar a coluna alvo original
TARGET_ORIGINAL_NAME = 'passed'
if TARGET_ORIGINAL_NAME not in student_df_original.columns:
    st.error(f"❌ Coluna alvo original '{TARGET_ORIGINAL_NAME}' não encontrada no dataset. A aplicação pode não funcionar corretamente.")
    # st.stop() # Opcional: Parar se a coluna alvo não existir

# Definir os nomes das classes para a saída da previsão e avaliação
CLASS_NAMES = ['no', 'yes']

# Definir o nome da coluna alvo APÓS o mapeamento (usado no teste processado)
TARGET_PROCESSED_NAME = 'passed_mapped'


# --- Função para carregar os conjuntos de dados processados (treino e teste) ---
@st.cache_data
def load_processed_data(target_col_name):
    processed_train_path = 'data/processed/train_processed.csv'
    processed_test_path = 'data/processed/test_processed.csv'

    train_df_processed = None
    test_df_processed = None
    errors = []

    try:
        train_df_processed = pd.read_csv(processed_train_path)
        if target_col_name not in train_df_processed.columns:
             errors.append(f"❌ Erro: A coluna alvo processada '{target_col_name}' não foi encontrada no ficheiro '{processed_train_path}'.")
             train_df_processed = None
        else:
             st.success(f"✅ Conjunto de treino processado carregado ({train_df_processed.shape[0]} linhas).")
    except FileNotFoundError:
        errors.append(f"⚠️ Ficheiro de treino processado '{processed_train_path}' não encontrado. Algumas funcionalidades podem estar limitadas.")
    except Exception as e:
        errors.append(f"❌ Ocorreu um erro ao carregar o conjunto de treino processado: {e}")
        train_df_processed = None


    try:
        test_df_processed = pd.read_csv(processed_test_path)
        if target_col_name not in test_df_processed.columns:
             errors.append(f"❌ Erro: A coluna alvo processada '{target_col_name}' não foi encontrada no ficheiro '{processed_test_path}'.")
             test_df_processed = None
        else:
             st.success(f"✅ Conjunto de teste processado carregado ({test_df_processed.shape[0]} linhas).")
    except FileNotFoundError:
        errors.append(f"⚠️ Ficheiro de teste processado '{processed_test_path}' não encontrado. Algumas funcionalidades podem estar limitadas.")
    except Exception as e:
        errors.error(f"❌ Ocorreu um erro ao carregar o conjunto de teste processado: {e}")
        test_df_processed = None

    for err in errors:
        st.markdown(err)

    return train_df_processed, test_df_processed

# Carregar os conjuntos de treino e teste processados
train_df_processed_global, test_df_processed_global = load_processed_data(TARGET_PROCESSED_NAME)


# --- Lista de modelos disponíveis para a secção "Análise de Matriz" ---
AVAILABLE_MODELS_FOR_ANALYSIS = {
    "Regressão Logística": LogisticRegression(random_state=42, max_iter=1000),
    "KNN": KNeighborsClassifier(),
    "Árvore de Decisão": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM (Kernel RBF)": SVC(probability=True, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42)
}


# --- Sidebar para navegação ---
with st.sidebar:
    # --- Adicionar espaço para o Logotipo ---
    # Para adicionar o seu logotipo:
    # 1. Coloque o ficheiro da imagem (ex: logo.png) na mesma pasta do seu script Streamlit
    #    ou numa subpasta (ex: assets/logo.png).
    # 2. Descomente a linha abaixo que começa com 'st.image('.
    # 3. Substitua '"caminho/para/o/seu/logotipo.png"' pelo caminho real do seu ficheiro.
    # 4. Ajuste o parâmetro 'width' conforme necessário para o tamanho desejado.
    # st.image("caminho/para/o/seu/logotipo.png", width=250) # <-- COLOQUE O CAMINHO DA IMAGEM AQUI

    st.markdown('<h1 class="sub-header" style="text-align: center;">Sistema de Intervenção Estudantil</h1>', unsafe_allow_html=True)

    menu = option_menu(
        menu_title=None,
        options=["Início", "Exploração de Dados", "Previsão Individual", "Análise do Modelo Treinado", "Análise de Matriz", "Documentação"],
        icons=["house-door", "bar-chart-line", "clipboard-data", "robot", "grid-3x3", "book"],
        menu_icon="cast",
        default_index=0,
    )

    st.markdown("---")
    st.markdown("### Sobre a Aplicação")
    st.info("""
    Ferramenta interativa para explorar o dataset estudantil, fazer previsões
    individuais e analisar o modelo de Machine Learning treinado e suas propriedades.
    """)

    # --- Adicionar Nomes dos Autores/Alunos e Orientador ---
    st.markdown("---")
    st.markdown("### Projeto Académico")
    st.write("Desenvolvido por:")
    st.write("- Afonso Marcos")
    st.write("- Afonso Salgado")
    st.write("- Pedro Afonso")
    st.write("---") # Separador opcional
    st.write("Orientador:")
    st.write("- [Nome do Orientador]") # <--- SUBSTITUIR PELO NOME REAL DO ORIENTADOR

    st.markdown("---")
    st.markdown("### Detalhes Técnicos") # Rebatizei para evitar conflito com 'Projeto Académico'
    st.write("Framework: Streamlit")
    st.write("Linguagem: Python")
    st.write("Bibliotecas: scikit-learn, pandas, numpy, plotly, joblib")


# --- Conteúdo Principal ---

if menu == "Início":
    st.markdown('<h1 class="main-header">Bem-vindo ao Sistema de Intervenção Estudantil 🚀</h1>', unsafe_allow_html=True)

    st.markdown('<p class="info-text">Este aplicativo é uma ferramenta interativa baseada no seu modelo de Machine Learning para prever o desempenho estudantil, usando o dataset "UCI Student Performance".</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Amostras no Dataset", f"{student_df_original.shape[0]}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if 'original_cols' in locals() and original_cols is not None:
             st.metric("Características Originais", f"{len(original_cols)}")
        else:
             st.metric("Características Originais", "N/A")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Status do Pipeline", "Carregado ✅")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<h2 class="sub-header">Funcionalidades:</h2>', unsafe_allow_html=True)

    st.markdown("""
    *   **Exploração de Dados:** Visualize resumos, distribuições e correlações do dataset original, com foco na interpretabilidade.
    *   **Previsão Individual:** Insira dados de um aluno e obtenha uma previsão do seu desempenho final usando o modelo treinado.
    *   **Análise do Modelo Treinado:** Veja as métricas de avaliação e a matriz de confusão do modelo carregado no conjunto de teste.
    *   **Análise de Matriz:** Explore visualmente e analiticamente propriedades de matrizes relevantes (Confusão de *qualquer* modelo, Correlação/Covariância dos seus dados, Matriz Personalizada).
    *   **Documentação:** Encontre mais informações sobre a aplicação e o projeto.
    """)


# --- Exploração de Dados ---
elif menu == "Exploração de Dados":
    st.markdown('<h1 class="main-header">Exploração do Dataset Estudantil</h1>', unsafe_allow_html=True)

    df = student_df_original.copy() # Use uma cópia para não modificar o dataset cacheado

    st.markdown('<p class="info-text">Analise a estrutura, distribuição e relações entre as características do seu dataset de dados estudantis (`student-data.csv`).</p>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📋 Resumo Geral", "📈 Distribuições", "🔍 Relações"])

    with tab1:
        st.markdown('<h2 class="sub-header">Resumo Geral do Dataset</h2>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Dimensões do Dataset:**", df.shape)
            if 'original_cols' in locals() and original_cols is not None:
                 st.write(f"**Características (Features):** {len(original_cols)}")
            else:
                 st.warning("Nomes das características originais não carregados.")
                 st.write(f"**Características (Features):** {df.shape[1] - (1 if TARGET_ORIGINAL_NAME in df.columns else 0)}")

            st.write(f"**Amostras:** {df.shape[0]}")

            if TARGET_ORIGINAL_NAME in df.columns:
                 st.write(f"**Variável Alvo:** '{TARGET_ORIGINAL_NAME}'")
                 unique_target_values = df[TARGET_ORIGINAL_NAME].dropna().unique().tolist() # Tratar NaNs
                 st.write(f"**Classes:** {', '.join(map(str, unique_target_values))}")

            st.markdown('---')
            st.write("**Primeiras 5 Linhas:**")
            st.dataframe(df.head(), use_container_width=True)

        with col2:
             if TARGET_ORIGINAL_NAME in df.columns:
                 st.write(f"**Distribuição da Coluna '{TARGET_ORIGINAL_NAME}':**")
                 class_counts = df[TARGET_ORIGINAL_NAME].value_counts().reset_index()
                 class_counts.columns = ['Class', 'Count']
                 fig_pie = px.pie(
                     values=class_counts['Count'],
                     names=class_counts['Class'],
                     title=f"Distribuição de '{TARGET_ORIGINAL_NAME}'",
                     hole=0.3,
                     color=class_counts['Class'], # Cor por classe
                     color_discrete_map={CLASS_NAMES[0]: 'salmon', CLASS_NAMES[1]: 'lightgreen'} # Cores para 'no' e 'yes'
                 )
                 fig_pie.update_layout(legend_title_text=TARGET_ORIGINAL_NAME.replace('_', ' ').title())
                 st.plotly_chart(fig_pie, use_container_width=True)
             else:
                  st.info(f"Não é possível mostrar a distribuição da coluna alvo '{TARGET_ORIGINAL_NAME}'.")

        st.markdown('<h2 class="sub-header">Estatísticas Descritivas</h2>', unsafe_allow_html=True)
        st.dataframe(df.describe(include='all'), use_container_width=True)

    with tab2:
        st.markdown('<h2 class="sub-header">Distribuição das Características</h2>', unsafe_allow_html=True)
        st.markdown('<p class="info-text">Visualize a distribuição de cada característica ou compare-a com a situação final do aluno.</p>', unsafe_allow_html=True)

        feature_options_dist = original_cols if 'original_cols' in locals() and original_cols is not None else df.columns.tolist()
        # Remover a coluna alvo da lista de opções para distribuição da feature
        if TARGET_ORIGINAL_NAME in feature_options_dist:
             feature_options_dist.remove(TARGET_ORIGINAL_NAME)

        selected_feature_dist = st.selectbox(
            "Selecione uma característica para visualizar a distribuição:",
            options=feature_options_dist
        )

        if selected_feature_dist:
             dtype = df[selected_feature_dist].dtype

             # Adiciona um rádio botão para escolher o tipo de visualização
             view_option = st.radio(
                 f"Visualizar a distribuição de '{selected_feature_dist.replace('_', ' ').title()}'",
                 ["Distribuição Geral", f"Comparar com '{TARGET_ORIGINAL_NAME.replace('_', ' ').title()}'"],
                 horizontal=True,
                 key=f"view_option_{selected_feature_dist}" # Usa uma chave única para o widget
             )

             if view_option == "Distribuição Geral":
                 # --- Lógica para Distribuição Geral (Existente, Ligeiramente Melhorada) ---
                 if selected_feature_dist in ordinal_numeric_features_to_map:
                     st.write(f"Distribuição Geral de **{selected_feature_dist.replace('_', ' ').title()}** (Interpretado com Rótulos):")
                     mapping_dict = ORDINAL_MAPPINGS[selected_feature_dist]

                     # Ensure NaNs are handled before mapping if necessary
                     temp_series_mapped = df[selected_feature_dist].dropna().map(mapping_dict)

                     counts_df = temp_series_mapped.value_counts().reset_index()
                     counts_df.columns = ['Label', 'Count']

                     ordered_labels = [mapping_dict.get(k) for k in sorted(mapping_dict.keys()) if mapping_dict.get(k) in counts_df['Label'].tolist()]

                     fig_bar = px.bar(
                         counts_df,
                         x='Label',
                         y='Count',
                         title=f'Distribuição Geral de "{selected_feature_dist.replace("_", " ").title()}"',
                         text_auto=True # Mostrar valores nas barras
                     )
                     st.plotly_chart(fig_bar, use_container_width=True)
                     st.info(f"Este gráfico mostra quantos alunos caem em cada nível de '{selected_feature_dist.replace('_', ' ').title()}'.")

                 elif dtype in [np.number, 'int64', 'float64']: # Numéricas contínuas ou count
                      st.write(f"Distribuição Geral de **{selected_feature_dist.replace('_', ' ').title()}**:")
                      fig_hist = px.histogram(
                          df.dropna(subset=[selected_feature_dist]), # Remover NaNs para o histograma
                          x=selected_feature_dist,
                          marginal="box",
                          title=f'Distribuição Geral de "{selected_feature_dist.replace("_", " ").title()}"'
                      )
                      st.plotly_chart(fig_hist, use_container_width=True)
                      st.info(f"Este gráfico mostra a frequência dos valores de '{selected_feature_dist.replace('_', ' ').title()}' (histograma) e um resumo da distribuição (box plot).")


                 elif dtype == 'object' or pd.api.types.is_categorical_dtype(df[selected_feature_dist]): # Categóricas string/object
                      st.write(f"Distribuição Geral de **{selected_feature_dist.replace('_', ' ').title()}**:")
                      counts_df = df[selected_feature_dist].value_counts().reset_index()
                      counts_df.columns = [selected_feature_dist, 'Count']
                      fig_bar = px.bar(
                          counts_df,
                          x=selected_feature_dist,
                          y='Count',
                          title=f'Distribuição Geral de "{selected_feature_dist.replace("_", " ").title()}"',
                          text_auto=True
                      )
                      st.plotly_chart(fig_bar, use_container_width=True)
                      st.info(f"Este gráfico mostra quantos alunos pertencem a cada categoria de '{selected_feature_dist.replace('_', ' ').title()}'.")

                 else:
                     st.info(f"A característica '{selected_feature_dist}' tem um tipo de dado ({dtype}) que não é suportado para visualização de distribuição geral neste momento.")

             elif view_option == f"Comparar com '{TARGET_ORIGINAL_NAME.replace('_', ' ').title()}'":
                 # --- Lógica para Comparação com a Variável Alvo ---
                 if TARGET_ORIGINAL_NAME not in df.columns:
                      st.warning(f"Coluna alvo '{TARGET_ORIGINAL_NAME}' não encontrada no seu dataset para comparação.")
                 else:
                    st.write(f"Comparação da Distribuição de **{selected_feature_dist.replace('_', ' ').title()}** por **{TARGET_ORIGINAL_NAME.replace('_', ' ').title()}**:")

                    # Certifica-se de que a coluna alvo tem apenas os valores esperados
                    if not set(df[TARGET_ORIGINAL_NAME].dropna().unique()).issubset(set(CLASS_NAMES)):
                         st.warning(f"A coluna alvo '{TARGET_ORIGINAL_NAME}' contém valores inesperados ({df[TARGET_ORIGINAL_NAME].dropna().unique().tolist()}). A comparação pode não ser exibida corretamente.")
                         # Tenta continuar, mas pode falhar nos gráficos


                    if selected_feature_dist in ordinal_numeric_features_to_map:
                         # Comparação para Ordinais Numéricas Mapeadas
                         mapping_dict = ORDINAL_MAPPINGS[selected_feature_dist]

                         # Criar coluna mapeada TEMPORÁRIA para o gráfico de comparação
                         # Remover NaNs tanto da feature quanto da coluna alvo
                         temp_df_mapped = df.dropna(subset=[selected_feature_dist, TARGET_ORIGINAL_NAME]).copy()
                         temp_df_mapped['Feature_Label'] = temp_df_mapped[selected_feature_dist].map(mapping_dict)

                         # Contar por Rótulo da Feature e Classe Alvo
                         comparison_counts = temp_df_mapped.groupby(['Feature_Label', TARGET_ORIGINAL_NAME]).size().reset_index(name='Count')

                         # Garantir que TODOS os rótulos possíveis da feature mapeada estejam presentes, mesmo com Count=0, para a ordem correta
                         # Cria um multi-índice com todos os rótulos da feature e todas as classes alvo
                         all_labels = [mapping_dict.get(k) for k in sorted(mapping_dict.keys())]
                         multi_index = pd.MultiIndex.from_product([all_labels, CLASS_NAMES], names=['Feature_Label', TARGET_ORIGINAL_NAME])

                         # Re-indexa o DataFrame de contagem e preenche NaNs com 0
                         comparison_counts = comparison_counts.set_index(['Feature_Label', TARGET_ORIGINAL_NAME]).reindex(multi_index, fill_value=0).reset_index()

                         # Garantir a ordem correta no eixo X (usa os rótulos mapeados em ordem, agora todos presentes)
                         ordered_labels = [mapping_dict.get(k) for k in sorted(mapping_dict.keys())]


                         fig_comp_bar = px.bar(
                             comparison_counts,
                             x='Feature_Label',
                             y='Count',
                             color=TARGET_ORIGINAL_NAME, # Cor por Passou/Não Passou
                             title=f'Distribuição de "{selected_feature_dist.replace("_", " ").title()}" por "{TARGET_ORIGINAL_NAME.replace("_", " ").title()}"',
                             # text_auto=True, # Auto text pode ficar confuso com muitas barras
                             barmode='group', # barras lado a lado
                             color_discrete_map={CLASS_NAMES[0]: 'salmon', CLASS_NAMES[1]: 'lightgreen'}
                         )
                         fig_comp_bar.update_layout(legend_title_text=TARGET_ORIGINAL_NAME.replace('_', ' ').title())
                         st.plotly_chart(fig_comp_bar, use_container_width=True)
                         st.info(f"""
                         Este gráfico compara o número de alunos que passaram ('yes') e não passaram ('no') em cada nível de '{selected_feature_dist.replace('_', ' ').title()}'.
                         Procure por níveis onde há uma proporção significativamente maior de barras verdes ('yes'). Isso sugere que este nível da característica está associado a uma maior probabilidade de aprovação.
                         """)

                    elif dtype in [np.number, 'int64', 'float64']:
                         # Comparação para Numéricas (Box Plot)
                         fig_comp_box = px.box(
                             df.dropna(subset=[selected_feature_dist, TARGET_ORIGINAL_NAME]), # Remover NaNs
                             x=TARGET_ORIGINAL_NAME,
                             y=selected_feature_dist,
                             title=f'Distribuição de "{selected_feature_dist.replace("_", " ").title()}" por "{TARGET_ORIGINAL_NAME.replace("_", " ").title()}"',
                             color=TARGET_ORIGINAL_NAME, # Cor por Passou/Não Passou
                             color_discrete_map={CLASS_NAMES[0]: 'salmon', CLASS_NAMES[1]: 'lightgreen'},
                             points="all" # Mostrar todos os pontos
                         )
                         st.plotly_chart(fig_comp_box, use_container_width=True)
                         st.info(f"""
                         Este gráfico mostra a distribuição dos valores de '{selected_feature_dist.replace('_', ' ').title()}' para alunos que passaram ('yes') e não passaram ('no').
                         *   A linha no meio da caixa é a mediana (valor típico).
                         *   A caixa representa 50% dos dados (entre o 1º e o 3º quartil).
                         *   Os "bigodes" estendem-se aos valores mínimo e máximo (excluindo outliers).
                         Observe se a mediana ou o intervalo de valores são significativamente diferentes entre os grupos 'yes' e 'no'. Isso sugere que a característica é importante para diferenciar entre alunos que passam e não passam.
                         """)


                    elif dtype == 'object' or pd.api.types.is_categorical_dtype(df[selected_feature_dist]):
                         # Comparação para Categóricas String/Object
                         # Contar por Categoria da Feature e Classe Alvo
                         comparison_counts = df.dropna(subset=[selected_feature_dist, TARGET_ORIGINAL_NAME]).groupby([selected_feature_dist, TARGET_ORIGINAL_NAME]).size().reset_index(name='Count')

                         # Opcional: garantir que TODAS as categorias da feature original estejam presentes, mesmo com Count=0
                         # Se a feature tiver muitas categorias, pode poluir o gráfico.
                         # Por enquanto, mantemos apenas as que aparecem nos dados após dropna.

                         fig_comp_bar = px.bar(
                             comparison_counts,
                             x=selected_feature_dist,
                             y='Count',
                             color=TARGET_ORIGINAL_NAME, # Cor por Passou/Não Passou
                             title=f'Distribuição de "{selected_feature_dist.replace("_", " ").title()}" por "{TARGET_ORIGINAL_NAME.replace("_", " ").title()}"',
                             text_auto=True,
                             barmode='group', # barras lado a lado
                             color_discrete_map={CLASS_NAMES[0]: 'salmon', CLASS_NAMES[1]: 'lightgreen'}
                         )
                         fig_comp_bar.update_layout(legend_title_text=TARGET_ORIGINAL_NAME.replace('_', ' ').title())
                         st.plotly_chart(fig_comp_bar, use_container_width=True)
                         st.info(f"""
                         Este gráfico compara o número de alunos que passaram ('yes') e não passaram ('no') em cada categoria de '{selected_feature_dist.replace('_', ' ').title()}'.
                         Procure por categorias onde há uma proporção significativamente maior de barras verdes ('yes'). Isso sugere que essa categoria está associada a uma maior probabilidade de aprovação.
                         """)

                    else:
                        st.info(f"A característica '{selected_feature_dist}' tem um tipo de dado ({dtype}) que não é suportado para comparação com a situação final neste momento.")

    
    with tab3: # Esta é a tab para "Relações", dentro de "Exploração de Dados"
        st.markdown('<h2 class="sub-header">Relações entre Características</h2>', unsafe_allow_html=True)
        st.markdown('<p class="info-text">Analise a relação entre pares de características no seu dataset, coloridas pela classe alvo.</p>', unsafe_allow_html=True)

        # Certifica-se de que usa o DataFrame 'df' da EDA, que é uma cópia do original
        df_source = df # df é a cópia do dataset original criada no início desta secção EDA

        # Verifica se a coluna alvo existe
        if TARGET_ORIGINAL_NAME not in df_source.columns:
            st.warning(f"Coluna alvo '{TARGET_ORIGINAL_NAME}' não encontrada no seu dataset. As visualizações de relação coloridas pela situação final não estarão disponíveis.")
            # Adiciona uma análise básica de correlação para features numéricas se a alvo estiver faltando
            st.markdown('### Matriz de Correlação (Sem cor por Classe)', unsafe_allow_html=True)
            df_features_only = df_source[original_cols] if 'original_cols' in locals() and original_cols is not None else df_source
            df_numeric_for_corr = df_features_only.select_dtypes(include=np.number)
            if df_numeric_for_corr.empty:
                st.info("Não há colunas numéricas para calcular a matriz de correlação.")
            else:
                 fig_corr, corr_matrix = plot_correlation_matrix_px(df_numeric_for_corr) # Assume plot_correlation_matrix_px is globally defined
                 if fig_corr is not None:
                      st.plotly_chart(fig_corr, use_container_width=True)
            pass # Sai do bloco tab3 se a coluna alvo não existir


        # --- Se a coluna alvo existir, continua com as visualizações coloridas ---

        st.markdown('### Visualização de Relações por Situação Final', unsafe_allow_html=True)
        st.markdown(f'<p class="info-text">Selecione duas características para visualizar sua relação e como a "{TARGET_ORIGINAL_NAME.replace("_", " ").title()}" se distribui.</p>', unsafe_allow_html=True)


        # Obter a lista de todas as features para seleção (exceto a alvo)
        all_features_options = original_cols if 'original_cols' in locals() and original_cols is not None else df_source.columns.tolist()
        if TARGET_ORIGINAL_NAME in all_features_options:
            all_features_options.remove(TARGET_ORIGINAL_NAME)


        # Controles para selecionar as duas features
        col_select1, col_select2 = st.columns(2)
        with col_select1:
            feature1 = st.selectbox("Selecione a Característica 1", all_features_options, index=0, key="rel_feature1")
        with col_select2:
            # Excluir a Característica 1 da lista de opções para a Característica 2
            options_feature2 = [col for col in all_features_options if col != feature1]
            # Encontrar um índice padrão seguro para feature2
            default_index_feature2 = 0
            # Tenta encontrar um default razoável que não seja igual a feature1
            if feature1 == all_features_options[0] and len(options_feature2) > 0:
                 if len(all_features_options) > 1:
                      # Tenta selecionar a segunda feature original como default para a segunda caixa
                      second_original_feature = all_features_options[1]
                      if second_original_feature in options_feature2:
                           default_index_feature2 = options_feature2.index(second_original_feature)

            feature2 = st.selectbox("Selecione a Característica 2", options_feature2, index=default_index_feature2, key="rel_feature2")


        if feature1 and feature2:
            # Determine os tipos das features selecionadas (numérica, categórica/ordinal)
            dtype1 = df_source[feature1].dtype
            dtype2 = df_source[feature2].dtype

            is_numeric1 = pd.api.types.is_numeric_dtype(dtype1) and feature1 not in ordinal_numeric_features_to_map # Trata ordinais numéricas como categóricas para plotagem específica
            is_numeric2 = pd.api.types.is_numeric_dtype(dtype2) and feature2 not in ordinal_numeric_features_to_map

            is_ordinal_or_categorical1 = (feature1 in ordinal_numeric_features_to_map) or (dtype1 == 'object') or pd.api.types.is_categorical_dtype(dtype1)
            is_ordinal_or_categorical2 = (feature2 in ordinal_numeric_features_to_map) or (dtype2 == 'object') or pd.api.types.is_categorical_dtype(dtype2)


            # --- Plotagem baseada nos tipos de features ---

            # Caso 1: Ambas são numéricas (contínuas)
            if is_numeric1 and is_numeric2:
                st.write(f"#### Dispersão: {feature1.replace('_', ' ').title()} vs {feature2.replace('_', ' ').title()} por {TARGET_ORIGINAL_NAME.replace('_', ' ').title()}")
                st.info("Este gráfico mostra a relação entre duas características numéricas contínuas, colorida pela situação final. Pontos próximos podem indicar grupos de alunos com características semelhantes.")
                fig = px.scatter(
                    df_source,
                    x=feature1,
                    y=feature2,
                    color=TARGET_ORIGINAL_NAME,
                    labels={"color": TARGET_ORIGINAL_NAME.replace('_', ' ').title()},
                    title=f"Dispersão: {feature1.replace('_', ' ').title()} vs {feature2.replace('_', ' ').title()}",
                    opacity=0.7,
                    hover_data={TARGET_ORIGINAL_NAME:False, feature1:True, feature2:True} # Exibe nome e valor das features no hover
                )
                fig.update_layout(legend_title_text=TARGET_ORIGINAL_NAME.replace('_', ' ').title())
                st.plotly_chart(fig, use_container_width=True)

            # Caso 2: Uma numérica (contínua) e outra categórica/ordinal
            elif is_numeric1 and is_ordinal_or_categorical2:
                st.write(f"#### Distribuição de {feature1.replace('_', ' ').title()} por {feature2.replace('_', ' ').title()} e {TARGET_ORIGINAL_NAME.replace('_', ' ').title()}")
                st.info(f"Este gráfico mostra a distribuição de **{feature1.replace('_', ' ').title()}** (numérica) para cada nível de **{feature2.replace('_', ' ').title()}** (categórica/ordinal), separada por situação final.")

                # Mapear feature2 se for ordinal numérica para exibir rótulos no gráfico
                df_plot = df_source.copy()
                if feature2 in ordinal_numeric_features_to_map:
                     df_plot[feature2] = df_plot[feature2].map(ORDINAL_MAPPINGS[feature2]).fillna('NaN') # Mapeia e trata NaNs
                     x_label_plot = feature2 # Nome da feature original
                     category_orders = {x_label_plot: [ORDINAL_MAPPINGS[feature2].get(k) for k in sorted(ORDINAL_MAPPINGS[feature2].keys())]} # Ordem correta para ordinais
                else:
                     x_label_plot = feature2
                     category_orders = None # Sem ordem específica para categóricas nominais

                fig = px.box( # Ou px.violin
                    df_plot.dropna(subset=[feature1, feature2, TARGET_ORIGINAL_NAME]),
                    x=x_label_plot,
                    y=feature1,
                    color=TARGET_ORIGINAL_NAME,
                    labels={"color": TARGET_ORIGINAL_NAME.replace('_', ' ').title(), x_label_plot: feature2.replace('_', ' ').title()}, # Rótulos
                    title=f"Distribuição de {feature1.replace('_', ' ').title()} por {feature2.replace('_', ' ').title()} e {TARGET_ORIGINAL_NAME.replace('_', ' ').title()}",
                    category_orders=category_orders,
                    # points="all" # Opcional: mostrar pontos individuais (pode sobrecarregar)
                )
                fig.update_layout(legend_title_text=TARGET_ORIGINAL_NAME.replace('_', ' ').title())
                st.plotly_chart(fig, use_container_width=True)

            elif is_ordinal_or_categorical1 and is_numeric2: # Simétrico ao caso anterior
                st.write(f"#### Distribuição de {feature2.replace('_', ' ').title()} por {feature1.replace('_', ' ').title()} e {TARGET_ORIGINAL_NAME.replace('_', ' ').title()}")
                st.info(f"Este gráfico mostra a distribuição de **{feature2.replace('_', ' ').title()}** (numérica) para cada nível de **{feature1.replace('_', ' ').title()}** (categórica/ordinal), separada por situação final.")

                # Mapear feature1 se for ordinal numérica
                df_plot = df_source.copy()
                if feature1 in ordinal_numeric_features_to_map:
                     df_plot[feature1] = df_plot[feature1].map(ORDINAL_MAPPINGS[feature1]).fillna('NaN')
                     x_label_plot = feature1
                     category_orders = {x_label_plot: [ORDINAL_MAPPINGS[feature1].get(k) for k in sorted(ORDINAL_MAPPINGS[feature1].keys())]}
                else:
                     x_label_plot = feature1
                     category_orders = None

                fig = px.box( # Ou px.violin
                    df_plot.dropna(subset=[feature1, feature2, TARGET_ORIGINAL_NAME]),
                    x=x_label_plot,
                    y=feature2,
                    color=TARGET_ORIGINAL_NAME,
                    labels={"color": TARGET_ORIGINAL_NAME.replace('_', ' ').title(), x_label_plot: feature1.replace('_', ' ').title()},
                    title=f"Distribuição de {feature2.replace('_', ' ').title()} por {feature1.replace('_', ' ').title()} e {TARGET_ORIGINAL_NAME.replace('_', ' ').title()}",
                    category_orders=category_orders,
                    # points="all"
                )
                fig.update_layout(legend_title_text=TARGET_ORIGINAL_NAME.replace('_', ' ').title())
                st.plotly_chart(fig, use_container_width=True)

            # Caso 3: Ambas são categóricas/ordinais
            elif is_ordinal_or_categorical1 and is_ordinal_or_categorical2:
                st.write(f"#### Contagem de Alunos por {feature1.replace('_', ' ').title()} vs {feature2.replace('_', ' ').title()} por {TARGET_ORIGINAL_NAME.replace('_', ' ').title()}")
                st.info(f"Este gráfico de barras agrupadas mostra o número de alunos em cada combinação de níveis de **{feature1.replace('_', ' ').title()}** e **{feature2.replace('_', ' ').title()}**, separada por situação final.")

                # Mapear ambas as features se forem ordinais numéricas
                df_plot = df_source.copy()
                x_label_plot = feature1
                color_label_plot = feature2
                category_orders_dict = {} # Para controlar a ordem dos eixos

                if feature1 in ordinal_numeric_features_to_map:
                     df_plot[feature1] = df_plot[feature1].map(ORDINAL_MAPPINGS[feature1]).fillna('NaN')
                     category_orders_dict[feature1] = [ORDINAL_MAPPINGS[feature1].get(k) for k in sorted(ORDINAL_MAPPINGS[feature1].keys())] # Ordem correta para x
                if feature2 in ordinal_numeric_features_to_map:
                     df_plot[feature2] = df_plot[feature2].map(ORDINAL_MAPPINGS[feature2]).fillna('NaN')
                     category_orders_dict[feature2] = [ORDINAL_MAPPINGS[feature2].get(k) for k in sorted(ORDINAL_MAPPINGS[feature2].keys())] # Ordem correta para cor

                # Contar ocorrências por combinação das 3 colunas
                counts_df = df_plot.dropna(subset=[feature1, feature2, TARGET_ORIGINAL_NAME]).groupby([feature1, feature2, TARGET_ORIGINAL_NAME]).size().reset_index(name='Count')

                # Garantir que todas as combinações possíveis estejam presentes para a ordem (pode ser complexo e não essencial para plotly express)
                # Plotly express geralmente ordena automaticamente categóricas se não houver category_orders

                fig = px.bar(
                    counts_df,
                    x=feature1,
                    y='Count',
                    color=feature2, # Colorir pela Característica 2
                    facet_col=TARGET_ORIGINAL_NAME, # Separar por Situação Final
                    facet_col_wrap=2, # Exibir 2 colunas de facetas
                    title=f"Contagem de Alunos por {feature1.replace('_', ' ').title()} vs {feature2.replace('_', ' ').title()} por {TARGET_ORIGINAL_NAME.replace('_', ' ').title()}",
                    category_orders=category_orders_dict, # Aplica as ordens se definidas
                    labels={TARGET_ORIGINAL_NAME: TARGET_ORIGINAL_NAME.replace('_', ' ').title()}, # Label da faceta
                    barmode='group' # barras agrupadas (uma barra para cada nível de feature2 dentro de cada nível de feature1)
                    # barmode='stack' # Opcional: barras empilhadas
                )

                # Ajustar títulos das facetas (opcional)
                fig.for_each_annotation(lambda a: a.update(text=f"{TARGET_ORIGINAL_NAME.replace('_', ' ').title()}={a.text.split('=')[-1]}"))

                fig.update_layout(legend_title_text=feature2.replace('_', ' ').title()) # Título da legenda é a Característica 2
                st.plotly_chart(fig, use_container_width=True)


            else:
                # Caso inesperado (e.g., colunas não reconhecidas)
                st.info(f"Não foi possível gerar uma visualização apropriada para os tipos de dados selecionados ({dtype1} para {feature1}, {dtype2} para {feature2}).")

        else: # Caso feature1 ou feature2 sejam None (não deveriam ser com os selectboxes populados)
             st.warning("Por favor, selecione duas características para visualizar a relação.")


        st.markdown("---") # Adiciona um separador visual antes da matriz de correlação

        # --- Matriz de Correlação (Mantida aqui, é relevante para relações entre features) ---
        st.markdown('### Matriz de Correlação', unsafe_allow_html=True)
        st.markdown('<p class="info-text">Veja a correlação linear entre as características numéricas do seu dataset.</p>', unsafe_allow_html=True)

        # Certifica-se de que usa o DataFrame 'df' da EDA (df_source)
        # Para a matriz principal, ainda é útil ver correlações APENAS entre features
        df_features_only = df_source[original_cols] if 'original_cols' in locals() and original_cols is not None else df_source.drop(columns=[TARGET_ORIGINAL_NAME] if TARGET_ORIGINAL_NAME in df_source.columns else [])
        df_numeric_for_corr_matrix = df_features_only.select_dtypes(include=np.number)

        if df_numeric_for_corr_matrix.empty:
             st.info("Não há colunas numéricas entre as características usadas para calcular a matriz de correlação no seu dataset.")
        else:
            # A função plot_correlation_matrix_px precisa estar definida em outro lugar
            fig_corr, corr_matrix = plot_correlation_matrix_px(df_numeric_for_corr_matrix) # Usa o df sem a target para a matriz principal
            if fig_corr is not None and corr_matrix is not None:
                st.plotly_chart(fig_corr, use_container_width=True)

                # REMOVER: Seção Pares com Alta Correlação entre features
                # st.markdown('#### Pares com Alta Correlação', unsafe_allow_html=True)
                # st.markdown('<p class="info-text">Identifica pares de características com forte correlação linear (|r| > 0.7).</p>', unsafe_allow_html=True)
                # corr_unstacked = corr_matrix.stack().reset_index()
                # corr_unstacked.columns = ['Feature1', 'Feature2', 'Correlation']
                # high_corr_pairs = corr_unstacked[
                #     (abs(corr_unstacked['Correlation']) > 0.7) &
                #     (corr_unstacked['Feature1'] != corr_corr_pairs['Feature2'])
                # ].sort_values(by='Correlation', ascending=False)

                # # Remover duplicados (Feature1, Feature2) e (Feature2, Feature1)
                # high_corr_pairs['Pair'] = high_corr_pairs.apply(lambda row: tuple(sorted((row['Feature1'], row['Feature2']))), axis=1)
                # high_corr_pairs = high_corr_pairs.drop_duplicates(subset=['Pair']).drop(columns=['Pair'])

                # if not high_corr_pairs.empty:
                #     st.dataframe(high_corr_pairs.round(4), use_container_width=True)
                #     st.warning("⚠️ Alta correlação entre algumas características pode indicar redundância.")
                # else:
                #     st.info("Não foram encontrados pares de características com correlação linear forte (|r| > 0.7) entre as características numéricas usadas.")

            else:
                 st.info("Não há dados numéricos suficientes entre as características originais no seu dataset para calcular a matriz de correlação.")


        # --- Nova Seção: Correlação com a Variável Alvo "passed" ---
        st.markdown(f'### Correlação com a Variável Alvo: "{TARGET_ORIGINAL_NAME}"', unsafe_allow_html=True)
        st.markdown(f'<p class="info-text">Veja a correlação linear das características numéricas com a variável alvo "{TARGET_ORIGINAL_NAME}".</p>', unsafe_allow_html=True)

        # Prepara o DataFrame para calcular a correlação com a target
        df_for_target_corr = None # Inicializa como None
        target_col_processed = None # Vai guardar a série da coluna target (original ou convertida)

        # Verifica se a coluna alvo existe
        if TARGET_ORIGINAL_NAME in df_source.columns:
            target_col_original = df_source[TARGET_ORIGINAL_NAME]

            # Verifica se a coluna alvo já é numérica ou booleana (Pandas trata booleanos como 0/1 em ops numéricas)
            if pd.api.types.is_numeric_dtype(target_col_original) or pd.api.types.is_bool_dtype(target_col_original):
                st.info(f"A coluna alvo '{TARGET_ORIGINAL_NAME}' é numérica/booleana. Calculando correlação diretamente.")
                df_for_target_corr = df_source.select_dtypes(include=np.number).copy()
                # Garante que a coluna target está incluída se for numérica/booleana mas não pega por select_dtypes (ex: bool)
                if TARGET_ORIGINAL_NAME not in df_for_target_corr.columns:
                     df_for_target_corr[TARGET_ORIGINAL_NAME] = target_col_original.astype(float) # Converte para float para consistência
                target_col_processed = df_for_target_corr[TARGET_ORIGINAL_NAME]


            # Se não for numérica/booleana, tenta converter se for binária
            elif not pd.api.types.is_numeric_dtype(target_col_original) and not pd.api.types.is_bool_dtype(target_col_original):
                 unique_target_values = target_col_original.dropna().unique()
                 if len(unique_target_values) == 2:
                     st.warning(f"A coluna alvo '{TARGET_ORIGINAL_NAME}' não é numérica, mas parece ser binária ({unique_target_values[0]} vs {unique_target_values[1]}). Convertendo para 0/1 para calcular a correlação linear (Ponto-Bisserial).")
                     df_for_target_corr = df_source.select_dtypes(include=np.number).copy()
                     # Converte os valores binários para 0 e 1. cat.codes mapeia categorias para inteiros.
                     target_col_processed = target_col_original.astype('category').cat.codes.astype(float) # Converte para float para consistência
                     df_for_target_corr[TARGET_ORIGINAL_NAME] = target_col_processed

                 else:
                     # Não é numérica e não é binária
                     st.warning(f"A coluna alvo '{TARGET_ORIGINAL_NAME}' não é numérica e não parece ser binária (mais de 2 valores únicos não-NaN: {len(unique_target_values)}). Não é possível calcular correlação linear de Pearson com ela.")
                     df_for_target_corr = None # Define como None para pular o cálculo

            # Se chegou aqui e df_for_target_corr ainda é None, significa que a coluna alvo não é adequada
            # Nada a fazer, a mensagem de warning já foi exibida.

        else:
             # A coluna alvo não foi encontrada no DataFrame
             st.warning(f"A coluna alvo '{TARGET_ORIGINAL_NAME}' não foi encontrada no DataFrame para calcular a correlação.")
             df_for_target_corr = None # Define como None para pular o cálculo


        # --- Cálculo e Exibição da Correlação com a Target (se df_for_target_corr foi preparado) ---
        if df_for_target_corr is not None and TARGET_ORIGINAL_NAME in df_for_target_corr.columns:
             # Verifica se há outras colunas numéricas além da target para correlacionar
             other_cols_for_corr = [col for col in df_for_target_corr.columns if col != TARGET_ORIGINAL_NAME]

             if not other_cols_for_corr:
                  st.info(f"Não há outras colunas numéricas no dataset para calcular a correlação com '{TARGET_ORIGINAL_NAME}'.")
             else:
                # Calcula a matriz de correlação usando o DataFrame preparado (com a target, potencialmente convertida)
                corr_matrix_with_target = df_for_target_corr.corr()

                # Extrai as correlações com a coluna alvo
                if TARGET_ORIGINAL_NAME in corr_matrix_with_target.columns:
                     # Remove a correlação da target com ela mesma (que é 1)
                     target_correlations = corr_matrix_with_target[TARGET_ORIGINAL_NAME].drop(TARGET_ORIGINAL_NAME, errors='ignore') # errors='ignore' para segurança

                     # Classifica pela magnitude absoluta da correlação (para ver as mais "relacionadas", positivas ou negativas)
                     if not target_correlations.empty:
                         sorted_target_corr_abs = target_correlations.abs().sort_values(ascending=False)

                         # Prepara o DataFrame para exibição, mantendo o valor original da correlação
                         # Garante que a ordem é a da classificação por valor absoluto
                         ordered_target_correlations = target_correlations.loc[sorted_target_corr_abs.index]

                         corr_df = ordered_target_correlations.reset_index()
                         corr_df.columns = ['Feature', f'Correlation_with_{TARGET_ORIGINAL_NAME}']

                         st.dataframe(corr_df.round(4), use_container_width=True)
                         # Mensagem informativa ajustada
                         status_message = f"As características acima são listadas pela força da correlação linear com '{TARGET_ORIGINAL_NAME}' (do mais forte para o mais fraco)."
                         if pd.api.types.is_numeric_dtype(target_col_original) or pd.api.types.is_bool_dtype(target_col_original):
                              pass # Já informado acima que é numérica/booleana
                         else: # Implica que foi convertida de binária
                              status_message += f" (A coluna '{TARGET_ORIGINAL_NAME}' foi convertida para 0/1 para este cálculo)."
                         st.info(status_message)

                     else:
                         st.info(f"Não há outras colunas numéricas para calcular a correlação com '{TARGET_ORIGINAL_NAME}'.")

                else:
                    st.warning(f"Erro interno: A coluna alvo '{TARGET_ORIGINAL_NAME}' não foi encontrada na matriz de correlação calculada após o processamento.")

        # Se df_for_target_corr foi None (target não encontrada ou não binária), as mensagens de warning já foram exibidas.

elif menu == "Previsão Individual":
    st.markdown('<h2 class="sub-header">Dados do Aluno</h2>', unsafe_allow_html=True)
    st.markdown('<p class="info-text">Insira os dados de um aluno para prever se ele passará ou não.</p>', unsafe_allow_html=True)

    if 'original_cols' not in locals() or original_cols is None:
        st.error("Não foi possível carregar os nomes das características originais. A secção de Previsão Individual não está disponível.")
    else:
        input_data = {}

        original_dtypes = student_df_original[original_cols].dtypes

        numeric_features = [col for col in original_cols if original_dtypes[col] in [np.number, 'int64', 'float64']]
        categorical_features = [col for col in original_cols if original_dtypes[col] == 'object'] # Assumindo que categóricas são object/string

        # --- Descrição Curta das Características (Para a Previsão Individual) ---
        # É recomendado que este dicionário esteja definido globalmente para evitar repetição.
        # Se o seu dicionário global é mais completo, use esse em vez de redefini-lo aqui.
        # Exemplo de como deve ser (verificar o seu topo se já existe):
        feature_descriptions_short = {
            "school": "Escola", "sex": "Gênero", "age": "Idade", "address": "Residência",
            "famsize": "Tamanho família", "Pstatus": "Status pais", "Medu": "Escolaridade mãe",
            "Fedu": "Escolaridade pai", "Mjob": "Ocupação mãe", "Fjob": "Ocupação pai",
            "reason": "Motivo escola", "guardian": "Guardião", "traveltime": "Tempo viagem",
            "studytime": "Tempo estudo", "failures": "Reprovações", "schoolsup": "Apoio escola",
            "famsup": "Apoio família", "paid": "Aulas pagas", "activities": "Atividades extra",
            "nursery": "Frequentou creche", "higher": "Deseja superior", "internet": "Acesso internet",
            "romantic": "Relacionamento", "famrel": "Qualidade relações familiares", "freetime": "Tempo livre",
            "goout": "Sair c/amigos", "Dalc": "Álcool dias semana", "Walc": "Álcool fins semana",
            "health": "Estado saúde", "absences": "Faltas", "passed": "Aprovado"
        }
        # Fim do Dicionário Curto (Verifique se já existe no topo do script)


        # --- CSS para o Grid Layout e Descrição Pequena ---
        # Este CSS DEVE estar definido no bloco <style> principal no topo do script.
        # Replicado aqui APENAS PARA REFERÊNCIA visual do que esta secção precisa de CSS.
        # NÃO MANTENHA ESTE BLOCO <style> DUPLICADO se já tiver o CSS globalmente.
        st.markdown("""
        <style>
            /* CSS para o Grid Layout na Previsão Individual */
            .input-grid-container {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-bottom: 20px;
            }
            .grid-item { /* Alterado de grid_item para grid-item (padrão CSS) */
                display: flex;
                flex-direction: column;
            }
            .grid-item div[data-testid="stAlert"] { /* Alterado */
                margin-top: 5px; margin-bottom: 5px; padding: 10px;
                font-size: 0.9rem; line-height: 1.4;
            }
             .grid-item div[data-testid="stNumberInput"], /* Alterado */
             .grid-item div[data-testid="stSelectbox"] { /* Alterado */
                 margin-top: 5px;
             }
             .feature-label-container {
                 margin-bottom: 5px;
                 display: flex;
                 align-items: baseline;
                 flex-wrap: wrap;
             }
             .feature-label-container strong {
                  margin-right: 5px;
             }
             .small-description {
                 font-size: 0.8em;
                 font-weight: normal;
                 color: #555;
             }
             .grid-item strong { /* Alterado */
                 display: inline-block;
             }
        </style>
        """, unsafe_allow_html=True)
        # --- Fim CSS (Verifique se já está no topo do script) ---

        # --- Opção para escolher o modelo ---
        st.markdown("### Seleção do Modelo para Previsão")

        artefacts_path = 'artefacts/'
        selected_model_instance = None # Inicializa como None
        selected_model_filename = None # Inicializa como None

        try:
            # Lista todos os ficheiros .joblib na pasta 'artefacts'
            all_joblib_files = [f for f in os.listdir(artefacts_path) if f.endswith('.joblib')]

            # Filtra para obter apenas os ficheiros que são modelos (exclui preprocessor, colunas)
            model_files = [f for f in all_joblib_files if f not in ['preprocessor.joblib', 'original_input_columns.joblib', 'processed_feature_names.joblib']]
            model_files.sort() # Ordena alfabeticamente para uma lista consistente

            # --- Dicionário de mapeamento de NOME DE FICHEIRO para NOME DE EXIBIÇÃO ---
            # Você DEVE atualizar este dicionário se salvar novos tipos de modelos com nomes diferentes.
            # A chave é o nome do ficheiro .joblib (sem o caminho, apenas o nome).
            # O valor é o nome amigável que você quer que apareça no selectbox.
            filename_to_display_name_map = {

                'DecisionTreeClassifier.joblib': 'Árvore de Decisão',
                'random_forest_model.joblib': 'Random Forest',
                'svm_model.joblib': 'SVM (SVC Padrão)',
                'svm_(otimizado)_model.joblib': 'SVM (SVC Otimizado)',
                  # Assumindo que seu SVC foi salvo como SVC.joblib
                # 'SVC_linear.joblib': 'SVM (Kernel Linear)', # Exemplo se salvou um SVM linear com nome diferente
                'random_forest_(otimizado)_model.joblib': 'Gradient Boosting',
                'best_model.joblib': 'AdaBoost(modelo recomendado)',
                # ADICIONE AQUI OUTROS NOMES DE FICHEIRO SEUS -> NOME AMIGÁVEL
                # Ex: 'MeuNovoModeloPersonalizado.joblib': 'Modelo Personalizado'
            }
            # --- Fim do dicionário de mapeamento ---


            # Cria a lista de nomes a serem exibidos no selectbox e o mapeamento DE EXIBIÇÃO para FICHEIRO
            model_display_options = []
            model_filename_map = {} # Mapeia o nome exibido de volta para o nome do ficheiro .joblib
            default_model_display_name = 'AdaBoost(modelo recomendado)' # Nome de exibição do modelo padrão

            # --- ESTE É O BLOCO CORRIGIDO PARA CRIAR A LISTA DE SELEÇÃO ---
            for f in model_files:
                # Obtém o nome amigável do mapeamento, ou usa o nome do ficheiro (sem .joblib) como fallback
                display_name_base = filename_to_display_name_map.get(f, f[:-8]) # Remove .joblib do nome do ficheiro
                default_model_display_name = 'AdaBoost(modelo recomendado)'

                display_name = display_name_base # Começa com o nome base
                # Adiciona o nome de exibição à lista
                model_display_options.append(display_name)
                # Mapeia o nome de exibição de volta para o nome do ficheiro real .joblib
                model_filename_map[display_name] = f
            # --- FIM DO BLOCO CORRIGIDO PARA CRIAR A LISTA DE SELEÇÃO ---


            # Encontra o índice padrão na lista de exibição
            default_model_index = 0 # Default para o primeiro da lista se best_model não estiver lá ou lista vazia
            if default_model_display_name and default_model_display_name in model_display_options:
                 default_model_index = model_display_options.index(default_model_display_name)
            # Não precisamos de else, pois default_model_index já é 0

            if len(model_display_options) > 0:
                 selected_display_name = st.selectbox(
                    "Escolha o modelo treinado para fazer a previsão:",
                    options=model_display_options,
                    index=default_model_index,
                    key="prediction_model_selector"
                 )

                 # Obtém o nome do ficheiro real a partir do nome de exibição selecionado
                 selected_model_filename = model_filename_map.get(selected_display_name)

                 # Carrega o modelo selecionado usando a função load_specific_model
                 # ESTA FUNÇÃO load_specific_model DEVE ESTAR DEFINIDA ANTES DESTE BLOCO elif
                 # Assume que load_specific_model trata erros de ficheiro não encontrado
                 if selected_model_filename: # Só tenta carregar se obteve um nome de ficheiro válido do mapeamento
                      selected_model_instance = load_specific_model(selected_model_filename)
                 else:
                      st.error(f"Erro interno: Não foi possível mapear o nome selecionado '{selected_display_name}' para um ficheiro de modelo.")
                      selected_model_instance = None


                 # --- Exibe o tipo do MELHOR modelo (best_model.joblib) e a informação recomendada ---
                 # A variável 'model' (carregada globalmente no início) representa best_model.joblib.
                 # Se 'model' foi carregado com sucesso, usamos o seu tipo.
                 # Isso informa o TIPO do best_model e a recomendação, independentemente de qual modelo está selecionado no momento.
                 if 'model' in globals() and model is not None:
                     best_model_type = type(model).__name__
                     st.info(f"O modelo recomendado é o **{best_model_type}** ('best_model.joblib') pois foi o que obteve melhores resultados no conjunto de teste.")
                 # Removidos os warnings/erros redundantes aqui para simplificar a UI se best_model global não carregar


            else: # Caso não encontre nenhum ficheiro de modelo (model_files está vazio)
                 st.warning("Nenhum ficheiro de modelo (.joblib) encontrado na pasta 'artefacts/' para seleção. Certifique-se de que os modelos foram salvos.")
                 selected_model_instance = None # Garante que selected_model_instance seja None
                 selected_model_filename = None # Garante que selected_model_filename seja None


        except FileNotFoundError:
             st.error("❌ A pasta 'artefacts/' não foi encontrada. Certifique-se de que existe.")
             selected_model_instance = None
             selected_model_filename = None
        except Exception as e:
             st.error(f"❌ Ocorreu um erro ao listar ou carregar os modelos: {e}")
             selected_model_instance = None
             selected_model_filename = None


        st.markdown("### Dados do Aluno") # Subtítulo antes dos inputs de dados

        st.markdown("#### Características Numéricas")
        # Abre o container do grid para as features numéricas
        st.markdown('<div class="input-grid-container">', unsafe_allow_html=True)

        # Não precisa de col_idx aqui, o grid cuida das colunas
        for feature in numeric_features:
            # Calcula os valores min/max/mean originais
            min_val_original = student_df_original[feature].min()
            max_val_original = student_df_original[feature].max()
            mean_val_original = student_df_original[feature].mean()

            # Determina o tipo de dado e formato apropriado para o input numérico
            if original_dtypes[feature] == 'int64' or feature in ordinal_numeric_features_to_map:
                # Para inteiros ou ordinais mapeadas (que são int no dataset),
                # explicitamente converte min/max/mean para int, tratando NaN.
                input_min = int(min_val_original) if pd.notna(min_val_original) else 0
                input_max = int(max_val_original) if pd.notna(max_val_original) else None
                input_value = int(round(mean_val_original)) if pd.notna(mean_val_original) else (int(min_val_original) if pd.notna(min_val_original) else 0)
                input_step = 1
                input_format = "%d" # Formato para exibir como inteiro
            else: # Assume float ou outro tipo numérico
                # Para floats, use valores float e formato float
                input_min = float(min_val_original) if pd.notna(min_val_original) else 0.0
                input_max = float(max_val_original) if pd.notna(max_val_original) else None
                input_value = float(mean_val_original) if pd.notna(mean_val_original) else (float(min_val_original) if pd.notna(min_val_original) else 0.0)
                input_step = 0.1
                input_format = "%.2f" # Formato para exibir 2 casas decimais (ou remova para default do streamlit)

            # Abre o item do grid para esta feature
            st.markdown('<div class="grid-item">', unsafe_allow_html=True) # Corrigido para grid-item

            # --- Adiciona o nome da feature em negrito e a descrição curta em letra menor ---
            # Usa o dicionário feature_descriptions_short (que deve estar definido acima)
            description = feature_descriptions_short.get(feature, 'Descrição não disponível')
            st.markdown(f"<div class='feature-label-container'><strong>{feature.replace('_', ' ').title()}</strong> <span class='small-description'>({description})</span></div>", unsafe_allow_html=True)

            # Verifica se a feature é ordinal numérica para mostrar a descrição da escala como info
            if feature in ordinal_numeric_features_to_map: # Assume ORDINAL_MAPPINGS is globally available
                mapping_dict = ORDINAL_MAPPINGS[feature]
                mapping_desc = ", ".join([f"{k}: {v}" for k, v in mapping_dict.items()])
                st.info(mapping_desc) # Mostra a descrição como um info box

            # Cria o widget de input numérico
            # label="" para não duplicar o nome já mostrado com Markdown
            input_data[feature] = st.number_input(
                 label="",
                min_value=input_min,
                max_value=input_max,
                value=input_value,
                step=input_step,
                format=input_format,
                key=f"input_{feature}" # Chave única
            )

            # Fecha o item do grid para esta feature
            st.markdown('</div>', unsafe_allow_html=True) # Corrigido para grid-item


        # Fecha o container do grid para as features numéricas
        st.markdown('</div>', unsafe_allow_html=True)


        st.markdown("#### Características Categóricas/Binárias")
        # Abre um novo container do grid para as features categóricas
        st.markdown('<div class="input-grid-container">', unsafe_allow_html=True)
        # Não precisa de col_idx aqui dentro pois cada item do grid cuida da sua própria coluna
        for feature in categorical_features:
             # Abre o item do grid para esta feature
             st.markdown('<div class="grid-item">', unsafe_allow_html=True) # Corrigido para grid-item

             # --- Adiciona o nome da feature em negrito e a descrição curta em letra menor ---
             # Usa o dicionário feature_descriptions_short (que deve estar definido acima)
             description = feature_descriptions_short.get(feature, 'Descrição não disponível')
             st.markdown(f"<div class='feature-label-container'><strong>{feature.replace('_', ' ').title()}</strong> <span class='small-description'>({description})</span></div>", unsafe_allow_html=True)


             # Cria o widget de input selectbox com label vazio
             options = student_df_original[feature].dropna().unique().tolist()
             input_data[feature] = st.selectbox(
                 label="", # Label vazio para não duplicar o nome já mostrado com Markdown
                 options=options,
                 index=options.index(student_df_original[feature].mode()[0]) if not student_df_original[feature].mode().empty and student_df_original[feature].mode()[0] in options else 0, # Usar o modo como default
                 key=f"input_{feature}" # Chave única
             )
             # Fecha o item do grid para esta feature
             st.markdown('</div>', unsafe_allow_html=True) # Corrigido para grid-item


        # Fecha o container do grid para as features categóricas
        st.markdown('</div>', unsafe_allow_html=True)


        st.markdown("---")
        # Verifica se o modelo foi carregado com sucesso antes de permitir a previsão
        if selected_model_instance is not None:
            if st.button("🚀 Prever Resultado do Aluno"):
                # Código de previsão (mantido o mesmo, com ajustes nos warnings/erros conforme conversamos)
                # Criar DataFrame com dados de input, garantindo que todas as colunas originais estejam presentes
                input_df_raw = pd.DataFrame(columns=original_cols)
                input_df_raw.loc[0] = pd.NA # Inicializa com NA para garantir dtypes
                for col, val in input_data.items():
                     input_df_raw.loc[0, col] = val

                st.write("Dados de entrada para previsão:")
                st.dataframe(input_df_raw, use_container_width=True)


                loading_animation(f"Aplicando pré-processamento e prevendo com {selected_model_filename}...")
                try:
                    # Verificar se as colunas de input_df_raw correspondem às colunas que o preprocessor espera
                    if list(input_df_raw.columns) != list(original_cols):
                        st.error("❌ Erro de compatibilidade: As colunas dos dados de entrada não correspondem às colunas originais esperadas pelo pré-processador.")
                        raise ValueError("Colunas de input incompatíveis")


                    input_processed = preprocessor.transform(input_df_raw) # Assume preprocessor is globally available
                    st.success("✅ Pré-processamento aplicado.")

                    prediction = selected_model_instance.predict(input_processed) # Usa o modelo SELECIONADO

                    y_proba_input = None
                    if hasattr(selected_model_instance, 'predict_proba'):
                         try:
                              y_proba_input = selected_model_instance.predict_proba(input_processed) # Usa o modelo SELECIONADO
                         except Exception as proba_e:
                               st.info("Probabilidades não disponíveis para este modelo ou houve um erro ao calculá-las.")
                               y_proba_input = None


                    predicted_class_index = prediction[0]
                    # Garantir que o índice existe em CLASS_NAMES
                    if 0 <= predicted_class_index < len(CLASS_NAMES): # Assume CLASS_NAMES is globally available
                        predicted_class_label = CLASS_NAMES[predicted_class_index]
                    else:
                        predicted_class_label = f"Índice Desconhecido ({predicted_class_index})"
                        st.error(f"Previsão retornou um índice de classe inesperado: {predicted_class_index}")


                    st.markdown('<h2 class="sub-header">Resultado da Previsão:</h2>', unsafe_allow_html=True)

                    if predicted_class_label == 'yes':
                         st.balloons()
                         st.success(f"🎉 Previsão: O aluno **PROVAVELMENTE PASSARÁ** no exame final!")
                    elif predicted_class_label == 'no':
                         st.info(f"😟 Previsão: O aluno **PROVAVELMENTE NÃO PASSARÁ** no exame final.")
                    else:
                         st.info(f"Previsão: {predicted_class_label}") # Mostrar label desconhecida em caso de erro


                    st.markdown("---")
                    st.markdown("#### Detalhes da Previsão")
                    st.write(f"- Modelo Utilizado: **{selected_model_filename}**") # Mostrar qual modelo foi usado
                    st.write(f"- Classe Prevista: **{predicted_class_label}**")

                    if y_proba_input is not None and y_proba_input.shape[1] == len(CLASS_NAMES): # Verifica se há probabilidades para todas as classes esperadas
                         try:
                              # Encontra a probabilidade da classe 'yes'
                              proba_yes = y_proba_input[0][CLASS_NAMES.index('yes')]
                              proba_no = y_proba_input[0][CLASS_NAMES.index('no')]

                              st.write(f"- Probabilidade de Passar ('yes'): **{proba_yes:.2f}**")
                              st.write(f"- Probabilidade de Não Passar ('no'): **{proba_no:.2f}**")
                         except ValueError:
                               st.info("Não foi possível exibir as probabilidades para as classes esperadas.")
                         except Exception as e:
                               st.info(f"Ocorreu um erro ao exibir as probabilidades: {e}")

                    else:
                         st.info("Probabilidades não disponíveis ou incompatíveis para este modelo/previsão.")

                    st.info("Nota: Esta é uma previsão baseada no modelo treinado e nos dados fornecidos.")

                except Exception as e:
                     st.error(f"❌ Ocorreu um erro ao fazer a previsão: {e}")
                     st.info("Verifique se todos os dados de entrada estão corretos e se o pré-processador e modelo carregados são compatíveis.")
        else: # Mensagem se selected_model_instance for None (nenhum modelo carregado ou erro)
             st.warning("Não é possível fazer a previsão. Por favor, selecione um modelo válido e verifique se os artefactos (modelos joblib) estão na pasta 'artefacts/'.")


# --- Análise do Modelo Treinado ---
elif menu == "Análise do Modelo Treinado":
    st.markdown('<h1 class="main-header">Análise do Modelo Treinado para Intervenção Estudantil</h1>', unsafe_allow_html=True)
    st.markdown('<p class="info-text">Aqui pode ver as métricas de avaliação, a matriz de confusão e a interpretabilidade do modelo (`best_model.joblib`) que foi treinado no seu dataset e guardado como artefacto.</p>', unsafe_allow_html=True)

    st.warning("⚠️ Esta secção mostra a performance do modelo PRÉ-TREINADO (`best_model.joblib`) nos dados de teste processados. Para comparar diferentes algoritmos, vá à secção 'Avaliação e Comparação de Modelos'.")

    # Assume que test_df_processed_global, model, original_cols, processed_cols,
    # TARGET_PROCESSED_NAME, CLASS_NAMES estão definidos globalmente.

    if test_df_processed_global is None:
        st.warning("Conjunto de teste processado não foi carregado. Esta secção não está disponível. Verifique o caminho do ficheiro 'data/processed/test_processed.csv'.")
    elif model is None: # 'model' é a variável global carregada de best_model.joblib
         st.error("Modelo treinado ('best_model.joblib') não foi carregado. Esta secção não está disponível. Verifique a pasta 'artefacts/'.")
    elif 'processed_cols' not in locals() or processed_cols is None:
         st.error("Não foi possível carregar os nomes das características processadas. A secção de Análise do Modelo Treinado não está disponível.")
    else:
        # Verifica se a coluna alvo processada existe no dataframe de teste
        if TARGET_PROCESSED_NAME in test_df_processed_global.columns:
            # Prepara X_test e y_test usando os dados processados
            X_test_processed = test_df_processed_global.drop(columns=[TARGET_PROCESSED_NAME])
            y_test_processed = test_df_processed_global[TARGET_PROCESSED_NAME]

            # Verifica a compatibilidade das colunas de teste com as colunas processadas esperadas
            if list(X_test_processed.columns) != list(processed_cols):
                st.error("❌ Erro de compatibilidade: As colunas do conjunto de teste processado não correspondem aos nomes das características processadas carregadas.")
                st.warning("Verifique se os ficheiros em 'data/processed/' e os artefactos foram gerados consistentemente.")
            else:
                # Botão para iniciar a avaliação
                if st.button("Avaliar o Modelo Treinado no Conjunto de Teste"):
                    loading_animation("Avaliando o modelo treinado...")
                    try:
                        # Faz previsões usando o modelo global 'model' (best_model.joblib)
                        y_pred_loaded_model = model.predict(X_test_processed)

                        # Tenta obter probabilidades se o modelo suportar
                        y_proba_loaded_model = None
                        if hasattr(model, 'predict_proba'):
                            try:
                                # Obtém probabilidades para as classes
                                # Assume que a classe positiva ('yes') corresponde à coluna 1
                                y_proba_loaded_model = model.predict_proba(X_test_processed)
                            except Exception as proba_e:
                                st.info(f"Probabilidades não disponíveis para este modelo ou houve um erro ao calculá-las: {proba_e}")
                                y_proba_loaded_model = None


                        st.markdown('<h2 class="sub-header">Métricas de Avaliação no Conjunto de Teste</h2>', unsafe_allow_html=True)

                        # --- Exibe Métricas de Avaliação ---
                        accuracy = accuracy_score(y_test_processed, y_pred_loaded_model)
                        # Report de classificação
                        report_dict = classification_report(y_test_processed, y_pred_loaded_model,
                                                            target_names=CLASS_NAMES,
                                                            output_dict=True, zero_division=0)
                        report_df = pd.DataFrame(report_dict).transpose()

                        # AUC ROC (se probabilidades disponíveis)
                        roc_auc = None
                        # Verifica se o modelo tem predict_proba e se tem 2 colunas de probabilidade
                        if y_proba_loaded_model is not None and y_proba_loaded_model.shape[1] == 2:
                             try:
                                  # Certifica-se que está a usar a probabilidade da classe 'yes' (classe 1)
                                  # Assume que a label 1 mapeia para 'yes' no y_test_processed
                                  roc_auc = roc_auc_score(y_test_processed, y_proba_loaded_model[:, 1])
                             except ValueError as auc_ve:
                                  st.warning(f"Não foi possível calcular AUC ROC: {auc_ve}. Verifique as labels das classes nos dados de teste.")
                             except Exception as auc_e:
                                  st.warning(f"Erro inesperado ao calcular AUC ROC: {auc_e}")


                        col_metrics1, col_metrics2 = st.columns(2)

                        with col_metrics1:
                            st.markdown("#### Relatório de Classificação")
                            st.dataframe(report_df.round(2), use_container_width=True)

                            st.markdown("#### Métricas Resumo")
                            col_met1, col_met2, col_met3, col_met4 = st.columns(4)
                            with col_met1: st.metric("Acurácia", f"{accuracy:.2f}")
                            with col_met2:
                                 if 'weighted avg' in report_df.index:
                                     st.metric("Precisão (Avg)", f"{report_df.loc['weighted avg', 'precision']:.2f}")
                                 else: st.info("N/A")
                            with col_met3:
                                if 'weighted avg' in report_df.index:
                                    st.metric("Recall (Avg)", f"{report_df.loc['weighted avg', 'recall']:.2f}")
                                else: st.info("N/A")
                            with col_met4:
                                if 'weighted avg' in report_df.index:
                                    st.metric("F1-Score (Avg)", f"{report_df.loc['weighted avg', 'f1-score']:.2f}")
                                else: st.info("N/A")
                            # Exibe AUC ROC se calculado
                            if roc_auc is not None:
                                 st.metric("AUC ROC", f"{roc_auc:.2f}")
                            else:
                                 st.info("AUC ROC: N/A (Probabilidades não disponíveis ou erro)")


                        with col_metrics2:
                             # --- Exibe Matriz de Confusão ---
                             # Assumes plot_confusion_matrix_interactive is globally defined
                             fig_cm_loaded_model, cm_matrix_loaded_model = plot_confusion_matrix_interactive(y_test_processed, y_pred_loaded_model, class_names=CLASS_NAMES)
                             st.plotly_chart(fig_cm_loaded_model, use_container_width=True)

                        st.markdown("---")
                        st.markdown('<h3 class="sub-header">Análise da Matriz de Confusão</h3>', unsafe_allow_html=True)
                        # REMOVIDA a chamada para analyze_square_matrix aqui.
                        # analyze_square_matrix(cm_matrix_loaded_model, title="Propriedades Matemáticas da CM") # <-- REMOVIDO

                        # Exibe TP, TN, FP, FN para matriz 2x2 (mantido, é relevante para CM)
                        if cm_matrix_loaded_model.shape == (2, 2):
                             # Assumindo classe 0 = 'no', classe 1 = 'yes' e que os resultados são 0/1
                             # Verificação mais robusta pode usar model.classes_ se necessário
                             if all(x in [0, 1] for x in np.unique(y_test_processed)) and all(x in [0, 1] for x in np.unique(y_pred_loaded_model)):
                                 tn, fp, fn, tp = cm_matrix_loaded_model[0,0], cm_matrix_loaded_model[0,1], cm_matrix_loaded_model[1,0], cm_matrix_loaded_model[1,1]
                                 st.write(f"**Verdadeiros Positivos (TP):** {tp}")
                                 st.write(f"**Verdadeiros Negativos (TN):** {tn}")
                                 st.write(f"**Falsos Positivos (FP):** {fp}")
                                 st.write(f"**Falsos Negativos (FN):** {fn}")
                                 st.info("""
                                 *   **TP:** Previsto Passou ('yes'), Real Passou ('yes')
                                 *   **TN:** Previsto Não Passou ('no'), Real Não Passou ('no')
                                 *   **FP:** Previsto Passou ('yes'), Real Não Passou ('no') - Intervenção perdida...
                                 *   **FN:** Previsto Não Passou ('no'), Real Passou ('yes') - Intervenção desnecessária...
                                 """)
                                 st.warning("💡 No contexto de intervenção estudantil, Falsos Negativos (FN) são geralmente mais críticos, pois representam alunos que precisavam de ajuda mas não foram identificados.")
                             else:
                                 st.warning("As labels das classes nos resultados não são 0 e 1. As métricas TN/FP/FN/TP podem não ser exibidas corretamente.")


                        # Mantido o separador e o título para a Importância das Features
                        st.markdown('---')
                        st.markdown('<h3 class="sub-header">Importância das Características (Modelo Treinado)</h3>', unsafe_allow_html=True)
                        st.markdown('<p class="info-text">Quais características foram mais relevantes para a decisão do seu modelo treinado (`best_model.joblib`), em relação aos dados PÓS pré-processamento.</p>', unsafe_allow_html=True)

                        # Exibe Feature Importance ou Coeficientes para o modelo GLOBAL 'model' (best_model)
                        # Assume processed_cols is globally available
                        processed_feature_names_for_plot = processed_cols

                        if hasattr(model, 'feature_importances_'): # Usa o modelo global 'model'
                            # Ensure feature_importances_ length matches processed_feature_names_for_plot
                            if len(model.feature_importances_) == len(processed_feature_names_for_plot):
                                feature_importance_df = pd.DataFrame({
                                    'Característica Processada': processed_feature_names_for_plot,
                                    'Importância': model.feature_importances_
                                }).sort_values('Importância', ascending=False)

                                fig_importance = px.bar(
                                    feature_importance_df.head(min(30, len(feature_importance_df))), # Mostrar mais features se houver espaço
                                    x='Importância',
                                    y='Característica Processada',
                                    orientation='h',
                                    title=f"Importância das Características (Processadas) para o Modelo Treinado ({type(model).__name__})" # Inclui o tipo do modelo
                                )
                                fig_importance.update_layout(yaxis={'categoryorder':'total ascending'})
                                st.plotly_chart(fig_importance, use_container_width=True)
                                st.info("A importância mostrada é para as características APÓS o pré-processamento (incluindo One-Hot Encoding, etc.).")
                            else:
                                st.error(f"❌ Erro: O número de importâncias ({len(model.feature_importances_)}) não corresponde ao número de características processadas ({len(processed_feature_names_for_plot)}).")
                                st.warning("Isso pode acontecer se os artefactos de colunas processadas não corresponderem ao modelo guardado.")

                        elif hasattr(model, 'coef_'): # Usa o modelo global 'model'
                             # Verifica se o modelo é linear e tem coeficientes por feature
                             if hasattr(model.coef_, 'ndim') and (model.coef_.ndim == 1 or (model.coef_.ndim == 2 and model.coef_.shape[0] == 1)) and len(model.coef_[0] if model.coef_.ndim == 2 else model.coef_) == len(processed_feature_names_for_plot):
                                  coef_values = model.coef_[0] if model.coef_.ndim == 2 else model.coef_ # Use coef_ for binary, first row for multi-class (simplistic)
                                  feature_coef_df = pd.DataFrame({
                                     'Característica Processada': processed_feature_names_for_plot,
                                     'Coeficiente': coef_values
                                  }).sort_values('Coeficiente', ascending=False)

                                  coef_min = feature_coef_df['Coeficiente'].min()
                                  coef_max = feature_coef_df['Coeficiente'].max()
                                  abs_max = max(abs(coef_min), abs(coef_max)) if coef_min is not None and coef_max is not None else 1.0 # Evitar divisão por zero

                                  fig_coef = px.bar(
                                      feature_coef_df.head(min(30, len(feature_coef_df))), # Mostrar mais features
                                      x='Coeficiente',
                                      y='Característica Processada',
                                      orientation='h',
                                      color='Coeficiente',
                                      color_continuous_scale='RdBu',
                                      range_color=[-abs_max, abs_max] if abs_max > 1e-9 else None, # Set range only if valid
                                      title=f"Coeficientes das Características (Processadas) para o Modelo Treinado ({type(model).__name__})" # Inclui o tipo do modelo
                                  )
                                  fig_coef.update_layout(yaxis={'categoryorder':'total ascending'})
                                  st.plotly_chart(fig_coef, use_container_width=True)
                                  st.info("Coeficientes mostrados são para características APÓS pré-processamento. A magnitude indica a importância; o sinal indica a direção do efeito na probabilidade da classe positiva.")
                             else:
                                st.warning("O modelo tem coeficientes, mas a visualização direta é complexa ou incompatível com as características processadas.")


                        else:
                            st.info(f"O modelo treinado ({type(model).__name__}) não fornece importância ou coeficientes de característica de forma padrão.")


                    except Exception as e:
                         st.error(f"❌ Ocorreu um erro ao avaliar o modelo treinado: {e}")
                         st.info("Verifique se o conjunto de teste processado corresponde ao formato esperado pelo modelo carregado.")

        else: # Handle case where target column is missing in processed test data
             st.warning(f"A coluna alvo processada '{TARGET_PROCESSED_NAME}' não foi encontrada no conjunto de teste processado.")


# Importações adicionais necessárias (adicione no topo do seu script)
# from sklearn.tree import plot_tree
# import matplotlib.pyplot as plt # Para exibir a árvore de decisão

# Importações adicionais necessárias (adicione no topo do seu script, se ainda não o fez)
# from sklearn.tree import plot_tree
# import matplotlib.pyplot as plt # Para exibir a árvore de decisão

# Importações adicionais necessárias (adicione no topo do seu script, se ainda não o fez)
# from sklearn.tree import plot_tree
# import matplotlib.pyplot as plt # Para exibir a árvore de decisão

# --- Análise de Matriz (Modificado para apenas Matriz de Confusão por Modelo com Análises Únicas e sem propriedades matemáticas genéricas da CM) ---
elif menu == "Análise de Matriz":
    # Título e subtítulo atualizados para refletir o foco na comparação de modelos via CM
    st.markdown('<h1 class="main-header">Avaliação e Comparação de Modelos</h1>', unsafe_allow_html=True)
    st.markdown('<p class="info-text">Selecione diferentes tipos de modelos para visualizar a sua Matriz de Confusão, principais métricas e algumas características únicas do algoritmo no conjunto de teste processado.</p>', unsafe_allow_html=True)


    st.markdown('<h2 class="sub-header">Matriz de Confusão por Tipo de Modelo</h2>', unsafe_allow_html=True)
    st.markdown('<p class="info-text">Escolha um algoritmo de Machine Learning para treinar *temporariamente* nos seus dados de treino processados e avaliar o seu desempenho no conjunto de teste.</p>', unsafe_allow_html=True)


    # Verificações essenciais antes de tentar treinar e avaliar
    # Assume que train_df_processed_global, test_df_processed_global,
    # TARGET_PROCESSED_NAME, processed_cols, CLASS_NAMES,
    # AVAILABLE_MODELS_FOR_ANALYSIS estão definidos globalmente.
    if train_df_processed_global is None or test_df_processed_global is None:
        st.warning("Os conjuntos de treino ou teste processados não foram carregados. Não é possível gerar a Matriz de Confusão ou métricas. Verifique os ficheiros em 'data/processed/'.")
    elif 'processed_cols' not in locals() or processed_cols is None:
        st.error("Não foi possível carregar os nomes das características processadas. A avaliação de modelos não está disponível.")
    else:
        # Verifica se as colunas alvo processadas existem nos dataframes
        if TARGET_PROCESSED_NAME in train_df_processed_global.columns and TARGET_PROCESSED_NAME in test_df_processed_global.columns:
             # Prepara X e y para treino e teste
             X_train_processed = train_df_processed_global.drop(columns=[TARGET_PROCESSED_NAME])
             y_train_processed = train_df_processed_global[TARGET_PROCESSED_NAME]
             X_test_processed = test_df_processed_global.drop(columns=[TARGET_PROCESSED_NAME])
             y_test_processed = test_df_processed_global[TARGET_PROCESSED_NAME] # <-- Correta

             # Verifica a compatibilidade das colunas
             if list(X_train_processed.columns) != list(X_test_processed.columns) or list(X_train_processed.columns) != list(processed_cols): # Comparar com a lista carregada também
                  st.error("❌ Erro de compatibilidade: As colunas dos dados de treino/teste processados não correspondem aos nomes das features processadas carregadas.")
                  st.warning("Verifique se os ficheiros em 'data/processed/' foram gerados consistentemente com os artefactos.")
             else:
                # Selectbox para escolher o ALGORITMO (tipo de modelo)
                # Assumes AVAILABLE_MODELS_FOR_ANALYSIS is globally defined (no topo do script)
                selected_model_name = st.selectbox(
                    "Escolha o tipo de algoritmo para treinar e avaliar:",
                    list(AVAILABLE_MODELS_FOR_ANALYSIS.keys()),
                    key="cm_model_selector" # Chave única para este widget
                )

                # --- Opção para configurar parâmetros do modelo selecionado (Opcional) ---
                # Note: Estes parâmetros aplicam-se apenas ao modelo TREINADO AQUI,
                # não ao best_model.joblib da secção "Análise do Modelo Treinado".
                st.markdown('#### Configuração do Algoritmo (Opcional)', unsafe_allow_html=True)
                current_model_key = selected_model_name # Usa o nome selecionado
                model_params = {} # Dicionário para guardar parâmetros configuráveis

                # Lógica de configuração baseada no tipo de modelo selecionado
                # Verifica se o modelo selecionado está no dicionário AVAILABLE_MODELS_FOR_ANALYSIS
                if current_model_key in AVAILABLE_MODELS_FOR_ANALYSIS:
                    # Obtém a instância base do dicionário para verificar o tipo e defaults
                    base_model_instance = AVAILABLE_MODELS_FOR_ANALYSIS[current_model_key]
                    # model_class = type(base_model_instance) # Não precisamos da classe aqui

                    # Adiciona sliders de parâmetros comuns ou específicos
                    if current_model_key == "KNN":
                        # Exemplo de parâmetro para KNN: n_neighbors
                        # Usamos getattr para verificar se o atributo existe antes de tentar usá-lo
                        default_n = getattr(base_model_instance, 'n_neighbors', 5) # Padrão 5 se não existir ou for None
                        model_params['n_neighbors'] = st.slider(f"{current_model_key}: Número de Vizinhos (n_neighbors)", 1, min(20, len(X_train_processed)), int(default_n), key=f"{current_model_key}_n_neighbors") # Max vizinhos limitado pelo tamanho do treino


                    elif current_model_key in ["Decision Tree", "Random Forest"]:
                        # Parâmetros comuns para árvores
                        # Valor máximo razoável para max_depth, None significa profundidade total
                        # max_possible_depth = base_model_instance.get_params().get('max_depth', 1000) # Get default if any (not needed here)
                        default_max_depth = getattr(base_model_instance, 'max_depth', 3 if current_model_key == "Decision Tree" else 5) # Padrão diferente para DT vs RF
                        # Slider para max_depth - Adicionar opção 'Profundidade Total' ou None? Sliders não suportam None.
                        # Vamos usar um valor alto para "quase total" ou limitar a um valor razoável.
                        slider_max_depth = st.slider(f"{current_model_key}: Profundidade Máxima (max_depth)", 1, 15, int(default_max_depth) if default_max_depth is not None and default_max_depth <= 15 else (3 if current_model_key == "Decision Tree" else 5), key=f"{current_model_key}_max_depth")
                        model_params['max_depth'] = slider_max_depth # Atribui o valor do slider

                        default_min_samples_split = getattr(base_model_instance, 'min_samples_split', 2)
                        model_params['min_samples_split'] = st.slider(f"{current_model_key}: Mínimo de Amostras para Dividir (min_samples_split)", 2, 20, int(default_min_samples_split), key=f"{current_model_key}_min_samples_split")

                        if current_model_key == "Random Forest":
                            default_n_estimators = getattr(base_model_instance, 'n_estimators', 100)
                            model_params['n_estimators'] = st.slider(f"{current_model_key}: Número de Árvores (n_estimators)", 50, 500, int(default_n_estimators), key=f"{current_model_key}_n_estimators")

                    elif current_model_key in ["Regressão Logística", "SVM (Kernel RBF)", "Gradient Boosting", "AdaBoost"]:
                         # Adicionar parâmetros para estes modelos se quiser
                         pass # Por agora, sem parâmetros configuráveis para estes


                else: # No specific configurable parameters for this model in the list above
                    st.info(f"Não há parâmetros configuráveis disponíveis para o algoritmo **{current_model_key}** neste momento.")


                if st.button(f"Treinar e Avaliar {selected_model_name}", key="train_evaluate_button"): # Adicionada chave para evitar Warning
                    loading_animation(f"Treinando {selected_model_name} com {len(X_train_processed)} amostras e avaliando em {len(X_test_processed)} amostras...")
                    try:
                        # Obtém a instância base do modelo e cria uma nova instância com os parâmetros configurados
                        if current_model_key in AVAILABLE_MODELS_FOR_ANALYSIS:
                            base_model_instance = AVAILABLE_MODELS_FOR_ANALYSIS[current_model_key]
                            model_class = type(base_model_instance)
                            # Cria uma nova instância com os parâmetros configurados
                            model_instance = model_class(**model_params)

                            # Treina o modelo nos dados de treino processados
                            model_instance.fit(X_train_processed, y_train_processed)

                            # Faz previsões no conjunto de teste processado
                            y_pred = model_instance.predict(X_test_processed)

                            # Tenta obter probabilidades se o modelo suportar
                            y_proba_loaded_model = None
                            if hasattr(model_instance, 'predict_proba'):
                                try:
                                    y_proba_loaded_model = model_instance.predict_proba(X_test_processed)
                                except Exception as proba_e:
                                    st.info(f"Probabilidades (predict_proba) não disponíveis para {selected_model_name} ou houve um erro ao calculá-las: {proba_e}")
                                    y_proba_loaded_model = None

                            st.markdown('<h3 class="sub-header">Resultados de Avaliação no Conjunto de Teste</h3>', unsafe_allow_html=True)

                            # --- Exibe Métricas de Avaliação ---
                            accuracy = accuracy_score(y_test_processed, y_pred)
                            report_dict = classification_report(y_test_processed, y_pred,
                                                                target_names=CLASS_NAMES,
                                                                output_dict=True, zero_division=0)
                            report_df = pd.DataFrame(report_dict).transpose()

                            # AUC ROC (se probabilidades disponíveis e binário)
                            roc_auc = None
                            # Verifica se o modelo tem predict_proba e se é um problema de classificação binária (2 classes)
                            if y_proba_loaded_model is not None and y_proba_loaded_model.shape[1] == 2 and len(model_instance.classes_) == 2:
                                try:
                                    # Encontra o índice da classe positiva ('yes') nas classes do modelo treinado
                                    class_labels_in_model = list(model_instance.classes_)
                                    # Preferir label numérica 1 se existir e 'yes' for a classe positiva
                                    positive_class_label_in_model = None
                                    if 'yes' in class_labels_in_model:
                                        positive_class_label_in_model = 'yes'
                                    elif 1 in class_labels_in_model: # Se 1 existir
                                        positive_class_label_in_model = 1
                                    elif len(class_labels_in_model) == 2: # Binário com outras labels, assumir a segunda como positiva
                                         positive_class_label_in_model = class_labels_in_model[1]

                                    if positive_class_label_in_model is not None:
                                        positive_class_index_in_model = class_labels_in_model.index(positive_class_label_in_model)
                                        # Certifica-se que y_test_processed tem as labels corretas (0 e 1 ou 'no' e 'yes') para roc_auc_score
                                        # roc_auc_score espera y_true com 0s e 1s
                                        y_true_binary_for_auc = y_test_processed.map({'no': 0, 'yes': 1}) if y_test_processed.dtype == 'object' else y_test_processed
                                        # Se o target original era numérico, pode já ser 0/1. Se era categórico ('no'/'yes'), precisa do map.

                                        # Verifica se y_true_binary_for_auc tem apenas 0s e 1s (após o map, se aplicável)
                                        if all(x in [0, 1] for x in y_true_binary_for_auc.dropna().unique()):
                                             roc_auc = roc_auc_score(y_true_binary_for_auc, y_proba_loaded_model[:, positive_class_index_in_model])
                                        else:
                                             st.warning("Formato inesperado na coluna alvo de teste para cálculo de AUC ROC. Esperava 0s e 1s após mapeamento.")


                                    else:
                                         st.warning(f"Classe positiva ('yes' ou 1) não encontrada nas classes do modelo treinado: {class_labels_in_model}. Não é possível calcular AUC ROC.")

                                except ValueError as auc_ve:
                                     st.warning(f"Não foi possível calcular AUC ROC: {auc_ve}. Verifique as labels das classes.")
                                except Exception as auc_e:
                                     st.warning(f"Erro inesperado ao calcular AUC ROC: {auc_e}")


                            col_metrics1, col_metrics2 = st.columns(2)

                            with col_metrics1:
                                st.markdown(f"#### Relatório de Classificação ({selected_model_name})")
                                st.dataframe(report_df.round(2), use_container_width=True)

                                st.markdown("#### Métricas Resumo")
                                col_met1, col_met2, col_met3, col_met4 = st.columns(4)
                                with col_met1: st.metric("Acurácia", f"{accuracy:.2f}")
                                with col_met2:
                                     if 'weighted avg' in report_df.index:
                                         st.metric("Precisão (Avg)", f"{report_df.loc['weighted avg', 'precision']:.2f}")
                                     else: st.info("N/A")
                                with col_met3:
                                    if 'weighted avg' in report_df.index:
                                        st.metric("Recall (Avg)", f"{report_df.loc['weighted avg', 'recall']:.2f}")
                                    else: st.info("N/A")
                                with col_met4:
                                    if 'weighted avg' in report_df.index:
                                        st.metric("F1-Score (Avg)", f"{report_df.loc['weighted avg', 'f1-score']:.2f}")
                                    else: st.info("N/A")
                                # Exibe AUC ROC se calculado
                                if roc_auc is not None:
                                     st.metric("AUC ROC", f"{roc_auc:.2f}")
                                else:
                                     st.info("AUC ROC: N/A (Probabilidades não disponíveis ou erro)")


                            with col_metrics2:
                                 # --- Exibe Matriz de Confusão ---
                                 fig_cm, cm_matrix = plot_confusion_matrix_interactive(y_test_processed, y_pred, class_names=CLASS_NAMES)
                                 st.plotly_chart(fig_cm, use_container_width=True)

                            st.markdown("---")
                            st.markdown('<h3 class="sub-header">Análise da Matriz de Confusão</h3>', unsafe_allow_html=True)
                            # REMOVIDA a chamada para analyze_square_matrix aqui.
                            # analyze_square_matrix(cm_matrix, title="Propriedades Matemáticas da CM") # <-- REMOVIDO

                            # Exibe TP, TN, FP, FN para matriz 2x2
                            if cm_matrix.shape == (2, 2):
                                 # Assumindo classe 0 = 'no', classe 1 = 'yes' e que os resultados são 0/1
                                 # Verificação mais robusta pode usar model_instance.classes_ se necessário
                                 # A lógica aqui deve espelhar o que é feito no AUC ROC para as labels
                                 class_labels_in_model = list(model_instance.classes_)
                                 try: # Usar try-except para acesso seguro aos elementos da CM
                                     # Encontra os índices para 'no' (0) e 'yes' (1) nas classes do modelo
                                     no_idx_in_model = class_labels_in_model.index('no') if 'no' in class_labels_in_model else (class_labels_in_model.index(0) if 0 in class_labels_in_model else None)
                                     yes_idx_in_model = class_labels_in_model.index('yes') if 'yes' in class_labels_in_model else (class_labels_in_model.index(1) if 1 in class_labels_in_model else None)

                                     if no_idx_in_model is not None and yes_idx_in_model is not None:
                                         # Assumindo que a Matriz de Confusão gerada pelo sklearn segue a ordem model_instance.classes_
                                         # cm[real_idx, predicted_idx]
                                         tn = cm_matrix[no_idx_in_model, no_idx_in_model]
                                         fp = cm_matrix[no_idx_in_model, yes_idx_in_model]
                                         fn = cm_matrix[yes_idx_in_model, no_idx_in_model]
                                         tp = cm_matrix[yes_idx_in_model, yes_idx_in_model]

                                         st.write(f"**Verdadeiros Positivos (TP):** {tp}")
                                         st.write(f"**Verdadeiros Negativos (TN):** {tn}")
                                         st.write(f"**Falsos Positivos (FP):** {fp}")
                                         st.write(f"**Falsos Negativos (FN):** {fn}")
                                         st.info("TP: Previsto Passou, Real Passou | TN: Previsto Não Passou, Real Não Passou | FP: Previsto Passou, Real Não Passou | FN: Previsto Não Passou, Real Passou")
                                         st.warning("💡 No contexto de intervenção estudantil, Falsos Negativos (FN) são geralmente mais críticos, pois representam alunos que precisavam de ajuda mas não foram identificados.")
                                     else:
                                        st.warning("Não foi possível determinar os índices das classes 'no' e 'yes' nas classes do modelo para extrair TP/TN/FP/FN da Matriz de Confusão.")

                                 except IndexError:
                                      st.warning("Erro ao acessar elementos da Matriz de Confusão. As dimensões ou índices das classes podem estar incorretos.")
                                 except Exception as cm_extract_e:
                                      st.warning(f"Erro inesperado ao extrair TP/TN/FP/FN: {cm_extract_e}")


                            st.markdown("---")
                            st.markdown(f'<h3 class="sub-header">Análise Única do Algoritmo: {selected_model_name}</h3>', unsafe_allow_html=True)

                            # --- Adiciona visualizações/informações únicas por TIPO DE MODELO ---
                            # Assume processed_cols is globally available for feature names
                            feature_names_processed = X_train_processed.columns.tolist()


                            if selected_model_name == "Decision Tree":
                                st.write("#### Visualização da Árvore")
                                # Verifica se a árvore tem profundidade maior que 0 antes de tentar plotar
                                # get_depth() retorna None para árvores não treinadas ou com apenas um nó
                                if model_instance.get_depth() is not None and model_instance.get_depth() > 0:
                                    st.info(f"A visualização exibe a árvore até a profundidade **{model_params.get('max_depth', 'total')}** configurada ou até 6 níveis para clareza visual. Considere ajustar a profundidade máxima (max_depth) nas configurações acima.")
                                    # Determina a profundidade a plotar: a menor entre a profundidade real, a profundidade configurada pelo slider e um limite visual
                                    max_depth_from_params = model_params.get('max_depth')
                                    tree_actual_depth = model_instance.get_depth()
                                    max_visual_limit = 6 # Limite visual fixo recomendado para plot_tree

                                    # Calculate depth_to_plot ensuring we don't pass None to min
                                    depth_options = [tree_actual_depth if tree_actual_depth is not None else float('inf'),
                                                     max_depth_from_params if max_depth_from_params is not None else float('inf'),
                                                     max_visual_limit]
                                    depth_to_plot = int(min(depth_options))


                                    # Ajusta o tamanho da figura baseado na profundidade a plotar
                                    fig_width = max(20, len(feature_names_processed) * 1.5) # Ajusta largura com base no número de features? Pode ser muito largo.
                                    fig_width = min(fig_width, 40) # Limite máximo de largura para não ficar excessivo
                                    fig_height = max(8, depth_to_plot * 2.5) # Ajusta altura com base na profundidade


                                    fig_tree, ax_tree = plt.subplots(figsize=(fig_width, fig_height))

                                    try:
                                        plot_tree(model_instance, # Usar a instância treinada
                                                ax=ax_tree, # <-- Passar o Axes
                                                filled=True,
                                                feature_names=feature_names_processed,
                                                class_names=[str(c) for c in CLASS_NAMES],
                                                rounded=True,
                                                fontsize=8,
                                                max_depth=depth_to_plot, # Usar a profundidade calculada para plotar
                                                impurity=False, # Ocultar impureza para clareza
                                                node_ids=False, # Ocultar IDs dos nós
                                                proportion=True, # Mostrar proporção de amostras
                                                # rotate=True # Opcional: rotacionar para mais espaço horizontal
                                                )
                                        st.pyplot(fig_tree)
                                    except Exception as tree_e:
                                        st.error(f"❌ Não foi possível gerar a visualização da árvore: {tree_e}. A árvore pode ser muito complexa ou há um problema com as dependências (verifique matplotlib e Graphviz).")
                                    finally:
                                        plt.close(fig_tree)
                                else:
                                     st.info("A árvore de decisão treinada tem profundidade 0 (apenas um nó, ou seja, só faz uma previsão baseada na classe majoritária). Não há estrutura de árvore para visualizar.")


                            elif selected_model_name == "Random Forest":
                                st.write("#### Visualização de uma Árvore no Random Forest")
                                st.info(f"Random Forest é um ensemble de **{model_instance.n_estimators}** árvores. Aqui está a visualização da **primeira árvore** (`estimators_[0]`). Considere reduzir a profundidade máxima (max_depth) nas configurações acima para uma visualização mais clara.")
                                # Visualiza a primeira árvore na floresta
                                if hasattr(model_instance, 'estimators_') and len(model_instance.estimators_) > 0:
                                    estimator = model_instance.estimators_[0] # Primeira árvore

                                    # Determina a profundidade a plotar: similar à Decision Tree
                                    max_depth_from_params = model_params.get('max_depth')
                                    estimator_actual_depth = estimator.get_depth() if hasattr(estimator, 'get_depth') and estimator.get_depth() is not None else 1000
                                    max_visual_limit = 6

                                    depth_options_rf = [estimator_actual_depth,
                                                        max_depth_from_params if max_depth_from_params is not None else float('inf'),
                                                        max_visual_limit]
                                    depth_to_plot_rf = int(min(depth_options_rf))


                                    if depth_to_plot_rf > 0: # Só plotar se a árvore não for trivial
                                         # Ajusta o tamanho da figura baseado na profundidade a plotar
                                         fig_width_rf = max(20, len(feature_names_processed) * 1.5)
                                         fig_width_rf = min(fig_width_rf, 40) # Limite máximo
                                         fig_height_rf = max(8, depth_to_plot_rf * 2.5)

                                         fig_tree_rf, ax_tree_rf = plt.subplots(figsize=(fig_width_rf, fig_height_rf))
                                         try:
                                             plot_tree(estimator,
                                                     ax=ax_tree_rf, # <-- Passar o Axes
                                                     filled=True,
                                                     feature_names=feature_names_processed, # Usar nomes das features processadas
                                                     class_names=[str(c) for c in CLASS_NAMES],
                                                     rounded=True,
                                                     fontsize=8,
                                                     max_depth=depth_to_plot_rf,
                                                     impurity=False,
                                                     node_ids=False,
                                                     proportion=True
                                                     )
                                             st.pyplot(fig_tree_rf)
                                         except Exception as tree_e:
                                             st.error(f"❌ Não foi possível gerar a visualização da árvore: {tree_e}. A árvore pode ser muito complexa ou há um problema com as dependências.")
                                         finally:
                                              plt.close(fig_tree_rf)
                                    else:
                                        st.info("A primeira árvore do Random Forest tem profundidade 0 (apenas um nó). Não há estrutura de árvore para visualizar.")


                                    # Opcional: Mostrar Importância das Features do Random Forest como um todo
                                    if hasattr(model_instance, 'feature_importances_'):
                                        st.write("#### Importância das Características (Random Forest)")
                                        try:
                                            feature_importance_df_rf = pd.DataFrame({
                                                'Característica Processada': feature_names_processed, # Usar nomes das features processadas
                                                'Importância': model_instance.feature_importances_
                                            }).sort_values('Importância', ascending=False)

                                            fig_importance_rf = px.bar(
                                                feature_importance_df_rf.head(min(20, len(feature_importance_df_rf))), # Mostrar top 20 ou menos
                                                x='Importância',
                                                y='Característica Processada',
                                                orientation='h',
                                                color_continuous_scale=px.colors.sequential.Viridis, # Cores diferentes para distinção
                                                title="Importância Global das Características (Random Forest)"
                                            )
                                            fig_importance_rf.update_layout(yaxis={'categoryorder':'total ascending'})
                                            st.plotly_chart(fig_importance_rf, use_container_width=True)
                                            st.info("A importância das características em modelos de ensemble baseados em árvores é a soma das importâncias que cada característica contribui para a redução da impureza ou erro em todas as árvores do ensemble.")
                                        except Exception as imp_e:
                                             st.error(f"❌ Não foi possível exibir a importância das features do Random Forest: {imp_e}")
                                else:
                                     st.info(f"O modelo Random Forest ({selected_model_name}) não tem estimadores treinados acessíveis.")


                            elif selected_model_name in ["Regressão Logística", "SVM (Kernel Linear)"]: # Assume SVM com kernel linear se o tipo for 'SVC' e você o configura com kernel='linear' em AVAILABLE_MODELS_FOR_ANALYSIS
                                st.write("#### Coeficientes das Características")
                                # Para modelos lineares, coef_ mostra a influência de cada feature
                                if hasattr(model_instance, 'coef_'):
                                    try:
                                         # Assume coef_ é 1D para classificação binária, ou 2D (n_classes, n_features)
                                         coef_values = model_instance.coef_[0] if model_instance.coef_.ndim == 2 else model_instance.coef_
                                         # Usar feature_names_processed (colunas de X_train_processed)
                                         feature_coef_df = pd.DataFrame({
                                            'Característica Processada': feature_names_processed,
                                            'Coeficiente': coef_values
                                         }).sort_values('Coeficiente', ascending=False)

                                         coef_min = feature_coef_df['Coeficiente'].min()
                                         coef_max = feature_coef_df['Coeficiente'].max()
                                         # Adiciona um pequeno epsilon para evitar divisão por zero se todos os coeficientes forem 0
                                         abs_max = max(abs(coef_min), abs(coef_max)) if (coef_min is not None and coef_max is not None and (abs(coef_min) > 1e-9 or abs(coef_max) > 1e-9)) else 1.0

                                         fig_coef = px.bar(
                                             feature_coef_df.head(min(30, len(feature_coef_df))),
                                             x='Coeficiente',
                                             y='Característica Processada',
                                             orientation='h',
                                             color='Coeficiente',
                                             color_continuous_scale='RdBu',
                                             range_color=[-abs_max, abs_max] if abs_max > 1e-9 else [-1, 1],
                                             title=f"Coeficientes para {selected_model_name}"
                                         )
                                         fig_coef.update_layout(yaxis={'categoryorder':'total ascending'})
                                         st.plotly_chart(fig_coef, use_container_width=True)
                                         st.info("A magnitude do coeficiente indica a importância da característica; o sinal indica a direção da relação com a classe positiva (geralmente 1).")
                                    except Exception as coef_e:
                                         st.error(f"❌ Não foi possível visualizar os coeficientes: {coef_e}. Verifique a estrutura do objeto coef_ do modelo.")
                                else:
                                    st.info(f"Este modelo ({selected_model_name}) não tem coeficientes acessíveis ou não é linear.")


                            elif selected_model_name == "KNN":
                                st.write("#### Princípios do KNN")
                                st.info(f"""
                                O algoritmo K-Nearest Neighbors ({selected_model_name}) faz previsões baseando-se nos **{model_instance.n_neighbors}** vizinhos mais próximos da amostra no espaço de características.
                                *   **Como funciona:** Para classificar uma nova amostra, o KNN encontra as **{model_instance.n_neighbors}** amostras mais próximas nos dados de treino e atribui à nova amostra a classe mais comum entre esses vizinhos.
                                *   **Parâmetro chave:** `n_neighbors` (Número de Vizinhos), configurado aqui como {model_instance.n_neighbors}.
                                """)


                            elif selected_model_name in ["Gradient Boosting", "AdaBoost"]:
                                st.write(f"#### Princípios de Modelos de Boosting ({selected_model_name})")
                                st.info(f"""
                                Modelos de Boosting como {selected_model_name} constroem um ensemble de modelos fracos (geralmente árvores de decisão pequenas), aprendendo sequencialmente onde os modelos anteriores erraram.
                                *   **Gradient Boosting:** Constrói árvores sequencialmente, onde cada nova árvore tenta corrigir os erros residuais do ensemble anterior. O parâmetro chave `n_estimators` (número de árvores) e `learning_rate` (taxa de aprendizado) controlam o processo.
                                *   **AdaBoost:** Também constrói modelos sequencialmente, mas ajusta o peso das amostras, dando mais importância às que foram mal classificadas pelos modelos anteriores. O parâmetro chave `n_estimators` (número de estimadores base) e `learning_rate` controlam o processo.
                                Estes modelos são poderosos mas podem ser mais difíceis de interpretar diretamente do que uma única árvore ou modelo linear.
                                """)
                                # Opcional: Mostrar Importância das Features para Boosting
                                if hasattr(model_instance, 'feature_importances_'):
                                    st.write(f"#### Importância das Características ({selected_model_name})")
                                    try:
                                        feature_importance_df_boost = pd.DataFrame({
                                            'Característica Processada': feature_names_processed, # Usar nomes das features processadas
                                            'Importância': model_instance.feature_importances_
                                        }).sort_values('Importância', ascending=False)

                                        fig_importance_boost = px.bar(
                                            feature_importance_df_boost.head(min(20, len(feature_importance_df_boost))), # Mostrar top 20 ou menos
                                            x='Importância',
                                            y='Característica Processada',
                                            orientation='h',
                                            color_continuous_scale=px.colors.sequential.Viridis, # Cores diferentes para distinção
                                            title=f"Importância Global das Características ({selected_model_name})"
                                        )
                                        fig_importance_boost.update_layout(yaxis={'categoryorder':'total ascending'})
                                        st.plotly_chart(fig_importance_boost, use_container_width=True)
                                        st.info("A importância das características em modelos de ensemble baseados em árvores é a soma das importâncias que cada característica contribui para a redução da impureza ou erro em todas as árvores do ensemble.")
                                    except Exception as imp_e:
                                         st.error(f"❌ Não foi possível exibir a importância das features do {selected_model_name}: {imp_e}")
                                else:
                                     st.info(f"O modelo {selected_model_name} não suporta importância de features.")


                            elif selected_model_name == "SVM (Kernel RBF)": # RBF kernel não é linear, coef_ não se aplica diretamente
                                st.write("#### Princípios do SVM com Kernel RBF")
                                st.info("""
                                O Support Vector Machine (SVM) com Kernel RBF é um modelo não-linear que encontra um hiperplano de separação no espaço de características.
                                *   **Kernel RBF:** Permite mapear os dados para um espaço de dimensão mais alta onde a separação pode ser mais fácil, mesmo que não seja linear no espaço original. É eficaz para relações complexas.
                                *   **Parâmetros chaves:** `C` (penalidade do erro) e `gamma` (influência de um único exemplo de treino).
                                A interpretabilidade direta das "importâncias" das características não é tão simples quanto em modelos lineares ou baseados em árvores.
                                """)


                            else:
                                st.info(f"Não há análise única específica configurada para o algoritmo **{selected_model_name}** neste momento. Veja as métricas e a Matriz de Confusão acima para avaliar o seu desempenho.")

                        else: # Caso current_model_key não esteja em AVAILABLE_MODELS_FOR_ANALYSIS (não deveria acontecer)
                             st.error(f"Erro interno: Tipo de modelo '{current_model_key}' não reconhecido em AVAILABLE_MODELS_FOR_ANALYSIS.")


                    except Exception as e:
                         st.error(f"❌ Ocorreu um erro ao treinar ou avaliar o modelo {selected_model_name}: {e}")
                         st.warning("Verifique a compatibilidade entre o modelo e os dados processados, ou se há problemas com a instância do modelo selecionado. Erro detalhado: " + str(e))


        else: # Handle cases where target is missing in processed dataframes
             if TARGET_PROCESSED_NAME not in train_df_processed_global.columns:
                  st.error(f"A coluna alvo '{TARGET_PROCESSED_NAME}' não foi encontrada no dataframe de treino processado.")
             if TARGET_PROCESSED_NAME not in test_df_processed_global.columns:
                  st.error(f"A coluna alvo '{TARGET_PROCESSED_NAME}' não foi encontrada no dataframe de teste processado.")


    # REMOVIDOS os blocos elif para Matriz de Correlação, Matriz de Covariância e Matriz Personalizada
    # Eles não são necessários para comparar modelos.

# --- Documentação ---
elif menu == "Documentação":
    st.markdown('<h1 class="main-header">Documentação e Exemplos</h1>', unsafe_allow_html=True)
    st.markdown('<p class="info-text">Bem-vindo à secção de documentação...</p>', unsafe_allow_html=True)

    st.markdown('<h2 class="sub-header">Sobre o Dataset</h2>', unsafe_allow_html=True)
    st.markdown(f"""
    A aplicação utiliza o seu dataset original: **`student-data.csv`**. Este dataset contém informações sobre alunos...
    """)
    st.markdown('### Descrição das Características (Features)', unsafe_allow_html=True)
    st.markdown('<p class="info-text">Aqui está a lista completa das características do dataset e o seu significado:</p>', unsafe_allow_html=True)

    # Exibir a lista de features com suas descrições (usa o dicionário completo)
    for feature, desc in full_feature_descriptions.items():
        # Adiciona formatação condicional para a variável alvo
        if feature == TARGET_ORIGINAL_NAME:
             st.markdown(f"- **{feature.replace('_', ' ').title()}**: {desc} **(Variável Alvo)**")
        else:
             st.markdown(f"- **{feature.replace('_', ' ').title()}**: {desc}")


    st.markdown('<h2 class="sub-header">Sobre o Modelo de Previsão</h2>', unsafe_allow_html=True)
    st.markdown("""
    Um modelo de classificação binária foi treinado no dataset `student-data.csv` para prever se um aluno passará ou não...
    *   O **Pré-processador** (`preprocessor.joblib`) é responsável por transformar os dados brutos do aluno para o formato que o modelo entende.
    *   O **Modelo Treinado Principal** (`best_model.joblib`) é o resultado do processo de treino e otimização realizado no seu notebook e é usado para a Previsão Individual e secção de Análise.
    Pode obter previsões individuais na secção "Previsão Individual" e ver a avaliação detalhada deste modelo principal no conjunto de teste na secção "Análise do Modelo Treinado".
    """)

    st.markdown('<h2 class="sub-header">Sobre a Análise de Matriz</h2>', unsafe_allow_html=True)
    st.markdown("""
    A secção "Análise de Matriz" permite visualizar e analisar propriedades matemáticas...
    *   **Matriz de Confusão (Escolher Modelo):** Permite selecionar diferentes tipos de modelos para visualizar o seu desempenho *temporário* no conjunto de teste processado. Útil para comparar o desempenho de diferentes algoritmos.
    *   **Matriz de Correlação (Seu Dataset):** Mostra a correlação linear entre pares de variáveis numéricas no seu dataset original.
    *   **Matriz de Covariância (Seu Dataset):** Semelhante à correlação, mas dependente da escala...
    *   **Matriz Personalizada:** Permite introduzir qualquer matriz quadrada...
    """)

    st.markdown('<h2 class="sub-header">Próximos Passos e Melhorias</h2>', unsafe_allow_html=True)
    st.markdown("""
    Pode considerar as seguintes melhorias...
    """)


# --- Footer ---
st.markdown("---")
st.markdown("© 2025 Sistema de Intervenção Estudantil. Desenvolvido com Streamlit.")
