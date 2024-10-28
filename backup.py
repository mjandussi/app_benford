import streamlit as st
import collections
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


######## CONFIG ###########

st.set_page_config(
    page_title="App Lei Benford",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


######## FUNÇÕES ###########

def benford_law_distribution():
    return [math.log10(1 + 1/d) for d in range(1, 10)]

def calculate_first_digit_distribution(data):
    digit_count = collections.Counter()

    for number in data:
        first_digit = str(number)[0]
        digit_count[first_digit] += 1

    total_numbers = sum(digit_count.values())
    digit_distribution = {digit: count / total_numbers for digit, count in digit_count.items()}

    return digit_distribution


######## PÁGINAS ###########

def pagina_inicial():

    st.header("Seja bem vindo !!")
    st.header("Análises de Empenhos pela Lei de Benford")
    texto_com_cor = """
    <p>Esta aplicação web oferece análises detalhadas de empenhos utilizando a <span style="color:#cc255f;"> Lei de Benford</span>. 
    Com sua eficiência, permite que auditores e contadores apliquem essa metodologia em dados de empenhos em questão de minutos, 
    proporcionando uma ferramenta valiosa para a detecção de anomalias e a garantia da integridade dos dados financeiros.</p>
    """
    st.markdown(texto_com_cor, unsafe_allow_html=True)




def analises():   

    st.header("Análises")

    ### Coluna para os Arquivos
    st.write('Upload de Arquivos')
    upload_file1 = st.file_uploader('Lista de Empenhos da UG')

    if upload_file1 is None:
        st.write('CARREGUE O ARQUIVO')

    else:

        df = pd.read_excel(upload_file1, header=3)
        df = df.iloc[:-3]
        # Filtrando as linhas que não terminam com 'P'
        df = df[~df['Número'].str.endswith('P')]
        # Convertendo para o tipo 'object'
        df[['Fonte', 'Natureza']] = df[['Fonte', 'Natureza']].astype('object')
        # Removendo o sufixo '.0' ao converter para string
        df['Fonte'] = df['Fonte'].astype(str).str.replace(r'\.0$', '', regex=True)
        df['Natureza'] = df['Natureza'].astype(str).str.replace(r'\.0$', '', regex=True)
        with st.expander("Expandir Dataframe"):
            st.write(df)

        # Perguntar ao usuário se deseja modificar o DataFrame
        modificar_df = st.checkbox("DESEJA FILTRAR POR ND?")
    
        if modificar_df:
            # Obter as opções únicas de "Natureza"
            naturezas_unicas = df['Natureza'].unique()
            
            # Permitir que o usuário selecione as "Naturezas" que deseja analisar
            naturezas_selecionadas = st.multiselect("Selecione as Naturezas para filtrar", options=naturezas_unicas)
            
            # Filtrar o DataFrame com base nas "Naturezas" selecionadas
            if naturezas_selecionadas:
                df = df[df['Natureza'].isin(naturezas_selecionadas)]
                st.write("DataFrame Filtrado:")
    
                #############################################################################################################################3

                # Passando a coluna Valor como uma lista
                distribution = calculate_first_digit_distribution(df['Valor'])
                benford_distribution = benford_law_distribution()

                # Criar três colunas
                col1, col2, col3 = st.columns(3)

                # Tabela de distribuição observada
                with col1:
                    st.subheader("Distribuição Observada nos Dados")
                    observed_data = {"Dígito": [], "Distribuição Observada": []}
                    for digit in range(1, 10):
                        observed_data["Dígito"].append(digit)
                        observed_data["Distribuição Observada"].append(f"{distribution.get(str(digit), 0):.2%}")
                    observed_df = pd.DataFrame(observed_data)
                    st.table(observed_df)

                # Tabela de distribuição esperada pela Lei de Benford
                with col2:
                    st.subheader("Distribuição Esperada pela Lei de Benford")
                    expected_data = {"Dígito": [], "Distribuição Esperada": []}
                    for digit, expected in enumerate(benford_distribution, start=1):
                        expected_data["Dígito"].append(digit)
                        expected_data["Distribuição Esperada"].append(f"{expected:.2%}")
                    expected_df = pd.DataFrame(expected_data)
                    st.table(expected_df)

                # Tabela de diferenças
                with col3:
                    st.subheader("Diferença entre Distribuições")
                    differences_data = {"Dígito": [], "Diferença": []}
                    differences = {digit: distribution.get(str(digit), 0) - benford_distribution[digit - 1] for digit in range(1, 10)}
                    for digit in range(1, 10):
                        differences_data["Dígito"].append(digit)
                        differences_data["Diferença"].append(f"{differences[digit]:+.2%}")
                    differences_df = pd.DataFrame(differences_data)
                    st.table(differences_df)

                st.divider()

                ###################################################################################################################33
                # Preparando os dados para o gráfico
                digits = list(range(1, 10))
                distribution_values = [distribution.get(str(digit), 0) for digit in digits]
                benford_values = benford_distribution

                # Criando o gráfico
                plt.figure(figsize=(10, 6))
                bar_width = 0.35
                index = range(len(digits))

                # Barras para a distribuição dos dados
                plt.bar(index, distribution_values, bar_width, label='Dados', alpha=0.7, color='b')

                # Linha para a distribuição esperada pela Lei de Benford
                plt.plot(index, benford_values, label='Benford', color='r', marker='o', linestyle='--')

                # Configurações do gráfico
                plt.xlabel('Dígitos')
                plt.ylabel('Frequência')
                plt.title('Comparação da Distribuição dos Dígitos Iniciais')
                plt.xticks(index, digits)
                plt.legend()

                # Ajuste do layout
                plt.tight_layout()

                # Exibindo o gráfico no Streamlit
                st.pyplot(plt)

                st.divider()

                ################################################################################################3

                corte = st.selectbox("Escolha o corte para as repetições", [5, 10, 15, 20])

                st.write('### Contagem por tipo')
                # Criar colunas
                col1, col2, col3 = st.columns(3)

                with col1:
                    with st.expander(f'{corte} - Valores mais repetidos'):
                        contagem_valores = df['Valor'].value_counts().reset_index().sort_values(by="count", ascending=False)
                        st.write(f"Contagem dos {corte} Valores Mais Repetidos")
                        st.write(contagem_valores.head(corte))

                with col2:
                    with st.expander(f'{corte} - Credores mais repetidos'):
                        contagem_valores = df['Nome do Credor'].value_counts().reset_index().sort_values(by="count", ascending=False)
                        st.write(f"Contagem dos {corte} Credores Mais Repetidos")
                        st.write(contagem_valores.head(corte))
                
                with col3:
                    with st.expander(f'{corte} - NDs mais repetidos'):
                        contagem_valores = df['Natureza'].value_counts().reset_index().sort_values(by="count", ascending=False)
                        st.write(f"Contagem dos {corte} NDs Mais Repetidas")
                        st.write(contagem_valores.head(corte))

            
                st.divider()

                st.write('### Análise Por Totais')
                # Criar colunas
                col1, col2 = st.columns([3,2])

                # Tabela de distribuição observada
                with col1:
                    with st.expander('Soma dos Valores por Credor'):
                        soma_credor = df.groupby('Nome do Credor')['Valor'].sum().reset_index().sort_values(by="Valor", ascending=False)
                        st.write("### Soma dos Valores por Credor")
                        st.write(soma_credor)

                # Tabela de distribuição esperada pela Lei de Benford
                with col2:
                    with st.expander('Soma dos Valores por Natureza'):
                        soma_nd = df.groupby('Natureza')['Valor'].sum().reset_index().sort_values(by="Valor", ascending=False)
                        st.write("### Soma dos Valores por Natureza")
                        st.write(soma_nd)

                st.divider()
                
                ################################################################################################################################
                # Criando uma coluna cópia
                df['valores'] = df['Valor']
                # Converter os valores float para string
                df['valores_str'] = df['valores'].astype(str)
                df['valores_str'] = df['valores_str'].astype(str).str.replace(r'\.0$', '', regex=True)

                ########################################################################################################################333333
                # Criar um selectbox para escolher o dígito inicial
                st.write('### Análise da Distribuição dos Valores por Dígito')
                digito_inicial = st.selectbox('Escolha o dígito inicial para análise', [str(i) for i in range(1, 10)])

                # Filtrar valores que começam com o dígito selecionado
                filtro_digito = df['valores_str'].str.startswith(digito_inicial)
                valores_comecam_com_digito = df[filtro_digito]

                valores_digito = valores_comecam_com_digito['Valor']

                # Definir intervalos dinamicamente com base no dígito inicial
                base = int(digito_inicial)
                intervalos = [(base * 10**i, (base + 1) * 10**i) for i in range(7)]  # Ajustar para incluir o valor final

                # Criar histogramas separados para cada segmento
                fig, axs = plt.subplots(len(intervalos) - 1, 1, figsize=(8, 4 * (len(intervalos) - 1)))

                for i, (inicio, fim) in enumerate(intervalos):
                    if inicio == 1 and fim == 10:
                        continue  # Pular o intervalo de 1 a 9

                    valores_intervalo = [valor for valor in valores_digito if inicio <= valor < fim]
                    # Calcular o passo do range, garantindo que não seja zero
                    step = max((fim - inicio) // 10, 1)
                    if valores_intervalo:  # Verificar se há valores para plotar
                        axs[i - 1].hist(valores_intervalo, bins=range(int(inicio), int(fim) + 1, step), color='skyblue', edgecolor='black', alpha=0.7)
                        axs[i - 1].set_title(f'Valores entre {inicio} e {fim - 1}')
                        axs[i - 1].set(xlabel='Valor do Empenho', ylabel='Frequência')
                        axs[i - 1].yaxis.get_major_locator().set_params(integer=True)

                plt.tight_layout()

                # Exibir o gráfico no Streamlit
                st.pyplot(fig)

        else:
            # Passando a coluna Valor como uma lista
            distribution = calculate_first_digit_distribution(df['Valor'])
            benford_distribution = benford_law_distribution()

            # Criar três colunas
            col1, col2, col3 = st.columns(3)

            # Tabela de distribuição observada
            with col1:
                st.subheader("Distribuição Observada nos Dados")
                observed_data = {"Dígito": [], "Distribuição Observada": []}
                for digit in range(1, 10):
                    observed_data["Dígito"].append(digit)
                    observed_data["Distribuição Observada"].append(f"{distribution.get(str(digit), 0):.2%}")
                observed_df = pd.DataFrame(observed_data)
                st.table(observed_df)

            # Tabela de distribuição esperada pela Lei de Benford
            with col2:
                st.subheader("Distribuição Esperada pela Lei de Benford")
                expected_data = {"Dígito": [], "Distribuição Esperada": []}
                for digit, expected in enumerate(benford_distribution, start=1):
                    expected_data["Dígito"].append(digit)
                    expected_data["Distribuição Esperada"].append(f"{expected:.2%}")
                expected_df = pd.DataFrame(expected_data)
                st.table(expected_df)

            # Tabela de diferenças
            with col3:
                st.subheader("Diferença entre Distribuições")
                differences_data = {"Dígito": [], "Diferença": []}
                differences = {digit: distribution.get(str(digit), 0) - benford_distribution[digit - 1] for digit in range(1, 10)}
                for digit in range(1, 10):
                    differences_data["Dígito"].append(digit)
                    differences_data["Diferença"].append(f"{differences[digit]:+.2%}")
                differences_df = pd.DataFrame(differences_data)
                st.table(differences_df)

            st.divider()

            ###################################################################################################################33
            # Preparando os dados para o gráfico
            digits = list(range(1, 10))
            distribution_values = [distribution.get(str(digit), 0) for digit in digits]
            benford_values = benford_distribution

            # Criando o gráfico
            plt.figure(figsize=(10, 6))
            bar_width = 0.35
            index = range(len(digits))

            # Barras para a distribuição dos dados
            plt.bar(index, distribution_values, bar_width, label='Dados', alpha=0.7, color='b')

            # Linha para a distribuição esperada pela Lei de Benford
            plt.plot(index, benford_values, label='Benford', color='r', marker='o', linestyle='--')

            # Configurações do gráfico
            plt.xlabel('Dígitos')
            plt.ylabel('Frequência')
            plt.title('Comparação da Distribuição dos Dígitos Iniciais')
            plt.xticks(index, digits)
            plt.legend()

            # Ajuste do layout
            plt.tight_layout()

            # Exibindo o gráfico no Streamlit
            st.pyplot(plt)

            st.divider()

            ################################################################################################3

            corte = st.selectbox("Escolha o corte para as repetições", [5, 10, 15, 20])

            st.write('### Contagem por tipo')
            # Criar colunas
            col1, col2, col3 = st.columns(3)

            with col1:
                with st.expander(f'{corte} - Valores mais repetidos'):
                    contagem_valores = df['Valor'].value_counts().reset_index().sort_values(by="count", ascending=False)
                    st.write(f"Contagem dos {corte} Valores Mais Repetidos")
                    st.write(contagem_valores.head(corte))

            with col2:
                with st.expander(f'{corte} - Credores mais repetidos'):
                    contagem_valores = df['Nome do Credor'].value_counts().reset_index().sort_values(by="count", ascending=False)
                    st.write(f"Contagem dos {corte} Credores Mais Repetidos")
                    st.write(contagem_valores.head(corte))
            
            with col3:
                with st.expander(f'{corte} - NDs mais repetidos'):
                    contagem_valores = df['Natureza'].value_counts().reset_index().sort_values(by="count", ascending=False)
                    st.write(f"Contagem dos {corte} NDs Mais Repetidas")
                    st.write(contagem_valores.head(corte))

        
            st.divider()

            st.write('### Análise Por Totais')
            # Criar colunas
            col1, col2 = st.columns([3,2])

            # Tabela de distribuição observada
            with col1:
                with st.expander('Soma dos Valores por Credor'):
                    soma_credor = df.groupby('Nome do Credor')['Valor'].sum().reset_index().sort_values(by="Valor", ascending=False)
                    st.write("### Soma dos Valores por Credor")
                    st.write(soma_credor)

            # Tabela de distribuição esperada pela Lei de Benford
            with col2:
                with st.expander('Soma dos Valores por Natureza'):
                    soma_nd = df.groupby('Natureza')['Valor'].sum().reset_index().sort_values(by="Valor", ascending=False)
                    st.write("### Soma dos Valores por Natureza")
                    st.write(soma_nd)

            st.divider()
            
            ################################################################################################################################
            # Criando uma coluna cópia
            df['valores'] = df['Valor']
            # Converter os valores float para string
            df['valores_str'] = df['valores'].astype(str)
            df['valores_str'] = df['valores_str'].astype(str).str.replace(r'\.0$', '', regex=True)

            ########################################################################################################################333333
            # Criar um selectbox para escolher o dígito inicial
            st.write('### Análise da Distribuição dos Valores por Dígito')
            digito_inicial = st.selectbox('Escolha o dígito inicial para análise', [str(i) for i in range(1, 10)])

            # Filtrar valores que começam com o dígito selecionado
            filtro_digito = df['valores_str'].str.startswith(digito_inicial)
            valores_comecam_com_digito = df[filtro_digito]

            valores_digito = valores_comecam_com_digito['Valor']

            # Definir intervalos dinamicamente com base no dígito inicial
            base = int(digito_inicial)
            intervalos = [(base * 10**i, (base + 1) * 10**i) for i in range(7)]  # Ajustar para incluir o valor final

            # Criar histogramas separados para cada segmento
            fig, axs = plt.subplots(len(intervalos) - 1, 1, figsize=(8, 4 * (len(intervalos) - 1)))

            for i, (inicio, fim) in enumerate(intervalos):
                if inicio == 1 and fim == 10:
                    continue  # Pular o intervalo de 1 a 9

                valores_intervalo = [valor for valor in valores_digito if inicio <= valor < fim]
                # Calcular o passo do range, garantindo que não seja zero
                step = max((fim - inicio) // 10, 1)
                if valores_intervalo:  # Verificar se há valores para plotar
                    axs[i - 1].hist(valores_intervalo, bins=range(int(inicio), int(fim) + 1, step), color='skyblue', edgecolor='black', alpha=0.7)
                    axs[i - 1].set_title(f'Valores entre {inicio} e {fim - 1}')
                    axs[i - 1].set(xlabel='Valor do Empenho', ylabel='Frequência')
                    axs[i - 1].yaxis.get_major_locator().set_params(integer=True)

            plt.tight_layout()

            # Exibir o gráfico no Streamlit
            st.pyplot(fig)




# Menu de navegação na barra lateral
st.sidebar.title("Menu de Navegação")
opcao = st.sidebar.radio("Escolha a análise desejada:", ("Página Inicial", "Análise Empenhos"))

# Chama a função correspondente à opção selecionada
if opcao == "Página Inicial":
    pagina_inicial()
elif opcao == "Análise Empenhos":
    analises()
