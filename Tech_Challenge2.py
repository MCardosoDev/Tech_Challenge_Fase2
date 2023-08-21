#%%
import pandas as pd
import numpy as np
import streamlit as st
from statsmodels.tsa.stattools import adfuller
from Utils import (
    convert_volume,
    plot_ts,
    decompose,
    test_adfuller,
    data_diff,
    test_diff_adfuller,
    mape_error,
    wmape_error,
    smape_error,
    acf_pacf,
    plot_error
)
#%%
df = pd.read_csv('df.csv')
columns_to_print = ['Data', 'Último', 'Abertura', 'Máxima', 'Mínima', 'Vol.', 'Var%']
df_erros = pd.read_csv('Data/Errors.csv')
df_erros_prophet = pd.read_csv('Data/ErrorsProphet.csv')
df_mesclado = pd.concat([df_erros_prophet, df_erros], ignore_index=True)
df_mesclado.to_csv('erros.csv', index=False)
#%%
def test_adfuller_streamlit(df):
    result = adfuller(df.values)
    st.write('Teste ADF')
    st.write(f'Teste estatístico: {result[0]}')
    st.write(f'Valor p: {result[1]}')
    st.write('Valores críticos:')
    for key, value in result[4].items(): #type: ignore
        st.write(f'\t{key}: {value}')
#%%
def test_diff_adfuller_streamlit(df):
    df_log = np.log(df[['Último', 'Abertura', 'Máxima', 'Mínima', 'Vol.', 'y']])
    df_log_meam = df_log.rolling(12).mean() # type: ignore
    df_log = (df_log - df_log_meam).dropna()
    df_diff = df_log.diff(1).dropna()# primeira derivada
    df_ultimo_diff = df_diff['y']
    result = adfuller(df_ultimo_diff.dropna().values)
    st.write('Teste ADf')
    st.write(f'Teste estatistico: {result[0]}')
    st.write(f'P-Value: {result[1]}')
    st.write(f'Valores criticos:')
    for key, value in result[4].items(): #type: ignore
        st.write(f'\t{key}: {value}')
#%%
mape_formula = r'MAPE = \frac{1}{n} \sum_{i=1}^{n} \left| \frac{y_{\text{true},i} - y_{\text{pred},i}}{y_{\text{true},i}} \right| \times 100'
wmape_formula = r'WMAPE = \frac{\sum_{i=1}^{n} \left| y_{\text{true},i} - y_{\text{pred},i} \right|}{\sum_{i=1}^{n} \left| y_{\text{true},i} \right|} \times 100'
smape_formula = r'SMAPE = \frac{100}{n} \sum_{i=1}^{n} \frac{\left| y_{\text{pred},i} - y_{\text{true},i} \right|}{(|y_{\text{pred},i}| + |y_{\text{true},i}|)/2}'
mape_explanation = "O MAPE (Mean Absolute Percentage Error) calcula a média das porcentagens absolutas dos erros entre as previsões e os valores verdadeiros."
wmape_explanation = "O WMAPE (Weighted Mean Absolute Percentage Error) é uma variação do MAPE que pondera os erros pela magnitude dos valores verdadeiros."
smape_explanation = "O SMAPE (Symmetric Mean Absolute Percentage Error) é uma métrica que considera a simetria dos erros, levando em conta a magnitude dos valores verdadeiros e das previsões."
#%%

def main():
    st.set_page_config(layout="wide")
    st.title('Análise de fechamento diário da IBOVESPA')
    tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs(['Geral', 'Dados Históricos - IBOVESPA', 'Diferenciação da Série', 'ACF e PCF', 'Avaliação', 'Melhor Modelo'])

    with tab0:
        '''
        ## Análise dos Dados para Prever Diariamente o Fechamento da base

        Esta análise tem como objetivo utilizar os dados disponíveis sobre as informações da bolsa de valores, 
        realizar um modelo preditivo, criar uma serie temporal e prever diariamente o fechamento da base.
        
        ### Dados Utilizados para análise

        **Análise dos dados históricos da IBOVESPA** - Os dados sobre o histórico da bolsa foram obtidos do investing.

        - Link: [Dados Históricos - Ibovespa](https://br.investing.com/indices/bovespa-historical-data)
        '''
        st.write("\n")
        st.markdown('##### Dados Históricos - Ibovespa')
        st.dataframe(df[columns_to_print].head(10).reset_index(drop=True), use_container_width=True)
        st.markdown(
            '<span class="small-font">Dados utilizados de um período entre 2003 até 2023</span>',
        unsafe_allow_html=True)   
        '''
        ### Diferenciação da série para conversão em uma série estacionaria.
        
        A diferenciação é usada para transformar uma série temporal não estacionária em uma série estacionária. Para **modelos como o ARIMA**, **Remoção de Tendências**, **Estabilização da Variância** e **Eliminação de Sazonalidade**.

        ### Autocorrelação (ACF) e Autocorrelação parcial (PACF).
        
        **Função de autocorrelação (ACF) e Função de Autocorrelação parcial (PACF)**. A ACF mede a correlação entre os valores passados e os valores presentes de uma série temporal, levando em consideração todos os atrasos possíveis e identifica padrões de correlação entre observações em diferentes pontos temporais.
        A PACF mede a correlação direta entre dois pontos temporais, controlando os efeitos dos atrasos intermediários.

        ### Teste e Avaliação de modelos preditivos

        **Avaliação dos de modelos para series temporais** - Para encontrar o melhor modelo preditivo utiliza-se testes de erros(MAPE, wMAPE, sMAPE) para avaliar a precisão de acerto de cada modelo.

        ### Melhor Modelo

        **Escolha do melhor modelo** - Com avaliação do erros cometidos é possível escolher o modelo com melhor desempenho.

        '''
    with tab1:
        '''
        ## Análise Exploratória de Dados (EDA)
        '''
        st.plotly_chart(plot_ts(df), use_container_width = True)
        st.plotly_chart(decompose(df, 'Decomposição da Série Temporal - Fechamento da base'), use_container_width = True)
        '''
        #### Teste Augmented Dickey Fuller
        Teste estatístico usado para determinar se uma série temporal possui raiz unitária, o que indica a presença de não-estacionariedade na série
        '''
        st.write("\n")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("  Hipóteses do Teste ADF:")
            st.write("  - H0: A série não é estacionária.")
            st.write("  - H1: A série é estacionária.")
            if st.button('Executar Teste ADF'):
                with col2:
                    test_adfuller_streamlit(df['Último'])
                with col3:
                    st.write("Resultado")
                    st.write("---")
                    st.markdown("<span style='color:red'>P-value 49% e teste estatístico maior que os valores criticos: A série não é estacionaria</span>", unsafe_allow_html=True)
    with tab2:
        '''
        ## Diferenciação da Série Temporal
            - Remover tendencia e sazonalidade, fazer a aproximação(transformada logaritmo e subtrair da média móvel)
            - Média móvel aplicada a linha da tendencia
            - Janela móvel de tamanho 12 meses é aplicada sobre os dados
            - Aplicar log
            - Subtrair a média móvel
            - Derivadas de um número de polinômio de primeiro grau deixando mais estacionaria.
        '''
        st.plotly_chart(data_diff(df), use_container_width = True)
        '''
        #### Novo Teste Augmented Dickey Fuller
        '''
        st.write("\n")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("  Hipóteses do Teste ADF:")
            st.write("  - H0: A série não é estacionária.")
            st.write("  - H1: A série é estacionária.")
            if st.button('Executar Novo Teste ADF'):
                with col2:
                    test_adfuller_streamlit(df['Último'])
                with col3:
                    st.write("Resultado")
                    st.write("---")
                    st.markdown("<span style='color:green'>P-value 0.0% e teste estatístico menor que os valores criticos: A série é estacionaria após a diferenciação</span>", unsafe_allow_html=True)

    with tab3:
        '''
        ## Autocorrelação (ACF) e Autocorrelação parcial (PACF)
        - 5% ACF (intervalo de confiança).
        - 1.96/sqrt(N-d)
            - **N** número de pontos e **d** é o número de vezes que os dados foram diferenciamos (intervalo de confiança para estimativas de autocorrelação significativa).
        '''
        st.plotly_chart(acf_pacf(df), use_container_width = True)
        st.write("  Ordem de diferenciação **D** = 1 (Foi necessária 1 diferenciação para tornar a série estacionaria)")
        st.write("  **Q acf** = 0.915")
        st.write("  **P pacf** = 0.915")
    with tab4:
        '''
        # Avaliação dos modelos
        '''
        st.title("Métricas de Erro em Previsões")
        st.subheader("MAPE - Mean Absolute Percentage Error")
        st.latex(mape_formula)
        st.write(mape_explanation)
        st.subheader("WMAPE - Weighted Mean Absolute Percentage Error")
        st.latex(wmape_formula)
        st.write(wmape_explanation)
        st.subheader("SMAPE - Symmetric Mean Absolute Percentage Error")
        st.latex(smape_formula)
        st.write(smape_explanation)
        st.plotly_chart(plot_error(df_mesclado, 'mape', 'MAPE', 'MAPE (Mean Absolute Percentage Error)'), use_container_width = True)
        st.plotly_chart(plot_error(df_mesclado, 'wmape', 'wMAPE', 'wMAPE (Weighted Mean Absolute Percentage Error)'), use_container_width = True)
        st.plotly_chart(plot_error(df_mesclado, 'smape', 'sMAPE', 'sMAPE (Weighted Mean Absolute Percentage Error)'), use_container_width = True)
    with tab5:
        '''
        ## Análise dos Dados
        '''

if __name__ == "__main__":
    main()
# %%