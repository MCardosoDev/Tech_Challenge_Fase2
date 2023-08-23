#%%
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots as sp
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from Utils import (
    plot_ts,
    decompose,
    data_diff,
    acf_pacf,
    plot_error
)
from statsforecast import StatsForecast
from statsforecast.models import SeasonalExponentialSmoothingOptimized
#%%
df = pd.read_csv('df.csv')
df_st = df.copy()
df_st['Data'] = pd.to_datetime(df_st['Data'], format='%Y-%m-%d')
df_st['ds'] = pd.to_datetime(df_st['ds'], format='%Y-%m-%d')
df_st.set_index('Data', inplace=True)
#%%
train = pd.read_csv('train.csv')
train['ds'] = pd.to_datetime(train['ds'])
test = pd.read_csv('test.csv')
test['ds'] = pd.to_datetime(test['ds'])
h = test.index.nunique()
#%%
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
model = StatsForecast(models=[SeasonalExponentialSmoothingOptimized(season_length=h)], freq='D', n_jobs=-1)
model.fit(train)
forecast = model.predict(h=h, level=[95])
forecast = forecast.reset_index().merge(test, on=['ds', 'unique_id'], how='left')
forecast.dropna(inplace=True)
#%%
def main():
    st.set_page_config(layout="wide")
    st.title('Análise de fechamento diário da IBOVESPA')
    tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(['Geral', 'Dados Históricos - IBOVESPA', 'Diferenciação da Série', 'ACF e PCF', 'Avaliação', 'Melhores Modelos', 'Melhor Modelo'])

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

        A decomposição da série temporal é empregada para desmembrar os dados em componentes distintos, como tendência, sazonalidade e resíduos, permitindo uma compreensão mais profunda dos padrões subjacentes e variações presentes ao longo do tempo.
        Essa abordagem facilita a análise individual de cada componente, auxiliando na identificação de informações relevantes para modelagem e previsão.
        '''
        st.plotly_chart(plot_ts(df_st), use_container_width = True)
        st.plotly_chart(decompose(df_st, 'Decomposição da Série Temporal - Fechamento da base'), use_container_width = True)
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
                    test_adfuller_streamlit(df_st['Último'])
                with col3:
                    st.write("Resultado")
                    st.write("---")
                    st.markdown("<span style='color:red'>P-value 49% e teste estatístico maior que os valores criticos: A série não é estacionaria</span>", unsafe_allow_html=True)
    with tab2:
        '''
        ## Diferenciação da Série Temporal

        A diferenciação de uma série não estacionária é crucial para transformá-la em uma série estacionária, permitindo a aplicação de técnicas estatísticas e modelos de previsão.
        Isso é fundamental para compreender padrões temporais e obter previsões mais precisas.

            - Remover tendência e sazonalidade, fazer a aproximação(transformada logaritmo e subtrair da média móvel)
            - Média móvel aplicada a linha da tendencia
            - Janela móvel de tamanho 12 meses é aplicada sobre os dados
            - Aplicar log
            - Subtrair a média móvel
            - Derivadas de um número de polinômio de primeiro grau deixando mais estacionaria.
        '''
        st.plotly_chart(data_diff(df_st), use_container_width = True)
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
                    test_adfuller_streamlit(df_st['Último'])
                with col3:
                    st.write("Resultado")
                    st.write("---")
                    st.markdown("<span style='color:green'>P-value 0.0% e teste estatístico menor que os valores criticos: A série é estacionaria após a diferenciação</span>", unsafe_allow_html=True)

    with tab3:
        '''
        ## Autocorrelação (ACF) e Autocorrelação parcial (PACF)

        Os gráficos de autocorrelação (ACF) e autocorrelação parcial (PACF) são ferramentas essenciais na análise de séries temporais.
        ACF revela correlações em diferentes defasagens, enquanto PACF ajuda a identificar correlações diretas e eliminação de defasagens irrelevantes em modelos de previsão.
        Ao observar onde essas correlações se tornam não significativas nos gráficos PACF e ACF, você pode obter sugestões para os parâmetros Q (ordem do termo MA) e P (ordem do termo AR) para modelos como o ARMA,
        ARIMA ou SARIMA, os quais ajudam a modelar adequadamente a série temporal.

        - 5% ACF (intervalo de confiança).
        - 1.96/sqrt(N-d)
            - **N** número de pontos e **d** é o número de vezes que os dados foram diferenciamos (intervalo de confiança para estimativas de autocorrelação significativa).
        '''
        st.plotly_chart(acf_pacf(df_st), use_container_width = True)
        st.write("  Ordem de diferenciação **D** = 1 (Foi necessária 1 diferenciação para tornar a série estacionaria)")
        st.write("  **Q acf** = 0.915")
        st.write("  **P pacf** = 0.915")

    with tab4:
        '''
        ## Avaliação dos modelos - Métricas de Erro em Previsões
        '''
        st.subheader("MAPE - Mean Absolute Percentage Error")
        st.latex(mape_formula)
        st.write(mape_explanation)
        st.subheader("WMAPE - Weighted Mean Absolute Percentage Error")
        st.latex(wmape_formula)
        st.write(wmape_explanation)
        st.subheader("SMAPE - Symmetric Mean Absolute Percentage Error")
        st.latex(smape_formula)
        st.write(smape_explanation)
        '''
        É evidente uma consistência na ordem dos modelos avaliados em relação aos erros (MAPE, WMAPE, SMAPE), considerando a mesma base de dados do desafio.
        Essa coerência simplifica a identificação dos modelos com melhor desempenho, oferecendo uma maneira clara de determinar suas eficácias na análise dos dados fornecidos.
        '''
        st.plotly_chart(plot_error(df_mesclado, 'mape', 'MAPE', 'MAPE (Mean Absolute Percentage Error)'), use_container_width = True)
        st.plotly_chart(plot_error(df_mesclado, 'wmape', 'wMAPE', 'wMAPE (Weighted Mean Absolute Percentage Error)'), use_container_width = True)
        st.plotly_chart(plot_error(df_mesclado, 'smape', 'sMAPE', 'sMAPE (Weighted Mean Absolute Percentage Error)'), use_container_width = True)

    with tab5:
        '''
        ## Explicação de Modelos de Previsão de Séries Temporais

        Modelos de séries temporais são estruturas analíticas que visam compreender e prever padrões em dados sequenciais ao longo do tempo, permitindo a captura de tendências, sazonalidades e variações temporais subjacentes.
        Esses modelos são fundamentais para a tomada de decisões.
        '''
        intro_text = (
            "Este aplicativo apresenta explicações dos seguintes modelos de previsão de séries temporais que são amplamente usados para avaliação:"
            "\n\n1. Múltiplas modelos da `statsforecast`."
            "\n2. SARIMAX do módulo `statsmodels`."
            "\n3. Rede Neural LSTM (Long Short-Term Memory) `tensorflow`."
            "\n4. Prophet, uma biblioteca de previsão desenvolvida pelo Facebook."
        )
        model_explanations = {
            "Window Average (Média Móvel Simples)":
                "O método da Média Móvel Simples calcula a média dos valores em uma janela móvel de tamanho fixo ao longo da série temporal. "
                "Ele suaviza flutuações de curto prazo, revelando tendências subjacentes.",

            "Holt-Winters":
                "O método de Holt-Winters é uma técnica de suavização exponencial tripla que considera três componentes: nível, tendência e sazonalidade. "
                "Ele modela séries temporais com tendência e padrões sazonais.",

            "TSB (Triple Seasonal Box-Jenkins)":
                "O TSB é uma extensão do método ARIMA que incorpora múltiplos componentes sazonais para modelar séries temporais com sazonalidades complexas.",

            "AutoTheta":
                "O AutoTheta é uma versão automatizada do método Theta, que é uma técnica de suavização exponencial dupla para séries temporais com sazonalidade aditiva.",

            "CES (Croston's Exponential Smoothing)":
                "O método CES é usado para prever a demanda de itens esporádicos ou intermitentes, considerando componentes de nível e probabilidade de demanda.",

            "AutoARIMA":
                "O AutoARIMA é uma abordagem automatizada que determina automaticamente os melhores parâmetros do modelo ARIMA para uma série temporal.",

            "Dynamic Optimized Theta":
                "O Dynamic Optimized Theta otimiza dinamicamente os parâmetros de suavização Theta para se adaptar às características específicas da série temporal.",

            "SESOpt (Simple Exponential Smoothing Otimizado)":
                "O SESOt é uma variação do SES que otimiza automaticamente os parâmetros de suavização exponencial para melhor se ajustar aos dados da série temporal.",

            "SarimaX":
                "O SarimaX é uma extensão do modelo SARIMA que permite a inclusão de covariáveis externas para melhorar a precisão das previsões.",

            "LSTM (Long Short-Term Memory)":
                "O LSTM é um modelo de rede neural recorrente capaz de capturar dependências de longo prazo em sequências temporais.",

            "Prophet":
                "O Prophet é uma biblioteca de previsão desenvolvida pelo Facebook, projetada para séries temporais com tendências sazonais e feriados. "
                "É indicado para modelagem sem conhecimento profundo em análise de séries temporais."
        }

        col1, col2 = st.columns([1, 2])

        with col1:
            selected_model = st.radio("Melhores modelos por desempenho", list(model_explanations.keys()))

        with col2:
            st.write(intro_text)
            st.write("---")
            st.write(model_explanations[selected_model]) # type: ignore
    with tab6:
        '''
        ## Modelo com melhor desempenho

        Após a avaliação dos erros (MAPE, WMAPE, SMAPE) gerados pelo modelo, observou-se que o Seasonal Exponential Smoothing Otimizado não somente demonstrou o melhor ajuste aos dados apresentados,
        mas também se destacou ao oferecer um desempenho superior com um custo computacional reduzido, consolidando-se assim como a escolha mais adequada para o desafio em questão.
        '''
        txt = (
            "StatsForecast (Seasonal Exponential Smoothing Otimizado)"
                "\nO modelo StatsForecast utiliza o Seasonal Exponential Smoothing Otimizado da biblioteca StatsForecast para prever séries temporais. "
                "\nEle considera padrões sazonais e ajusta automaticamente os parâmetros de suavização para se adaptar aos dados. "
                "\nO parâmetro `season_length` define o comprimento da sazonalidade na série temporal. O modelo é eficaz para séries com padrões sazonais complexos."
        )
        st.write(txt)
        st.code("model = StatsForecast(models=[SeasonalExponentialSmoothingOptimized(season_length=h)], freq='D', n_jobs=-1)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['y'], mode='lines', name='Original'))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['SeasESOpt'], mode='lines', name='SeasESOpt'))
        fig.update_layout(
            xaxis_title='Data',
            legend=dict(x=0, y=1),
            xaxis=dict(tickformat='%Y-%m-%d')
        )
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
# %%