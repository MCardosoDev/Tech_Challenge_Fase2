#%%
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.stattools import adfuller
from Utils import (
    plot_ts,
    decompose,
    data_diff,
    acf_pacf,
    plot_error
)
from statsforecast import StatsForecast
from statsforecast.models import MSTL, AutoETS, AutoARIMA

#%%
df = pd.read_csv('df.csv')
df_st = df.copy()
df_st['Data'] = pd.to_datetime(df_st['Data'], format='%Y-%m-%d')
df_st['ds'] = pd.to_datetime(df_st['ds'], format='%Y-%m-%d')
df_st.set_index('Data', inplace=True)

#%%
df_model = pd.read_csv('df_model.csv')
df_model['Data'] = pd.to_datetime(df_model['Data'], format='%Y-%m-%d')
df_model['ds'] = pd.to_datetime(df_model['ds'], format='%Y-%m-%d')
df_model.set_index('Data', inplace=True)

df_model_diff = pd.read_csv('df_diff_model.csv')
df_model_diff['Data'] = pd.to_datetime(df_model_diff['Data'], format='%Y-%m-%d')
df_model_diff['ds'] = pd.to_datetime(df_model_diff['ds'], format='%Y-%m-%d')
df_model_diff.set_index('Data', inplace=True)

#%%
df_log = np.log(df_st[['Último', 'Abertura', 'Máxima', 'Mínima', 'Vol.', 'y']])
df_log_meam = df_log.rolling(12).mean() # type: ignore
df_log = (df_log - df_log_meam).dropna()
df_diff = df_log.diff(1).dropna() # primeira derivada
df_ultimo_diff = df_diff['y']
df_ultimo_mean = df_ultimo_diff.rolling(12).mean()
df_ultimo_std = df_ultimo_diff.rolling(12).std()

#%%
train = pd.read_csv('train.csv')
train['Data'] = pd.to_datetime(train['Data'])
train['ds'] = pd.to_datetime(train['ds'])
train.set_index('Data', inplace=True)
test = pd.read_csv('test.csv')
test['Data'] = pd.to_datetime(test['Data'])
test['ds'] = pd.to_datetime(test['ds'])
test.set_index('Data', inplace=True)
valid = pd.read_csv('valid.csv')
valid['Data'] = pd.to_datetime(valid['Data'])
valid['ds'] = pd.to_datetime(valid['ds'])
valid.set_index('Data', inplace=True)

#%%
train_diff = pd.read_csv('train_diff.csv')
train_diff['ds'] = pd.to_datetime(train_diff['ds'])
train_diff['Data'] = pd.to_datetime(train_diff['Data'])
train_diff.set_index('Data', inplace=True)
test_diff = pd.read_csv('test_diff.csv')
test_diff['Data'] = pd.to_datetime(test_diff['Data'])
test_diff['ds'] = pd.to_datetime(test_diff['ds'])
test_diff.set_index('Data', inplace=True)
valid_diff = pd.read_csv('valid_diff.csv')
valid_diff['Data'] = pd.to_datetime(valid_diff['Data'])
valid_diff['ds'] = pd.to_datetime(valid_diff['ds'])
valid_diff.set_index('Data', inplace=True)

h = test.index.nunique()
#%%
columns_to_print = ['Data', 'Último', 'Abertura', 'Máxima', 'Mínima', 'Vol.', 'Var%']
df_erros = pd.read_csv('Data/Errors.csv')
df_erros_prophet = pd.read_csv('Data/ErrorsProphet.csv')
df_mesclado = pd.concat([df_erros_prophet, df_erros], ignore_index=True)
df_mesclado.to_csv('erros.csv', index=False)

df_erros_diff = pd.read_csv('erros_diff.csv')

#%%
def test_adfuller_streamlit(df):
    result = adfuller(df.values)
    st.write('Teste ADF')
    st.write(f'Teste estatístico: {result[0]}')
    st.write(f'Valor p: {result[1]}')
    st.write('Valores críticos:')
    for key, value in result[4].items(): #type: ignore
        st.write(f'\t{key}: {value}')

def wmape_error(y_true, y_pred):
    wmape = np.abs(y_true - y_pred).sum() / np.abs(y_true).sum()
    return wmape
#%%
wmape_formula = r'WMAPE = \frac{\sum_{i=1}^{n} \left| y_{\text{true},i} - y_{\text{pred},i} \right|}{\sum_{i=1}^{n} \left| y_{\text{true},i} \right|} \times 100'
smape_formula = r'SMAPE = \frac{100}{n} \sum_{i=1}^{n} \frac{\left| y_{\text{pred},i} - y_{\text{true},i} \right|}{(|y_{\text{pred},i}| + |y_{\text{true},i}|)/2}'

wmape_explanation = "O WMAPE (Weighted Mean Absolute Percentage Error) é uma variação do MAPE que pondera os erros pela magnitude dos valores verdadeiros."
smape_explanation = "O SMAPE (Symmetric Mean Absolute Percentage Error) é uma métrica que considera a simetria dos erros, levando em conta a magnitude dos valores verdadeiros e das previsões."

#%%
# Modelo MSTL com AutoATS
model = StatsForecast(models=[MSTL(season_length=[5, 7], trend_forecaster=AutoETS(model='AZN'))], freq='D', n_jobs=-1) #type: ignore
model.fit(train)
forecast = model.predict(h=h, level=[95])
forecast = forecast.reset_index().merge(test, on=['ds', 'unique_id'], how='left')
forecast.dropna(inplace=True)

forecast_valid = model.predict(h=h + h, level=[95])
forecast_valid = forecast_valid.reset_index().merge(valid, on=['ds', 'unique_id'], how='left')
forecast_valid.dropna(inplace=True)

#%%
# Modelo MSTL com AutoATS dados diferenciados
model_diff = StatsForecast(models=[MSTL(season_length=[5, 7], trend_forecaster=AutoETS(model='AZN'))], freq='D', n_jobs=-1)
model_diff.fit(train_diff)
forecast_diff = model_diff.predict(h=h, level=[95])
forecast_diff = forecast_diff.reset_index().merge(test_diff, on=['ds', 'unique_id'], how='left')
forecast_diff.dropna(inplace=True)

forecast_diff_valid = model_diff.predict(h=h + h, level=[95])
forecast_diff_valid = forecast_diff_valid.reset_index().merge(valid_diff, on=['ds', 'unique_id'], how='left')
forecast_diff_valid.dropna(inplace=True)

#%%
# Modelo MSTL com AutoArima
model_a = StatsForecast(models=[MSTL(season_length=[5, 7], trend_forecaster=AutoARIMA(season_length=1, start_P=1, start_Q=1))], freq='D', n_jobs=-1) # type: ignore
model_a.fit(train)
forecast_a = model_a.predict(h=h, level=[95])
forecast_a = forecast_a.reset_index().merge(test, on=['ds', 'unique_id'], how='left')
forecast_a.dropna(inplace=True)

forecast_valid_a = model_a.predict(h=h + h, level=[95])
forecast_valid_a = forecast_valid_a.reset_index().merge(valid, on=['ds', 'unique_id'], how='left')
forecast_valid_a.dropna(inplace=True)

#%%
def main():
    st.set_page_config(layout="wide")
    st.title('Análise de fechamento diário da IBOVESPA')
    tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(['Geral', 'Dados Históricos - IBOVESPA', 'Transformação da Série', 'ACF e PCF', 'Avaliação', 'Melhores Modelos', 'Melhor Modelo', 'Resultado'])

    with tab0:
        '''
        ## Análise dos Dados para Prever Diariamente o Fechamento da base

        Esta análise tem como objetivo utilizar os dados disponíveis sobre as informações da bolsa de valores, 
        realizar um modelo preditivo, criar uma série temporal e prever diariamente o fechamento da base.
        
        ### Dados Utilizados para análise

        *Análise dos dados históricos da IBOVESPA* - Os dados sobre o histórico da bolsa foram obtidos do investing.

        - Link: [Dados Históricos - Ibovespa](https://br.investing.com/indices/bovespa-historical-data)
        '''
        st.write("\n")
        st.markdown('##### Dados Históricos - Ibovespa')
        st.dataframe(df[columns_to_print].head(10).reset_index(drop=True), use_container_width=True)
        st.markdown(
            '<span class="small-font">Dados utilizados de um período entre 2003 até 2023</span>',
        unsafe_allow_html=True)   
        '''
        ### Diferenciação da série para conversão em uma série estacionária.
        
        A diferenciação é usada para transformar uma série temporal não estacionária em uma série estacionária. Para *modelos como o ARIMA, **Remoção de Tendências, **Estabilização da Variância* e *Eliminação de Sazonalidade*.

        ### Autocorrelação (ACF) e Autocorrelação parcial (PACF).
        
        *Função de autocorrelação (ACF) e Função de Autocorrelação parcial (PACF)*. A ACF mede a correlação entre os valores passados e os valores presentes de uma série temporal, levando em consideração todos os atrasos possíveis e identifica padrões de correlação entre observações em diferentes pontos temporais.
        A PACF mede a correlação direta entre dois pontos temporais, controlando os efeitos dos atrasos intermediários.

        ### Teste e Avaliação de modelos preditivos

        *Avaliação dos de modelos para séries temporais* - Para encontrar o melhor modelo preditivo utiliza-se testes de erros wMAPE para avaliar a precisão de acerto de cada modelo.

        ### Melhor Modelo

        *Escolha do melhor modelo* - Com avaliação do erros cometidos é possível escolher o modelo com melhor desempenho.

        '''
    
    with tab1:
        '''
        ## Análise Exploratória de Dados (EDA)

        A decomposição da série temporal é empregada para desmembrar os dados em componentes distintos, como tendência, sazonalidade e resíduos, permitindo uma compreensão mais profunda dos padrões subjacentes e variações presentes ao longo do tempo.
        Essa abordagem facilita a análise individual de cada componente, auxiliando na identificação de informações relevantes para modelagem e previsão.
        
        Ao plotar apenas os dados normalmente consegui-se notar apenas um crescimento no valor de fechamento diário da bolsa.,
        '''
        st.plotly_chart(plot_ts(df_st), use_container_width = True)
        '''
        Ao plotar a decomposição nota-se que realmente se tem um tendência de crescimento juntamente com o múltiplas sazonalidade e anomalias residuais em 2009 e 2020.

            * Crise econômica de 2008-2009 iniciada no mercado imobiliário dos Estados Unidos refletiu na economia brasileira.
            * Epidemia global de Covid19 2019-2020.
        '''
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
                    st.markdown("<span style='color:red'>P-value 62% e teste estatístico maior que os valores criticos: A série não é estacionária</span>", unsafe_allow_html=True)
    
    with tab2:
        '''
        ## Transformação da Série Temporal

        Realizar transformações matemáticas em uma série temporal não estacionária para torná-la estacionária permite a aplicação de técnicas estatísticas e modelos de previsão com esse pressuposto.
        Isso é fundamental para compreender padrões temporais e obter previsões mais precisas. As transformações realizadas na série foram:

            - Aplicação do Logaritmo, para reduzir flutuações
            - Subtração da Média Móvel com janela de tamanho 12 meses
            - Primeira diferenciação, para reduzir tendências
        '''
        st.plotly_chart(data_diff(df_ultimo_diff, df_ultimo_mean, df_ultimo_std), use_container_width = True)
        '''
        Nota-se a mesma anomalia do resíduo do plote da decomposição no resultado da transformação matemática em 2009 e 2020.
        
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
                    test_adfuller_streamlit(df_ultimo_diff)
                with col3:
                    st.write("Resultado")
                    st.write("---")
                    st.markdown("<span style='color:green'>P-value 0.0% e teste estatístico menor que os valores criticos: A série é estacionária após a diferenciação</span>", unsafe_allow_html=True)

    with tab3:
        '''
        ## Autocorrelação (ACF) e Autocorrelação parcial (PACF)

        Os gráficos de autocorrelação (ACF) e autocorrelação parcial (PACF) são ferramentas essenciais na análise de séries temporais.
        ACF revela correlações em diferentes defasagens, enquanto PACF ajuda a identificar correlações diretas e eliminação de defasagens irrelevantes em modelos de previsão.
        Ao observar onde essas correlações se tornam não significativas nos gráficos PACF e ACF, você pode obter sugestões para os parâmetros Q (ordem do termo MA) e P (ordem do termo AR) para modelos como o ARMA,
        ARIMA ou SARIMA, os quais ajudam a modelar adequadamente a série temporal.

        - 5% ACF (intervalo de confiança).
        - 1.96/sqrt(N-d)
            - *N* número de pontos e *d* é o número de vezes que os dados foram diferenciamos (intervalo de confiança para estimativas de autocorrelação significativa).
        '''
        st.plotly_chart(acf_pacf(df_ultimo_diff), use_container_width = True)
        st.write("  Ordem de diferenciação *D* = 1 (Foi necessária 1 diferenciação para tornar a série estacionaria)")
        st.write("  *Q acf* = 0.915")
        st.write("  *P pacf* = 0.915")

    with tab4:
        '''
        ## Avaliação dos modelos - Métricas de Erro em Previsões
        '''
        st.write("\n")
        st.markdown('##### Dados preenchidos com média móvel')
        st.write('''
                Pelo motivo da Ibovespa não operar em finais de semana e feriados o dataset tem um problema de dados faltantes e para resolver esse problema é optado pela solução de 
                preenchimento para dados faltantes que combina o conceito de ***forward fill*** (utilizar dados passados) que nessa oportunidade usa-se a média móvel (segunda a sexta-feira),
                excluindo a utilização de valores futuros ***lokahead***.

                O Período utilizado para treinar, testar e validar os modelos.
            '''
        )
        st.code('''
                train = df_model.loc[df_model.ds < '2023-07-01']
                test = df_model.loc[(df_model.ds >= '2023-07-01') & (df_model.ds < '2023-08-16')]
                valid = df_model.loc[df_model.ds >= '2023-08-16']'''
        )

        col1, col2 = st.columns(2)
        with col1:
            st.write("Dados originais")
            st.dataframe(df_model.head(7).reset_index(drop=True), use_container_width=True)
        with col2:
            st.write("Dados transformandos")
            st.dataframe(df_model_diff.head(7).reset_index(drop=True), use_container_width=True)

        st.write("\n")
        st.markdown('##### WMAPE - Weighted Mean Absolute Percentage Error')
        st.latex(wmape_formula)
        st.write(wmape_explanation)
        st.plotly_chart(plot_error(df_mesclado, 'wmape', 'wMAPE', 'wMAPE (Weighted Mean Absolute Percentage Error) - Dados originais'), use_container_width = True)
        st.write("\n")
        st.markdown('##### WMAPE - Weighted Mean Absolute Percentage Error')
        st.plotly_chart(plot_error(df_erros_diff, 'wmape', 'wMAPE', 'wMAPE (Weighted Mean Absolute Percentage Error) - Dados transformandos'), use_container_width = True)

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
            "AutoETS":
                "O AutoETS (Automatic Exponential Smoothing) é um método automatizado que seleciona automaticamente o melhor modelo de suavização exponencial para uma série temporal. "
                "Ele considera diferentes configurações de suavização (além de sazonalidade e tendência) para melhor se ajustar aos padrões da série e produzir previsões precisas.",

            "Holt":
                "O método de Holt é uma técnica de suavização exponencial dupla que leva em consideração a tendência da série temporal. "
                "Ele é apropriado para séries que possuem uma tendência linear. O modelo de Holt utiliza suavização exponencial para a tendência e para os valores observados, "
                "fornecendo previsões que capturam tanto a direção quanto a magnitude das mudanças ao longo do tempo.",

            "RandomWalkWithDrift":
                "O RandomWalkWithDrift modela uma série temporal onde cada valor é uma pequena variação do valor anterior, mas com a adição de um componente de deriva (drift) linear. "
                "Isso pode ser usado para modelar tendências lineares aproximadas.",

            "SeasonalWindowAverage":
                "O método SeasonalWindowAverage calcula a média dos valores em uma janela móvel de tamanho fixo ao longo da série temporal, considerando uma sazonalidade específica. "
                "Ele suaviza flutuações sazonais de curto prazo, proporcionando uma visão da tendência sazonal.",

            "CES (Croston's Exponential Smoothing)":
                "O método CES é usado para prever a demanda de itens esporádicos ou intermitentes, considerando componentes de nível e probabilidade de demanda.",

            "Window Average (Média Móvel Simples)":
                "O método da Média Móvel Simples calcula a média dos valores em uma janela móvel de tamanho fixo ao longo da série temporal. "
                "Ele suaviza flutuações de curto prazo, revelando tendências subjacentes.",

            "Dynamic Optimized Theta":
                "O Dynamic Optimized Theta otimiza dinamicamente os parâmetros de suavização Theta para se adaptar às características específicas da série temporal.",

            "AutoTheta":
                "O AutoTheta é uma versão automatizada do método Theta, que é uma técnica de suavização exponencial dupla para séries temporais com sazonalidade aditiva.",

            "IMAPA":
                "O IMAPA (Intermittent MA for PArtial data) é um método que lida com a previsão de demanda intermitente usando uma abordagem baseada em médias móveis ponderadas.",

            "ADIDA":
                "O ADIDA (Adaptive Intermittent Demand Analysis) é um método projetado para lidar com séries temporais de demanda intermitente, onde os padrões de demanda são imprevisíveis e irregulares.",

            "CrostonOptimized":
                "O CrostonOptimized é uma versão otimizada do método Croston, usado para prever a demanda de itens esporádicos. "
                "Ele ajusta automaticamente os parâmetros para melhorar a precisão das previsões.",

            "SESOpt (Simple Exponential Smoothing Otimizado)":
                "O SESOt é uma variação do SES que otimiza automaticamente os parâmetros de suavização exponencial para melhor se ajustar aos dados da série temporal.",
            
            "Holt-Winters":
                "O método de Holt-Winters é uma técnica de suavização exponencial tripla que considera três componentes: nível, tendência e sazonalidade. "
                "Ele modela séries temporais com tendência e padrões sazonais.",

            "MSTL":
                "O MSTL (Multiple Seasonal Decomposition of Time Series) é um método que decompoẽ séries temporais em diferentes componentes, como sazonalidade e tendência, "
                "para modelagem mais precisa. Ele suporta múltiplos comprimentos sazonais e utiliza o modelo AutoETS para previsões.",

            "LSTM (Long Short-Term Memory)":
                "O LSTM é um modelo de rede neural recorrente capaz de capturar dependências de longo prazo em sequências temporais.",

            "AutoARIMA":
                "O AutoARIMA é uma abordagem automatizada que determina automaticamente os melhores parâmetros do modelo ARIMA para uma série temporal.",

            "SarimaX":
                "O SarimaX é uma extensão do modelo SARIMA que permite a inclusão de covariáveis externas para melhorar a precisão das previsões.",

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

        Após a avaliação dos erros no ***wMAPE*** gerados pelo modelo, observou-se que apesar de múltiplos modelos terem uma porcentagem de ***wMAPE*** entre 1% e 5% o modelo 
        ***Multiple Seasonal Decomposition of Time Series - MSTL*** obteve uma melhor avaliação ao validar o modelo com novos dados.

        Como notou-se na decomposição dos dados múltiplas sazonalidades por conta do dataset do desafio conter dados caóticos tivemos um modelo de múltiplas sazonalidades com o melhor desempenho.
         
        Não somente demonstrou o melhor ajuste aos dados apresentados, mas também se destacou ao oferecer um desempenho superior com um custo computacional reduzido,
        consolidando-se assim como a escolha mais adequada para o desafio em questão.

        O MSTL (Decomposição Múltipla de Sazonalidade-Tendência usando LOESS) decompõe a série temporal em múltiplas sazonalidades (season_length=[5, 7]) usando LOESS (Local Regression).
        Que ajusta modelos de regressão localmente, em vez de globalmente, útil para capturar padrões não lineares em séries temporais e dados espaciais.

        Em seguida, faz previsões para a tendência usando um modelo personalizado não sazonal e para cada sazonalidade usando outro modelo selecionado.
        '''
        txt_model_1 = (
            "O modelo utiliza o método MSTL com o AutoETS (Automatic Exponential Smoothing model) para suavização exponencial automatizado."
            "\n O modelo ETS (Erro, Tendência, Sazonalidade) as equações state-space podem ser determinadas com base em seus componentes multiplicativos, aditivos, otimizados ou omitidos."
            "\n O modelo usa model='AZN' (additive error, optimized trend, and no seasonality)"
            "\n Se o componente for selecionado como 'Z', ele atua como um espaço reservado para pedir ao modelo AutoETS que descubra o melhor parâmetro."
        )
        txt_model_2 = (
            "\nO modelo utiliza o método MSTL com o AutoARIMA que seleciona automaticamente o melhor modelo ARIMA (AutoRegressive Integrated Moving Average)"
        )

        st.write(txt_model_1)
        st.code("model = StatsForecast(models=[MSTL(season_length=[5, 7], trend_forecaster=AutoETS(model='AZN'))], freq='D', n_jobs=-1)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train['ds'], y=train['y'], mode='lines', name='Dados de Treinamento (2004-01-02 a 2023-06-30)', line=dict(color='rgba(0, 255, 0, 0.8)')))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['y'], mode='lines', name='Dados de Treinamento (2004-01-02 a 2023-06-30)', line=dict(color='rgba(0, 255, 0, 0.8)')))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['MSTL'], mode='lines', name='Previsão MSTL', line=dict(color='rgba(0, 255, 255, 0.8)')))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['MSTL-lo-95'], mode='lines', name='Limite Inferior (95%)', line=dict(color='rgba(255, 0, 255, 0.8)')))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['MSTL-hi-95'], mode='lines', name='Limite Superior (95%)', line=dict(color='rgba(255, 0, 255, 0.8)')))
        fig.update_layout(
            xaxis_title='Data',
            legend=dict(x=0, y=1),
            xaxis=dict(tickformat='%Y-%m-%d')
        )
        fig.add_shape(
            type='line',
            x0='2023-07-01',
            x1='2023-07-01',
            y0=80,
            y1=150,
            line=dict(color='rgba(255, 255, 0, 0.8)', dash='dash')
        )
        fig.add_annotation(
            x='2023-07-01',
            y=150,
            text='First Forecast',
            showarrow=False
        )
        fig.add_trace(
            go.Scatter(
                x=forecast['ds'],
                y=forecast['MSTL-lo-95'],
                mode='lines',
                fill='tonexty',
                fillcolor='rgba(255, 0, 255, 0.1)',
                line=dict(width=0),
                showlegend=False
            )
        )
        fig.update_yaxes(range=[80, 150])
        fig.update_xaxes(
            range=['2022-01-01', forecast['ds'].max()]
        )
        st.write(f"WMAPE (Weighted Mean Absolute Percentage Error): {wmape_error(forecast['y'].values, forecast['MSTL'].values):.2%}")
        st.plotly_chart(fig, use_container_width=True)

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=forecast_valid['ds'], y=forecast_valid['y'], mode='lines', name='Dados de  Validação (2023-08-16 a 2023-08-25)', line=dict(color='rgba(0, 255, 0, 0.8)')))
        fig1.add_trace(go.Scatter(x=forecast_valid['ds'], y=forecast_valid['MSTL'], mode='lines', name='Previsão MSTL', line=dict(color='rgba(0, 255, 255, 0.8)')))
        fig1.add_trace(go.Scatter(x=forecast_valid['ds'], y=forecast_valid['MSTL-lo-95'], mode='lines', name='Limite Inferior (95%)', line=dict(color='rgba(255, 0, 255, 0.8)')))
        fig1.add_trace(go.Scatter(x=forecast_valid['ds'], y=forecast_valid['MSTL-hi-95'], mode='lines', name='Limite Superior (95%)', line=dict(color='rgba(255, 0, 255, 0.8)')))
        fig1.update_layout(
            xaxis_title='Data',
            legend=dict(x=0, y=1),
            xaxis=dict(tickformat='%Y-%m-%d')
        )
        fig1.add_trace(
            go.Scatter(
                x=forecast_valid['ds'],
                y=forecast_valid['MSTL-lo-95'],
                mode='lines',
                fill='tonexty',
                fillcolor='rgba(255, 0, 255, 0.1)',
                line=dict(width=0),
                showlegend=False
            )
        )
        fig1.update_yaxes(range=[90, 160])
        st.write(f"WMAPE (Weighted Mean Absolute Percentage Error): {wmape_error(forecast_valid['y'].values, forecast_valid['MSTL'].values):.2%}")
        st.plotly_chart(fig1, use_container_width=True)

        st.write(txt_model_2)
        st.code("model = StatsForecast(models=[MSTL(season_length=[5, 7], trend_forecaster=AutoARIMA(season_length=1, start_P=1, start_Q=1))], freq='D', n_jobs=-1)")
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=train['ds'], y=train['y'], mode='lines', name='Dados de Treinamento (2004-01-02 a 2023-06-30)', line=dict(color='rgba(0, 255, 0, 0.8)')))
        fig3.add_trace(go.Scatter(x=forecast_a['ds'], y=forecast_a['y'], mode='lines', name='Dados de Treinamento (2004-01-02 a 2023-06-30)', line=dict(color='rgba(0, 255, 0, 0.8)')))
        fig3.add_trace(go.Scatter(x=forecast_a['ds'], y=forecast_a['MSTL'], mode='lines', name='Previsão MSTL', line=dict(color='rgba(0, 255, 255, 0.8)')))
        fig3.add_trace(go.Scatter(x=forecast_a['ds'], y=forecast_a['MSTL-lo-95'], mode='lines', name='Limite Inferior (95%)', line=dict(color='rgba(255, 0, 255, 0.8)')))
        fig3.add_trace(go.Scatter(x=forecast_a['ds'], y=forecast_a['MSTL-hi-95'], mode='lines', name='Limite Superior (95%)', line=dict(color='rgba(255, 0, 255, 0.8)')))
        fig3.update_layout(
            xaxis_title='Data',
            legend=dict(x=0, y=1),
            xaxis=dict(tickformat='%Y-%m-%d')
        )
        fig3.add_shape(
            type='line',
            x0='2023-07-01',
            x1='2023-07-01',
            y0=80,
            y1=150,
            line=dict(color='rgba(255, 255, 0, 0.8)', dash='dash')
        )
        fig3.add_annotation(
            x='2023-07-01',
            y=150,
            text='First Forecast',
            showarrow=False
        )
        fig3.add_trace(
            go.Scatter(
                x=forecast_a['ds'],
                y=forecast_a['MSTL-lo-95'],
                mode='lines',
                fill='tonexty',
                fillcolor='rgba(255, 0, 255, 0.1)',
                line=dict(width=0),
                showlegend=False
            )
        )
        fig3.update_yaxes(range=[80, 150])
        fig3.update_xaxes(
            range=['2022-01-01', forecast['ds'].max()]
        )
        st.write(f"WMAPE (Weighted Mean Absolute Percentage Error): {wmape_error(forecast_a['y'].values, forecast_a['MSTL'].values):.2%}")
        st.plotly_chart(fig3, use_container_width=True)

        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=forecast_valid_a['ds'], y=forecast_valid_a['y'], mode='lines', name='Dados de  Validação (2023-08-16 a 2023-08-25)', line=dict(color='rgba(0, 255, 0, 0.8)')))
        fig4.add_trace(go.Scatter(x=forecast_valid_a['ds'], y=forecast_valid_a['MSTL'], mode='lines', name='Previsão MSTL', line=dict(color='rgba(0, 255, 255, 0.8)')))
        fig4.add_trace(go.Scatter(x=forecast_valid_a['ds'], y=forecast_valid_a['MSTL-lo-95'], mode='lines', name='Limite Inferior (95%)', line=dict(color='rgba(255, 0, 255, 0.8)')))
        fig4.add_trace(go.Scatter(x=forecast_valid_a['ds'], y=forecast_valid_a['MSTL-hi-95'], mode='lines', name='Limite Superior (95%)', line=dict(color='rgba(255, 0, 255, 0.8)')))
        fig4.update_layout(
            xaxis_title='Data',
            legend=dict(x=0, y=1),
            xaxis=dict(tickformat='%Y-%m-%d')
        )
        fig4.add_trace(
            go.Scatter(
                x=forecast_valid_a['ds'],
                y=forecast_valid_a['MSTL-lo-95'],
                mode='lines',
                fill='tonexty',
                fillcolor='rgba(255, 0, 255, 0.1)',
                line=dict(width=0),
                showlegend=False
            )
        )
        st.write(f"WMAPE (Weighted Mean Absolute Percentage Error): {wmape_error(forecast_valid_a['y'].values, forecast_valid_a['MSTL'].values):.2%}")
        fig4.update_yaxes(range=[90, 160])
        st.plotly_chart(fig4, use_container_width=True)

    with tab7:
        '''
        ## Resultado

        O MSTL (Multiple Seasonal Decomposition of Time Series) com o range de sazonalidade de 5 e 7 dias com o uso do modelo AutoETS (Automatic Exponential Smoothing model) 
        passando os hiperparametros 'AZN' (additive error, optimized trend, and no seasonality) ou com o uso do AutoARIMA tivera uma avaliação muito parecedos nos dado de validação.

        Mas com a opção do usao dos dados originais, pois os três passos de transformação aplicados à série temporal têm um impacto significativo em sua estrutura original.

        Essas etapas formam uma composição de funções que modificam a série original, tornando-a menos periódica e mais estacionária.
        No entanto, devido a essas transformações, a interpretação das estatísticas, como a média, pode se tornar desafiadora, já que a natureza da série temporal original foi significativamente alterada.

        Em essência, isso cria um cenário em que temos um problema de função composta, onde a série se comporta de maneira periódica, mas com um período que varia em função de outra função, tornando a média um aspecto complicado de se analisar.
        
        Segue a comparação de uso dos dois conjuntos de datasets:

        '''
        st.write("Uso dos dados originais")
        st.code("model = StatsForecast(models=[MSTL(season_length=[5, 7], trend_forecaster=AutoETS(model='AZN'))], freq='D', n_jobs=-1)")
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=train['ds'], y=train['y'], mode='lines', name='Dados de Treinamento (2004-01-02 a 2023-06-30)', line=dict(color='rgba(0, 255, 0, 0.8)')))
        fig5.add_trace(go.Scatter(x=forecast['ds'], y=forecast['y'], mode='lines', name='Dados de Treinamento (2004-01-02 a 2023-06-30)', line=dict(color='rgba(0, 255, 0, 0.8)')))
        fig5.add_trace(go.Scatter(x=forecast['ds'], y=forecast['MSTL'], mode='lines', name='Previsão MSTL', line=dict(color='rgba(0, 255, 255, 0.8)')))
        fig5.add_trace(go.Scatter(x=forecast['ds'], y=forecast['MSTL-lo-95'], mode='lines', name='Limite Inferior (95%)', line=dict(color='rgba(255, 0, 255, 0.8)')))
        fig5.add_trace(go.Scatter(x=forecast['ds'], y=forecast['MSTL-hi-95'], mode='lines', name='Limite Superior (95%)', line=dict(color='rgba(255, 0, 255, 0.8)')))
        fig5.update_layout(
            xaxis_title='Data',
            legend=dict(x=0, y=1),
            xaxis=dict(tickformat='%Y-%m-%d')
        )
        fig5.add_shape(
            type='line',
            x0='2023-07-01',
            x1='2023-07-01',
            y0=80,
            y1=150,
            line=dict(color='rgba(255, 255, 0, 0.8)', dash='dash')
        )
        fig5.add_annotation(
            x='2023-07-01',
            y=150,
            text='First Forecast',
            showarrow=False
        )
        fig5.add_trace(
            go.Scatter(
                x=forecast['ds'],
                y=forecast['MSTL-lo-95'],
                mode='lines',
                fill='tonexty',
                fillcolor='rgba(255, 0, 255, 0.1)',
                line=dict(width=0),
                showlegend=False
            )
        )
        fig5.update_yaxes(range=[80, 150])
        fig5.update_xaxes(
            range=['2022-01-01', forecast['ds'].max()]
        )
        st.write(f"WMAPE (Weighted Mean Absolute Percentage Error): {wmape_error(forecast['y'].values, forecast['MSTL'].values):.2%}")
        st.plotly_chart(fig5, use_container_width=True)

        fig6 = go.Figure()
        fig6.add_trace(go.Scatter(x=forecast_valid['ds'], y=forecast_valid['y'], mode='lines', name='Dados de  Validação (2023-08-16 a 2023-08-25)', line=dict(color='rgba(0, 255, 0, 0.8)')))
        fig6.add_trace(go.Scatter(x=forecast_valid['ds'], y=forecast_valid['MSTL'], mode='lines', name='Previsão MSTL', line=dict(color='rgba(0, 255, 255, 0.8)')))
        fig6.add_trace(go.Scatter(x=forecast_valid['ds'], y=forecast_valid['MSTL-lo-95'], mode='lines', name='Limite Inferior (95%)', line=dict(color='rgba(255, 0, 255, 0.8)')))
        fig6.add_trace(go.Scatter(x=forecast_valid['ds'], y=forecast_valid['MSTL-hi-95'], mode='lines', name='Limite Superior (95%)', line=dict(color='rgba(255, 0, 255, 0.8)')))
        fig6.update_layout(
            xaxis_title='Data',
            legend=dict(x=0, y=1),
            xaxis=dict(tickformat='%Y-%m-%d')
        )
        fig6.add_trace(
            go.Scatter(
                x=forecast_valid['ds'],
                y=forecast_valid['MSTL-lo-95'],
                mode='lines',
                fill='tonexty',
                fillcolor='rgba(255, 0, 255, 0.1)',
                line=dict(width=0),
                showlegend=False
            )
        )
        fig6.update_yaxes(range=[90, 160])
        st.write(f"WMAPE (Weighted Mean Absolute Percentage Error): {wmape_error(forecast_valid['y'].values, forecast_valid['MSTL'].values):.2%}")
        st.plotly_chart(fig6, use_container_width=True)

        st.write("Uso dos dados transformados matematicamente")
        fig7 = go.Figure()
        fig7.add_trace(go.Scatter(x=train_diff['ds'], y=train_diff['y'], mode='lines', name='Dados de Treinamento (2004-01-02 a 2023-06-30)', line=dict(color='rgba(0, 255, 0, 0.8)')))
        fig7.add_trace(go.Scatter(x=forecast_diff['ds'], y=forecast_diff['y'], mode='lines', name='Dados de Treinamento (2004-01-02 a 2023-06-30)', line=dict(color='rgba(0, 255, 0, 0.8)')))
        fig7.add_trace(go.Scatter(x=forecast_diff['ds'], y=forecast_diff['MSTL'], mode='lines', name='Previsão MSTL', line=dict(color='rgba(0, 255, 255, 0.8)')))
        fig7.add_trace(go.Scatter(x=forecast_diff['ds'], y=forecast_diff['MSTL-lo-95'], mode='lines', name='Limite Inferior (95%)', line=dict(color='rgba(255, 0, 255, 0.8)')))
        fig7.add_trace(go.Scatter(x=forecast_diff['ds'], y=forecast_diff['MSTL-hi-95'], mode='lines', name='Limite Superior (95%)', line=dict(color='rgba(255, 0, 255, 0.8)')))
        fig7.update_layout(
            xaxis_title='Data',
            legend=dict(x=0, y=1),
            xaxis=dict(tickformat='%Y-%m-%d')
        )
        fig7.add_shape(
            type='line',
            x0='2023-07-01',
            x1='2023-07-01',
            y0=-0.1,
            y1=0.1,
            line=dict(color='rgba(255, 255, 0, 0.8)', dash='dash')
        )
        fig7.add_annotation(
            x='2023-07-01',
            y=0.1,
            text='First Forecast',
            showarrow=False
        )
        fig7.add_trace(
            go.Scatter(
                x=forecast_diff['ds'],
                y=forecast_diff['MSTL-lo-95'],
                mode='lines',
                fill='tonexty',
                fillcolor='rgba(255, 0, 255, 0.1)',
                line=dict(width=0),
                showlegend=False
            )
        )
        fig7.update_yaxes(range=[-0.05, 0.05])
        fig7.update_xaxes(
            range=['2022-01-01', forecast_diff['ds'].max()]
        )
        st.write(f"WMAPE (Weighted Mean Absolute Percentage Error): {wmape_error(forecast_diff['y'].values, forecast_diff['MSTL'].values):.2%}")
        st.plotly_chart(fig7, use_container_width=True)

        fig8 = go.Figure()
        fig8.add_trace(go.Scatter(x=forecast_diff_valid['ds'], y=forecast_diff_valid['y'], mode='lines', name='Dados de  Validação (2023-08-16 a 2023-08-25)', line=dict(color='rgba(0, 255, 0, 0.8)')))
        fig8.add_trace(go.Scatter(x=forecast_diff_valid['ds'], y=forecast_diff_valid['MSTL'], mode='lines', name='Previsão MSTL', line=dict(color='rgba(0, 255, 255, 0.8)')))
        fig8.add_trace(go.Scatter(x=forecast_diff_valid['ds'], y=forecast_diff_valid['MSTL-lo-95'], mode='lines', name='Limite Inferior (95%)', line=dict(color='rgba(255, 0, 255, 0.8)')))
        fig8.add_trace(go.Scatter(x=forecast_diff_valid['ds'], y=forecast_diff_valid['MSTL-hi-95'], mode='lines', name='Limite Superior (95%)', line=dict(color='rgba(255, 0, 255, 0.8)')))
        fig8.update_layout(
            xaxis_title='Data',
            legend=dict(x=0, y=1),
            xaxis=dict(tickformat='%Y-%m-%d')
        )
        fig8.add_trace(
            go.Scatter(
                x=forecast_diff_valid['ds'],
                y=forecast_diff_valid['MSTL-lo-95'],
                mode='lines',
                fill='tonexty',
                fillcolor='rgba(255, 0, 255, 0.1)',
                line=dict(width=0),
                showlegend=False
            )
        )
        # fig8.update_yaxes(range=[90, 160])
        st.write(f"WMAPE (Weighted Mean Absolute Percentage Error): {wmape_error(forecast_diff_valid['y'].values, forecast_diff_valid['MSTL'].values):.2%}")
        st.plotly_chart(fig8, use_container_width=True)

if __name__ == "__main__":
    main()
# %%