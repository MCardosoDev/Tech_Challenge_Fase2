import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objs as go
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm

def convert_volume(volume_str):
    if isinstance(volume_str, str):  # Check if the value is a string
        volume_str = volume_str.replace(',', '.')
        if 'M' in volume_str:
            return int(float(volume_str.replace('M', '')) * 1_000_000)
        elif 'K' in volume_str:
            return int(float(volume_str.replace('K', '')) * 1_000)
        else:
            return int(volume_str)
    else:
        return volume_str

def plot_ts(df):
    fig = sp.make_subplots(rows=4, cols=2, subplot_titles=[
        'Variação do Último Fechamento', 'Variação de Aberturas',
        'Valor Máximo no dia', 'Valor Mínimo do dia',
        'Volume de Ações no dia', 'Variação das Ações no dia'
    ])
    fig.add_trace(go.Scatter(x=df.index, y=df['Último'], mode='lines', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Abertura'], mode='lines', line=dict(color='green')), row=1, col=2)
    fig.add_trace(go.Scatter(x=df.index, y=df['Máxima'], mode='lines', line=dict(color='orange')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Mínima'], mode='lines', line=dict(color='red')), row=2, col=2)
    fig.add_trace(go.Scatter(x=df.index, y=df['Vol.'], mode='lines', line=dict(color='purple')), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Var%'], mode='lines', line=dict(color='cyan')), row=3, col=2)
    for i in range(1, 4):
        fig.update_xaxes(title_text='Data', row=i, col=2)
        fig.update_yaxes(title_text='Valor', row=i, col=1)

    fig.update_layout(title='Variações e Volumes das Ações', showlegend=False)
    fig.update_layout(
        width=1200,
        height=800
    )
    return fig

def decompose(df, title):
    resultados = seasonal_decompose(df['Último'], model='additive', period=365, two_sided=True, extrapolate_trend=5)
    fig_observed = px.line(resultados.observed, x=resultados.observed.index, y='Último', title='Série Observada')
    fig_observed.update_xaxes(title_text='Data')
    fig_trend = px.line(resultados.trend, x=resultados.trend.index, y='trend', title='Componente de Tendência')
    fig_trend.update_xaxes(title_text='Data')
    fig_seasonal = px.line(resultados.seasonal, x=resultados.seasonal.index, y='seasonal', title='Componente de Sazonalidade')
    fig_seasonal.update_xaxes(title_text='Data')
    fig_resid = px.line(resultados.resid, x=resultados.resid.index, y='resid', title='Resíduo')
    fig_resid.update_xaxes(title_text='Data')
    fig = sp.make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1, 
        subplot_titles=[
            'Série Observada',
            'Componente de Tendência',
            'Componente de Sazonalidade',
            'Resíduo']
    )
    fig.add_trace(go.Scatter(x=fig_observed.data[0]['x'], y=fig_observed.data[0]['y'], showlegend=False), row=1, col=1) # type: ignore
    fig.add_trace(go.Scatter(x=fig_trend.data[0]['x'], y=fig_trend.data[0]['y'], showlegend=False), row=2, col=1) # type: ignore
    fig.add_trace(go.Scatter(x=fig_seasonal.data[0]['x'], y=fig_seasonal.data[0]['y'], showlegend=False), row=3, col=1) # type: ignore
    fig.add_trace(go.Scatter(x=fig_resid.data[0]['x'], y=fig_resid.data[0]['y'], showlegend=False), row=4, col=1) # type: ignore
    fig.update_xaxes(title_text='Data', row=4, col=1)
    fig.update_layout(title=title, font=dict(size=12))
    fig.update_layout(
        width=1200,
        height=600
    )
    return fig

def test_adfuller(df):
    result = adfuller(df.values)
    print('Teste ADf')
    print(f'Teste estatistico: {result[0]}')
    print(f'P-Value: {result[1]}')
    print(f'Valores criticos:')

    for key, value in result[4].items(): #type: ignore
        print(f'\t{key}: {value}')

def data_diff(df_ultimo_diff, df_ultimo_mean, df_ultimo_std):
    trace_diff = go.Scatter(x=df_ultimo_diff.index, y=df_ultimo_diff, mode='lines', name='Variação')
    trace_mean = go.Scatter(x=df_ultimo_mean.index, y=df_ultimo_mean, mode='lines', name='Média', line=dict(color='red'))
    trace_std = go.Scatter(x=df_ultimo_std.index, y=df_ultimo_std, mode='lines', name='Desvio Padrão', line=dict(color='green'))
    fig = go.Figure(data=[trace_diff, trace_mean, trace_std])
    fig.update_xaxes(title_text='Data')
    fig.update_yaxes(title_text='Último Fechamento')
    fig.update_layout(title='Variação, Média e Desvio Padrão do Último Fechamento', showlegend=True)
    return fig

def test_diff_adfuller(df):
    result = adfuller(df)
    print('Teste ADf')
    print(f'Teste estatistico: {result[0]}')
    print(f'P-Value: {result[1]}')
    print(f'Valores criticos:')
    for key, value in result[4].items(): #type: ignore
        print(f'\t{key}: {value}')

def autocorrelation_function(df, lag):
    return acf(df.dropna(), nlags=lag)

def partial_autocorrelation_function(df, lag):
    return pacf(df.dropna(), nlags=lag)

def mape_error(y_true, y_pred):
    n = len(y_true)
    mape = (np.abs((y_true - y_pred) / y_true).sum() / n)
    print(f"MAPE (Mean Absolute Percentage Error): {mape:.2%}")
    return mape

def wmape_error(y_true, y_pred):
    wmape = np.abs(y_true - y_pred).sum() / np.abs(y_true).sum()
    print(f"WMAPE (Weighted Mean Absolute Percentage Error): {wmape:.2%}")
    return wmape

def smape_error(y_true, y_pred):
    smape = (np.abs(y_pred - y_true) * 2 / (np.abs(y_pred) + np.abs(y_true))).mean()
    print(f"SMAPE (Symmetric Mean Absolute Percentage Error): {smape:.2%}")
    return smape

def acf_pacf(df_ultimo_diff):
    fig = sp.make_subplots(rows=1, cols=2, subplot_titles=['ACF - Último Fechamento', 'PACF - Último Fechamento'])
    trace_acf = go.Scatter(x=np.arange(len(autocorrelation_function(df_ultimo_diff, 14))),
                        y=autocorrelation_function(df_ultimo_diff, 14),
                        mode='lines',
                        name='ACF')
    trace_pacf = go.Scatter(x=np.arange(len(partial_autocorrelation_function(df_ultimo_diff, 14))),
                            y=partial_autocorrelation_function(df_ultimo_diff, 14),
                            mode='lines',
                            name='PACF')
    fig.add_trace(trace_acf, row=1, col=1)
    fig.add_trace(trace_pacf, row=1, col=2)
    hline = go.Scatter(x=[0, len(df_ultimo_diff) - 1],
                    y=[-1.96 / (np.sqrt((len(df_ultimo_diff) - 1))), -1.96 / (np.sqrt((len(df_ultimo_diff) - 1)))],
                    mode='lines',
                    line=dict(color='gray', dash='dash'),
                    showlegend=False)
    fig.add_trace(hline, row=1, col=1)
    fig.add_trace(hline, row=1, col=2)
    hline = go.Scatter(x=[0, len(df_ultimo_diff) - 1],
                    y=[1.96 / (np.sqrt((len(df_ultimo_diff) - 1))), 1.96 / (np.sqrt((len(df_ultimo_diff) - 1)))],
                    mode='lines',
                    line=dict(color='gray', dash='dash'),
                    showlegend=False)
    fig.add_trace(hline, row=1, col=1)
    fig.add_trace(hline, row=1, col=2)
    q_value = 0.915
    p_value = 0.915
    point_acf = go.Scatter(x=[q_value], y=[0.028], mode='markers', marker=dict(color='red'), name='Q')
    point_pacf = go.Scatter(x=[p_value], y=[0.028], mode='markers', marker=dict(color='red'), name='P')
    fig.add_trace(point_acf, row=1, col=1)
    fig.add_trace(point_pacf, row=1, col=2)
    fig.update_xaxes(title_text='Lag', range=[0, 15], row=1, col=1)
    fig.update_xaxes(title_text='Lag', range=[0, 15], row=1, col=2)
    fig.update_yaxes(title_text='Correlação', range=[-0.15, 0.1], row=1, col=1)
    fig.update_yaxes(title_text='Correlação', range=[-0.15, 0.1], row=1, col=2)
    fig.update_layout(title='ACF e PACF do Último Fechamento', showlegend=True)
    return fig

def plot_error(df, erro, label, title):
    df_sorted = df.sort_values(by=erro, ascending=True)
    num_models = len(df_sorted)
    custom_color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
                            '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78',
                            '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d2', '#c7c7c7',
                            '#dbdb8d', '#9edae5', '#aec7e8', '#ffbb78', '#98df8a', '#ff9896',
                            '#c5b0d5', '#c49c94', '#f7b6d2', '#c7c7c7']

    color_scale = custom_color_palette[:num_models][::-1]
    fig = px.bar(df_sorted, y=erro, x='model',
                labels={erro: label, 'model': 'Modelo'},
                title=title,
                text=df_sorted[erro].apply(lambda x: f'{x:.4%}'),
                color=df_sorted['model'],
                color_discrete_sequence=color_scale)
    fig.update_layout(
        yaxis_tickformat=".0%",
        yaxis_title=label,
        xaxis_title="Modelo",
        xaxis_categoryorder='total ascending',
        margin=dict(l=0, r=0, t=50, b=0),
        showlegend=False
    )
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_xaxes(showgrid=False)
    fig.update_layout(
        width=1100,
        height=400
    )
    return fig

def plot_acf_pacf(df):

    acf_values = sm.tsa.acf(df['Último'], fft=False)
    pacf_values = sm.tsa.pacf(df['Último'])
    
    fig = sp.make_subplots(rows=1, cols=2, subplot_titles=['ACF - Último Fechamento', 'PACF - Último Fechamento'])
    fig.add_trace(go.Bar(x=np.arange(len(df.index)), y=acf_values, name='ACF'),row=1, col=1)
    fig.add_trace(go.Bar(x=np.arange(len(df.index)), y=pacf_values, name='PACF'),row=1, col=2)

    # Atualizar layout e título
    fig.update_layout(title='ACF e PACF - Último Fechamento', xaxis_title='Lag', yaxis_title='Valor')
    
    return fig
    