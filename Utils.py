import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def plot_pareto(data, bar_x, bar_y, scatter_x, scatter_y, title_x, title_y, mean):
    pareto = px.bar(
        data,
        x=bar_x,
        y=bar_y,
        color=bar_x,
        color_discrete_sequence=px.colors.qualitative.T10,
        template='plotly_white',
        text_auto='.2s' # type: ignore
    )

    pareto.update_layout(
        width=800,
        height=600
    )

    pareto.add_scatter(
        x=data[scatter_x],
        y=data[scatter_y], 
        mode='lines+text',
        name='Porcentagem Acumulada',
        text=[f'{x:.2f}%' for x in data[scatter_y]],
        textposition='top center',
        yaxis='y2',
        line={'color': 'red'}
    )

    pareto.update_layout(
        xaxis=dict(title=title_x),
        yaxis=dict(title=title_y),
        yaxis2=dict(
            title='Porcentagem Acumulada',
            overlaying='y',
            side='right',
            showgrid=False,
            range=[0, 100]
        )
    )

    pareto.update_traces(showlegend=False)
    pareto.add_shape(
        type='line',
        x0=0,
        x1=1,
        y0=80,
        y1=80,
        line=dict(color='white', width=1),
        xref='paper',
        yref='y2'
    )

    pareto.add_shape(
        type='line',
        x0=0,
        x1=1,
        y0=mean,
        y1=mean,
        line=dict(color='LightBlue', width=1),
        xref='paper'
    )

    pareto.add_annotation(
        x=4.5,
        y=mean,
        text= f"Média: {mean / 1000000:.2f}M",
        showarrow=True,
        arrowhead=1,
        ax=40,
        ay=-80,
        xref='x',
        font=dict(color='LightBlue')
    )

    return pareto

def plot_regressao_estimada(data, title, data_type, country_order=None):
    num_paises = math.ceil(len(data)/2)

    if country_order is not None:
        data = data.set_index('pais').loc[country_order].reset_index()

    regressao = make_subplots(rows=num_paises, cols=2, start_cell="bottom-left")
    regressao.update_layout(
        title=title,
        width=1300,
        height=num_paises*150,
        showlegend=False,
        colorway = [
            "#ADD8E6",   # LightBlue
            "#90EE90",   # LightGreen
            "#FFA500",   # Orange
            "#E6E6FA",   # Lavender
            "#D2B48C",   # Tan
            "#FF00FF",   # Magenta
            "#00FFFF",   # Cyan
            "#FFFFE0",   # LightYellow
            "#E0FFFF",   # LightCyan
            "#FFB6C1",   # LightPink
            "#D3D3D3",   # LightGray
            "#FFE4E1",   # MistyRose
            "#98FB98",   # PaleGreen
            "#E0FFFF",   # LightCyan
            "#FFF0F5",   # LavenderBlush
            "#FFDAB9",   # PeachPuff
            "#AFEEEE",   # PaleTurquoise
            "#FFC0CB",   # Pink
            "#F0FFF0",   # Honeydew
            "#FFE4B5",   # Moccasin
            "#FFFFF0"    # Ivory
        ]
    )

    for i, pais in enumerate(data.pais):
        row = num_paises - (i // 2)
        col = (i % 2) + 1
        trace = go.Scatter(x=data.columns[1:], y=data.iloc[i, 1:], name=pais)
        x = np.array(data.columns[1:], dtype=data_type)
        y = np.array(data.iloc[i, 1:], dtype=data_type)
        coef = np.polyfit(x, y, 1)
        line = coef[1] + coef[0] * x
        reg = go.Scatter(x=x, y=line, mode='lines', name='Regressão', line=dict(color='red'))
        regressao.add_trace(reg, row=row, col=col)
        regressao.add_trace(trace, row=row, col=col)
        regressao.update_xaxes(
            title_text=pais,
            row=row,
            col=col,
            title_standoff=10,
            tickfont=dict(size=8),
            title_font=dict(size=10),
            tickvals=data.columns[1:]
        )

    return regressao

def plot_consumo_projetado(data, title, country_order=None):
    if country_order is not None:
        data = data[data["pais"].isin(country_order)]
        data["pais"] = pd.Categorical(data["pais"], categories=country_order, ordered=True)
        data = data.sort_values("pais")

    consumo = go.Figure()
    consumo.add_trace(
        go.Bar(
            x=data.pais, 
            y=data["2020"],
            name="2020",
            marker=dict(color="lightblue"),
            text=data["2020"],
            textposition="auto"
        )
    )
    consumo.add_trace(
        go.Bar(
            x=data.pais,
            y=data["2025"],
            name="2025",
            marker=dict(color="lightgreen"),
            text=data["2025"],
            textposition="auto"
        )
    )
    
    for i, pais in enumerate(data.pais):
        diff = round(data["diferenca"].iloc[i], 2)
        consumo.add_annotation(
            x=pais,
            y=max(data["2020"].iloc[i], data["2025"].iloc[i]),
            text=f"Diferença: {diff}",
            showarrow=False,
            font=dict(color="white"),
            yshift=10
        )
    
    consumo.update_layout(
        title=title,
        xaxis_title="Países",
        barmode="group"
    )
    
    return consumo

def plot_comparacao(data1, data2, name1, name2, title, data_type, country):
    regressao = make_subplots(rows=1, cols=2, subplot_titles=(name1, name2))
    regressao.update_layout(
        title=title,
        width=1300,
        height=300,
        showlegend=False,
        colorway=[
            "#ADD8E6",   # LightBlue
            "#90EE90",   # LightGreen
            "#FFA500",   # Orange
            "#E6E6FA",   # Lavender
            "#D2B48C",   # Tan
            "#FF00FF",   # Magenta
            "#00FFFF",   # Cyan
            "#FFFFE0",   # LightYellow
            "#E0FFFF",   # LightCyan
            "#FFB6C1",   # LightPink
            "#D3D3D3",   # LightGray
            "#FFE4E1",   # MistyRose
            "#98FB98",   # PaleGreen
            "#E0FFFF",   # LightCyan
            "#FFF0F5",   # LavenderBlush
            "#FFDAB9",   # PeachPuff
            "#AFEEEE",   # PaleTurquoise
            "#FFC0CB",   # Pink
            "#F0FFF0",   # Honeydew
            "#FFE4B5",   # Moccasin
            "#FFFFF0"    # Ivory
        ]
    )

    for i, data in enumerate([data1, data2]):
        trace = go.Scatter(x=data.columns[1:], y=data.loc[data['pais'] == country].values[0][1:], name=country)
        x = np.array(data.columns[1:], dtype=data_type)
        y = np.array(data.loc[data['pais'] == country].values[0][1:], dtype=data_type)
        coef = np.polyfit(x, y, 1)
        line = coef[1] + coef[0] * x
        reg = go.Scatter(x=x, y=line, mode='lines', name='Regressão', line=dict(color='red'))
        regressao.add_trace(reg, row=1, col=i+1)
        regressao.add_trace(trace, row=1, col=i+1)
        regressao.update_xaxes(
            row=1,
            col=i+1,
            title_standoff=10,
            tickfont=dict(size=8),
            title_font=dict(size=10),
            tickvals=data.columns[1:]
        )
        regressao.update_yaxes(
            row=1,
            col=i+1,
            title_standoff=10,
            tickfont=dict(size=8),
            title_font=dict(size=10)
        )

    return regressao

def plot_per_anual(data):
    cores = {
        'Rússia': 'rgb(255, 0, 0)',     # Vermelho
        'Paraguai': 'rgb(200, 162, 200)',   # Lilás
        'Estados Unidos': 'rgb(0, 0, 255)',     # Azul
        'Reino Unido':'rgb(128, 0, 128)',   # Roxo
        'China': 'rgb(255, 255, 0)',   # Amarelo
        'Outros': 'rgb(128, 128, 128)'      # Cinza
    }
    data_percent=data.mul(100).div(data.sum(axis=1),axis=0)

    fig = go.Figure()

    for coluna in data.columns:
        fig.add_trace(go.Bar(
            x=data.index,
            y=data[coluna],
            name=coluna,
            marker=dict(color=cores[coluna]),
            text=data_percent[coluna].apply(lambda x: f'{x:.1f}%'),  
            textposition='auto' 
        ))

    fig.update_layout(
        title='Percentual de exportação anual dos países com maior influência',
        barmode='stack',
        legend=dict(x=1, y=1, orientation='v'),
        xaxis=dict(tickmode='linear'),
        height=800,
        yaxis_title='Valor (em milhões de U$)'
        
    )
    return fig

def plot_regressao1(data, title, data_type,country_order=None):
    # if country_order is not None:
    #     data = data.set_index('pais').loc[country_order].reset_index()

    regressao =go.Figure()
    regressao.update_layout(
        title=title,
        width=600,
        height=300,
        showlegend=False,
        colorway = [
            "#ADD8E6",   # LightBlue
            "#90EE90",   # LightGreen
            "#FFA500",   # Orange
            "#E6E6FA",   # Lavender
            "#D2B48C",   # Tan
            "#FF00FF",   # Magenta
            "#00FFFF",   # Cyan
            "#FFFFE0",   # LightYellow
            "#E0FFFF",   # LightCyan
            "#FFB6C1",   # LightPink
            "#D3D3D3",   # LightGray
            "#FFE4E1",   # MistyRose
            "#98FB98",   # PaleGreen
            "#E0FFFF",   # LightCyan
            "#FFF0F5",   # LavenderBlush
            "#FFDAB9",   # PeachPuff
            "#AFEEEE",   # PaleTurquoise
            "#FFC0CB",   # Pink
            "#F0FFF0",   # Honeydew
            "#FFE4B5",   # Moccasin
            "#FFFFF0"    # Ivory
        ]
    )

    for i, pais in enumerate(data.pais):
        trace = go.Scatter(x=data.columns[1:], y=data.iloc[i, 1:], name=pais)
        x = np.array(data.columns[1:], dtype=data_type)
        y = np.array(data.iloc[i, 1:], dtype=data_type)
        coef = np.polyfit(x, y, 1)
        line = coef[1] + coef[0] * x
        reg = go.Scatter(x=x, y=line, mode='lines', name='Regressão', line=dict(color='red'))
        regressao.add_trace(reg)
        regressao.add_trace(trace)
        regressao.update_xaxes(
            title_text=pais,
            title_standoff=10,
            tickfont=dict(size=8),
            title_font=dict(size=10),
            tickvals=data.columns[1:]
        )

    return regressao