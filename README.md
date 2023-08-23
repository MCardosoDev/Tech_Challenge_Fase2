# Tech_Challenge_Fase2
### Análise dos dados de fechamento diário da IBOVESPA, criação de um modelo preditivo e serie temporal para prever o fechamento da base.

> Para visualizar online
>
>> <https://techchallenge2.streamlit.app>
> 
> Para visualizar as análises em localhost rodar
>
>> **streamlit run Tech_Challenge2.py**
>
> Caso problemas com os arquivos de dados, ao rodar as células do Tech_Challenge2.ipynb e Tech_Challenge2_prophet.ipynb serão gerados os arquivos necessários para rodar o streamlit.
>

- Para o Tech_Challenge2.ipynb é utilizado um ambiente com Python 3.10.12 com virtualenv.

- Para o Tech_Challenge2_prophet.ipynb é utilizado um ambiente com Python 3.10.12 com conda.

- Utilizado ambientes deferentes pois o Prophet não é compatível o o kernel atual utilizado virtualenv e o Tensorflow não é compativel com o kernel atual utilizado no conda.

## Libs

- pandas
- numpy
- matplotlib
- seaborn
- plotly.express
- plotly.graph_objs
- plotly.express
- plotly.subplots
- plotly.graph_objs
- sklearn.preprocessing
  - MinMaxScaler
- statsmodels.tsa.stattools
  - adfuller
  - acf
  - pacf
- statsmodels.tsa.seasonal
  - seasonal_decompose
- statsmodels.tsa.statespace.sarimax
  - SARIMAX
- statsmodels.graphics.tsaplots
  - plot_acf
  - plot_pacf
- statsforecast
  - StatsForecast
- statsforecast.models
  - Naive
  - SeasonalNaive
  - SeasonalWindowAverage
  - WindowAverage
  - RandomWalkWithDrift
  - HistoricAverage
  - AutoARIMA
  - AutoETS
  - AutoCES
  - AutoTheta
  - SimpleExponentialSmoothing
  - SimpleExponentialSmoothingOptimized
  - SeasonalExponentialSmoothing
  - SeasonalExponentialSmoothingOptimized
  - Holt
  - HoltWinters
  - ADIDA
  - CrostonClassic
  - CrostonOptimized
  - CrostonSBA
  - IMAPA
  - TSB
  - MSTL
  - DynamicOptimizedTheta
  - GARCH
  - ARCH
- tensorflow.keras.models
  - Sequential
- tensorflow.keras.layers
  - LSTM
  - Dense
- prophet
  - Prophet
- prophet.plot
  - plot_plotly
  - plot_components_plotly
- pip install streamlit

## Dados usados no período entre 2003 e 2023

- Dados Históricos - Ibovespa.csv
  - Dados Históricos - Ibovespa
  - <https://br.investing.com/indices/bovespa-historical-data>

    |    Data     |  Último  | Abertura |  Máxima  |  Mínima  |  Vol.   |  Var%   |
    |:-----------:|:--------:|:--------:|:--------:|:--------:|:-------:|:-------:|
    | 09.03.2023  | 105.071  | 106.540  | 106.724  | 105.053  |  19.17M |  -1.38% |
    | 08.03.2023  | 106.540  | 104.228  | 106.721  | 104.228  |  15.90M |   2.22% |

