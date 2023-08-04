# #%%
# import streamlit as st
# from Utils import plot_pareto

# #%%
# def main():
#     st.set_page_config(layout="wide")
#     st.title('Análise sobre a importação de vinhos e insights para melhorias')
#     tab0 = st.tabs(["Geral"])

#     with tab0:

#         '''
#         ## Análise dos Dados para Melhorar a Exportação de Vinhos

#         Esta análise tem como objetivo utilizar os dados disponíveis sobre a exportação de vinhos, dados econômicos, demográficos e o ranking de felicidade da população para fornecer insights sobre como melhorar a exportação de vinhos para diferentes países. Os dados foram obtidos de fontes confiáveis, como a Embrapa (Empresa Brasileira de Pesquisa Agropecuária), o Banco Mundial e o Relatório Mundial de Felicidade.
        
#         ## Dados Utilizados

#         1. **Exportação de Vinhos** - Os dados sobre a exportação de vinhos foram obtidos do Centro Nacional de Pesquisa de Uva e Vinho (CNPUV) da Embrapa. Esses dados fornecem informações relevantes sobre os países que podem ser alvos de exportação de vinhos.

#             - Link: [Exportação de Vinhos](http://vitibrasil.cnpuv.embrapa.br/index.php?opcao=opt_06)

#         2. **Dados Econômicos e Demográficos** - Para entender o contexto econômico e demográfico dos países, foram coletados os seguintes dados:

#             - **Produto Interno Bruto (PIB)**: O PIB é uma medida amplamente utilizada para avaliar o tamanho e o desempenho econômico de um país.
#                 - Link: [PIB dos Países](https://data.worldbank.org/indicator/NY.GDP.MKTP.CD)

#             - **Taxa de Inflação Anual**: A taxa de inflação anual é um indicador que mede a variação dos preços ao longo do tempo e é relevante para entender a estabilidade econômica dos países.
#                 - Link: [Taxa de Inflação Anual dos Países](https://data.worldbank.org/indicator/FP.CPI.TOTL.ZG)

#             - **Proporção do Comércio Internacional**: Essa proporção indica a importância do comércio internacional em relação ao PIB de um país, o que pode indicar o quão aberto ele é para o comércio exterior.
#                 - Link: [Proporção do Comércio Internacional em relação ao PIB dos Países](https://data.worldbank.org/indicator/NE.TRD.GNFS.ZS)
#         '''

#         # - **População Total**: O número total de pessoas em um país é relevante para entender o potencial mercado consumidor e a demanda por vinhos.
#         #     - Link: [População Total dos Países](https://data.worldbank.org/indicator/SP.POP.TOTL)

#         # - **Taxa de Desemprego**: A taxa de desemprego é um indicador importante para avaliar a situação econômica e a capacidade de compra dos consumidores em um país.
#         #     - Link: [Taxa de Desemprego Total dos Países](https://data.worldbank.org/indicator/SL.UEM.TOTL.ZS)
        
#         # 3. **Ranking de Felicidade da População** - O ranking de felicidade é um indicador que avalia o bem-estar e a satisfação da população em diferentes países.

#         # - Link: [World Happiness Report](https://www.kaggle.com/datasets/unsdsn/world-happiness)
        
#         '''
#         3. **Consumo de Álcool** - Os dados sobre o consumo de álcool per capita foram obtidos da Organização Mundial da Saúde (OMS). Esses dados fornecem informações sobre a quantidade de álcool (incluindo registro e não registro) consumida por pessoa com idade igual ou superior a 15 anos em diferentes países, juntamente com projeções com intervalo de confiança de 95% para os anos de 2020 e 2025. Isso pode ajudar a entender os hábitos de consumo de bebidas alcoólicas e o potencial de mercado para vinhos em cada país.

#            - Link: [WHO](https://www.who.int/data/gho/data/indicators/indicator-details/GHO/alcohol-total-(recorded-unrecorded)-per-capita-(15-)-consumption-with-95-ci-projections-to-2020-and-2025)

#         4. **Dados sobre o consumo de vinho** - A OIV fornece estatísticas abrangentes sobre o consumo de vinho em diferentes países. Esses dados podem incluir informações sobre o consumo per capita, o volume total de consumo, entre outros aspectos relevantes para entender o mercado do vinho em cada país.

#            - Link: [OIV](https://www.oiv.int/en/statistics)

#         ## Análise e Insights

#         Com base nos dados coletados, é possível realizar uma análise detalhada para melhorar e exportação de vinhos.
#         '''

#     with tab1:
#         valor = "{:,.2f}".format(dataset_exp.Valor.sum())
#         quantidade = "{:,.2f}".format(dataset_exp.Quantidade.sum())
#         st.markdown('***Valores totais de exportação de vinhos no período de 15 anos entre 2007 a 2021***')
#         st.markdown(
#             f"""
#             <div style="padding: 20px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); background-color: #f5f5f5; color: #000000; text-align: center;">
#                 <p style="font-size: 24px; font-weight: bold; display: inline; margin-right: 400px;">US$ {valor}</p>
#                 <p style="font-size: 24px; font-weight: bold; display: inline;">{quantidade} L</p>
#             </div>
#             """,
#             unsafe_allow_html=True
#         )
#         df = pd.DataFrame(dataset_exp)
#         st.write("\n")
#         st.markdown('##### Dados sobre exportação de vinhos no período')
#         st.dataframe(df, use_container_width=True)

#     with tab2:

#         '''
#         ##### Países que mais impactam na exportação de vinhos - Princípio 80/20

#         ***Valores correspondentes ao período de 15 anos entre 2007 a 2021***
#         '''

#         st.markdown('###### Quantidade (L)')
#         st.plotly_chart(
#             plot_pareto(
#                 dataset_Qexp_pareto.query("Porcentagem_acumulada_quantidade < 92"),
#                 "Pais_destino",
#                 "Quantidade",
#                 "Pais_destino",
#                 "Porcentagem_acumulada_quantidade",
#                 "País",
#                 "Quantidade",
#                 dataset_Qexp_pareto.Quantidade.mean()
#             ),  use_container_width = True
#         )

#         '''
#             Analisando o gráfico de pareto para volume total de vinho exportado pelo Brasil entre 2007 e 2021, nota-se que Rússia, Paraguai e Estados Unidos são responsáveis por 
#             mais de 80% desse volume.
        
#         '''

#         st.write("\n")
#         st.markdown('###### Valor (US$)')
#         st.plotly_chart(
#             plot_pareto(
#                 dataset_exp_pareto.query("Porcentagem_acumulada_valor < 81"),
#                 "Pais_destino",
#                 "Valor",
#                 "Pais_destino",
#                 "Porcentagem_acumulada_valor",
#                 "País",
#                 "Valor",
#                 dataset_exp_pareto.Valor.mean()
#             ),  use_container_width = True
#         )
        
#         '''
#             Ao observar o gráfico para valor total de vendas com a exportação de vinho nesse mesmo período, percebe-se que o Paraguai ultrapassa o percentual da Rússia, e
#             que Reino Unido passam a ter maior representatividade no percentual. O que indica que nesse país o vinho brasileiro é exportado por um valor superior.
        
#         '''
        
#     with tab3:
#         # '''
#         # ##### Valores por país para os países que mais impactam na exportação de vinhos

#         # ***Valores correspondentes ao período de 15 anos entre 2007 a 2021***
#         # '''
    
#         st.plotly_chart(
#             plot_per_anual(dataset_anos),
#             use_container_width = True
#         )

#         '''
#         O gráfico acima ilustra como foi a distribuição dos valores exportados de vinho pelo Brasil entre 2007 e 2021. Alguns pontos interessantes ilustrados no gráfico são:\n
#         - Os picos de exportação em 2009 e 2013
#         - Queda no total de exportações em 2010
#         - Crescimento das exportações para o Paraguai
                
#         #### Os picos de exportação em 2009 e 2013
#         Observando a altura total das barras, constata-se que houve dois picos, um no ano de 2009 e outro no ano de 2013. Avaliando o percentual de influência dos países na exportação, 
#         percebe-se que ambos os picos ocorreram devido ao crescimento das vendas para a **Rússia**. \n
#         Analisando abaixo o comportamento dos valores exportados para a Rússia ao longo dos anos, é possível notar esses picos. Contudo, após 2013, as vendas para a Rússia caíram bruscamente.
#         '''

#         st.plotly_chart(
#             plot_regressao1(
#                 dataset_exportacao.set_index('pais').query("pais=='Rússia'").reset_index(),
#                 'Valores de exportação para Rússia (US$)',
#                 int
#             ),
#             use_container_width = True
#         )

#         '''
#         "Após 18 anos de negociações, a adesão da Rússia à OMC foi aceita em 2011."
#         '''
#         st.markdown(
#         '<span class="small-font">Fonte: https://g1.globo.com/economia/noticia/2011/11/russia-entra-na-omc-apos-18-anos-de-negociacoes.html',
#         unsafe_allow_html=True
#         )
#         '''
#         "Após a anexação da Crimeia em março de 2014 e o envolvimento da Rússia no conflito em curso na Ucrânia, os Estados Unidos, a UE, o Canadá, o Japão e outros países impuseram sanções aos setores financeiro, energético e de defesa da Rússia."
#         '''
#         st.markdown(
#         '<span class="small-font">Fonte: https://g1.globo.com/bom-dia-brasil/noticia/2014/03/ue-e-eua-ampliam-sancoes-contra-russia-apos-anexacao-da-crimeia.html',
#         unsafe_allow_html=True
#         )
#         '''
#         #### Queda no total de exportações em 2010
#         Analisando abaixo os valores para os principais exportadores ao longo dos anos, juntamente com uma linha de tendência. Percebe-se que a crise global impactou negativamente as vendas, 
#         exceto para Rússia e China, que tiveram alta em 2009.\n
#         "A crise financeira de 2008 ocorreu devido a uma bolha imobiliária nos Estados Unidos, causada pelo aumento nos valores imobiliários, que não foi acompanhado por um aumento de renda da população."
#         '''

#         # st.markdown('###### Valor (US$)')
#         st.plotly_chart(
#             plot_regressao_estimada(
#                 dataset_exportacao[dataset_exportacao['pais'].isin(exp_ordem_pareto)],
#                 'Valores de exportação para os principais países (US$) no período',
#                 int,
#                 exp_ordem_pareto

#             ),
#             use_container_width = True
#         )

#         '''
#         Como a alta exportação para China e Rússia, o mercado de exportação de vinho brasileiro não refletiu a crise em 2009.
#         Entretanto em 2010 as exportações também caíram para esses países, resultando na queda dos valores em 2010.\n
        
#         #### Crescimento das exportações para o Paraguai
#         Nos gráficos acima também é possível perceber que as exportações para **Paraguai** vêm crescendo, principalmente a partir de 2014, ano no qual ultrapassa a Rússia como país 
#         que mais importa vinho do Brasil. \n
#         "Com o sucesso dos vinhos doces e que a maior parte do mercado é de vinhos econômicos o vinho espumante brasileiro é considerado como um produto de boa qualidade e com demanda crescente, principalmente na fronteira 
#         e existe a possibilidade de escalar os hábitos e o paladar de um segmento médio Premium que já consome vinhos brasileiros com frequência."\n
#         '''

#         st.markdown(
#         '<span class="small-font">Fonte: https://www.gov.br/empresas-e-negocios/pt-br/invest-export-brasil/exportar/conheca-os-mercados/pesquisas-de-mercado/estudo-de-mercado.pdf/Paraguai2021.pdf</span>',
#         unsafe_allow_html=True
#         )
#         st.plotly_chart(
#             plot_regressao1(
#                 dataset_exportacao.set_index('pais').query("pais=='Paraguai'").reset_index(),
#                 'Valores de exportação para Paraguai (US$)',
#                 int
#             ),
#             use_container_width = True
#         )

#         '''
#         Outras tendências identificadas são: a de aumento das exportações para **China** e a de estabilidade para **Estados Unidos** e **Reino Unido**.\n
#         "O total das exportações(geral) do Brasil para a China em 2022 representou mais que o dobro do total embarcado para os Estados Unidos, o segundo maior destino das exportações brasileiras no ano passado."
       
#         '''

#         st.markdown(
#         '<span class="small-font">Fonte: https://investnews.com.br/economia/participacao-da-china-na-exportacao-do-brasil-cresceu-56-em-10-anos/#:~:text=As%20exportações%20do%20Brasil%20para,o%20Ministério%20das%20Relações%20Exteriores.</span>',
#         unsafe_allow_html=True
#         )
#         st.plotly_chart(
#             plot_regressao1(
#                 dataset_exportacao.set_index('pais').query("pais=='China'").reset_index(),
#                 'Valores de exportação para China (US$)',
#                 int
#             ),
#             use_container_width = True
#         )
#         st.markdown("---")
#         st.markdown(
#             """
#             <style>
#             .small-font {
#              font-size: 12px;
#             }
#             </style>
#             """,
#             unsafe_allow_html=True
#         )
#         st.markdown(
#         '''<span class="small-font">
#             Para estimativa de Regressão foi utilizado para calcular o coeficientes do polinômio ajustado com a função polyfit do numpy para grau 1 que utiliza o método dos mínimos quadrados, que minimiza o erro quadrático.
#              </span>''',
#         unsafe_allow_html=True
#         )
#         st.markdown(
#         '<span class="small-font">E = Σᵢ(yᵢ - p(xᵢ))² </span>',
#         unsafe_allow_html=True
#         )
#         st.markdown(
#         '<span class="small-font">p(x) = p[0] * xᵈᵉᵍ + p[1] * xᵈᵉᵍ⁻¹ + ... + p[deg-1] * x + p[deg] </span>',
#         unsafe_allow_html=True
#         )
    
#     with tab4:
          
#         '''

#         Devido a mudança do perfil de exportação ao longo dos 15 anos e visando entender como está o comportamento em um período mais recente, é válido avaliar o gráfico de pareto para o período de 5 anos entre 2016 
#         e 2021 abaixo.
#         '''
#         st.write("\n")
#         st.markdown('###### Valor Total de Vendas por países responsáveis por até 90% da vendas no período de 2016 a 2021')
#         st.plotly_chart(
#             plot_pareto(
#                 dataset_exp_pareto_5.query("Porcentagem_acumulada_valor < 90").reset_index(),
#                 "Pais_destino",
#                 "Valor",
#                 "Pais_destino",
#                 "Porcentagem_acumulada_valor",
#                 "País",
#                 "Valor (U$)",
#                 dataset_exp_pareto_5.Valor.mean()
#             ),  use_container_width = True
#         )
#         st.write("\n")

#         '''        
#         Nota-se um perfil muito diferente do observado para o período de 15 anos. Sendo a maior diferença a pouca expressividade da Rússia, confirmando a tendência de queda 
#         observada nos gráficos anteriores.\n
#         Uma novidade que aparece no perfil mais recente é o **Haiti**, que aparece como um dos países do grupo com 80% da influência no valor exportado, ultrapassando o Reino Unido. 
#         Dando um pouco mais de atenção a esse país, abaixo encontra-se o gráfico dos valores exportados de vinho para Haiti ao longo dos 15 anos.
#         '''

#         st.plotly_chart(
#             plot_regressao1(
#                 dataset_exportacao.set_index('pais').query("pais=='Haiti'").reset_index(),
#                 'Valores de exportação para Haiti (US$)',
#                 int
#             ),
#             use_container_width = True
#         )
        
#         '''      
#         Observa-se que os valores das exportações para o Haiti apresentou uma grande crescente a partir de 2009. Sendo assim, Haiti é um país de que merece atenção dos investidores com 
#         grande potencial de exportação para os próximos anos.

#         "Cada vez mais os empresários brasileiros começam a considerar as exportações como uma decisão estratégica importante para o desenvolvimento dos seus negócios.
#         Os principais produtos brasileiros exportados para Haiti são em primeiro lugar - Produtos Alimentícios e Animais Vivos."
#         '''

#         st.markdown(
#         '<span class="small-font">Fonte: https://www.fazcomex.com.br/comexstat/america-central/exportacao-haiti/.</span>',
#         unsafe_allow_html=True
#         )
#         st.markdown("---")
#         st.markdown(
#             """
#             <style>
#             .small-font {
#              font-size: 12px;
#             }
#             </style>
#             """,
#             unsafe_allow_html=True
#         )
#         st.markdown(
#         '''<span class="small-font">
#             Para estimativa de Regressão foi utilizado para calcular o coeficientes do polinômio ajustado com a função polyfit do numpy para grau 1 que utiliza o método dos mínimos quadrados, que minimiza o erro quadrático.
#              </span>''',
#         unsafe_allow_html=True
#         )
#         st.markdown(
#         '<span class="small-font">E = Σᵢ(yᵢ - p(xᵢ))² </span>',
#         unsafe_allow_html=True
#         )
#         st.markdown(
#         '<span class="small-font">p(x) = p[0] * xᵈᵉᵍ + p[1] * xᵈᵉᵍ⁻¹ + ... + p[deg-1] * x + p[deg] </span>',
#         unsafe_allow_html=True
#         )
        
#     with tab5:

#         '''
#         ##### Valores por país dos que mais importamos vinhos

#         ***Valores correspondentes ao período de 15 anos entre 2007 a 2021***
#         '''

#         st.markdown('###### Valor US$')
#         st.plotly_chart(
#             plot_pareto(
#                 dataset_imp_pareto.query("Porcentagem_acumulada_valor < 99 or Porcentagem_acumulada_quantidade < 99"),
#                 "Pais_destino",
#                 "Valor",
#                 "Pais_destino",
#                 "Porcentagem_acumulada_valor",
#                 "País",
#                 "Valor",
#                  dataset_imp_pareto.Valor.mean()
#             ),  use_container_width = True
#         )
#         st.plotly_chart(
#             plot_regressao_estimada(
#                 dataset_importacao[dataset_importacao['pais'].isin(imp_ordem_pareto)],
#                 'Valores de importação para os principais países',
#                 int,
#                 imp_ordem_pareto
#             ),
#             use_container_width = True
#         )

#         '''
#         Com a analise os dados sobre as importações no período e os países dos quais mais importamos vinho através da analise de Pareto. 
#         E a analise individual sobre a importação ao longo do período.\n
#         Foram importantes para entender a relação entre a importação e exportação desses países e que a importação tem uma correlação negativa com a exportação. 
#         Consumimos mais vinhos desses países do que eles consomes os vinhos brasileiro, principalmente pela tradição e qualidade na fabricação.\n
#         Dessa lista vale aprofundar a analise no Estados Unidos que mesmo estando como um país responsável pela maior parte da importação, 
#         é uma país com uma relevância importante da exportação e um mercado muito importante para apenas ser retirado da analise final.
#         '''

#         st.plotly_chart(
#             plot_comparacao(
#                 dataset_exportacao.query("pais == 'Estados Unidos'"),
#                 dataset_importacao.query("pais == 'Estados Unidos'"),
#                 'Exportação',
#                 'Importação',
#                 'Regressão de Exportação e Importação Estados Unidos',
#                 int,
#                 'Estados Unidos'
#             ),
#             use_container_width = True
#         )
#         st.markdown("---")
#         st.markdown(
#             """
#             <style>
#             .small-font {
#              font-size: 12px;
#             }
#             </style>
#             """,
#             unsafe_allow_html=True
#         )
#         st.markdown(
#         '''<span class="small-font">
#             Para estimativa de Regressão foi utilizado para calcular o coeficientes do polinômio ajustado com a função polyfit do numpy para grau 1 que utiliza o método dos mínimos quadrados, que minimiza o erro quadrático.
#              </span>''',
#         unsafe_allow_html=True
#         )
#         st.markdown(
#         '<span class="small-font">E = Σᵢ(yᵢ - p(xᵢ))² </span>',
#         unsafe_allow_html=True
#         )
#         st.markdown(
#         '<span class="small-font">p(x) = p[0] * xᵈᵉᵍ + p[1] * xᵈᵉᵍ⁻¹ + ... + p[deg-1] * x + p[deg] </span>',
#         unsafe_allow_html=True
#         )
#     with tab6:

#         '''
#         ##### Valores analisados dos países que mais impactam na exportação de vinhos

#         ***Valores correspondentes ao período de 15 anos entre 2007 a 2021***
#         '''

#         st.markdown('### Econômicos - Banco Mundial')
#         st.markdown('###### Valor US$')

#         '''
#         Notamos que todos tem uma linha de crescimento no PIB
#         '''

#         st.plotly_chart(
#             plot_regressao_estimada(
#                 dataset_pib[dataset_pib['pais'].isin(ordem)],
#                 'PIB',
#                 float,
#                 ordem
#             ),
#             use_container_width = True
#         )

#         '''
#         Em relação a inflação com exceção do Haiti que tem um crescimento histórico mas com uma queda no ultimo ano, temos uma linha de queda histórica nos países mesmo com um aumento após o Covid19.
#         '''

#         st.plotly_chart(
#             plot_regressao_estimada(
#                 dataset_inflation[dataset_inflation['pais'].isin(ordem)],
#                 'Inflação',
#                 float,
#                 ordem
#             ),
#             use_container_width = True
#         )

#         '''
#         Notamos que o período do Covid19 impactou na relação comercial entre países mas 2021 notamos uma melhora em alguns países, mesmo Reino Unido e Haiti que não notamos o aumento ele vem de um crescimento hatórico indicando que podemos esperar uma melhora também.
#         '''

#         st.plotly_chart(
#             plot_regressao_estimada(
#                 dataset_trade[dataset_trade['pais'].isin(ordem)],
#                 'Comércio internacional',
#                 float,
#                 ordem
#             ),
#             use_container_width = True
#         )
#         # st.plotly_chart(
#         #     plot_regressao_estimada(
#         #         dataset_population[dataset_population['pais'].isin(exp_ordem_pareto)],
#         #         'População dos países responsáveis por 80% da exportação',
#         #         int,
#         #         exp_ordem_pareto
#         #     ),
#         #     use_container_width = True
#         # )
#         # st.plotly_chart(
#         #     plot_regressao_estimada(
#         #         dataset_unemployment[dataset_unemployment['pais'].isin(exp_ordem_pareto)],
#         #         'Desemprego dos países responsáveis por 80% da exportação',
#         #         float,
#         #         exp_ordem_pareto
#         #     ),
#         #     use_container_width = True
#         # )
#         # st.plotly_chart(
#         #     plot_regressao_estimada(
#         #         dataset_wht[dataset_wht['pais'].isin(exp_ordem_pareto)],
#         #         'World Happiness dos países responsáveis por 80% da exportação',
#         #         int,
#         #         exp_ordem_pareto
#         #     ),
#         #     use_container_width = True
#         # )
#         st.markdown('### Consumo de álcool - WHO/OIV')

#         '''
#         ##### Valores de consumo de vinho e álcool dos países que mais impactam na exportação de vinhos\n
#         ***Valores sobre o consumo de vinho correspondentes ao período de 15 anos entre 2007 a 2021***\n
#         Com a analise dos próximos gráficos nota-se que os dados da Organização Internacional de Vinha e Vinho apresentam um aumento de consumo de vinho no ultimo com exceção da China.\n
#         Mas ao analisar em conjunto com os dados de projeção de consumo de álcool para 2025 da Organização Mundial da Saúde esperamos um aumento no consumo de vinho no período
#         '''

#         st.markdown('###### Valor US$')
#         st.plotly_chart(
#             plot_regressao_estimada(
#                 dataset_consumo_vinho[dataset_consumo_vinho['pais'].isin(ordem)],
#                 'Consumo de vinho',
#                 int,
#                 ordem
#             ),
#             use_container_width = True
#         )

#         '''
#         ***Valores sobre o consumo de álcool correspondentes ao fato e projeção com intervalo de confiança de 95% para os anos de 2020 e 2025***
#         '''

#         st.plotly_chart(
#             plot_consumo_projetado(
#                 dataset_consumo[dataset_consumo['pais'].isin(ordem)],
#                 'Diferença entre a projeção e o consumo atual de álcool',
#                 ordem
#             ),
#             use_container_width = True
#         )

#         '''
#             Dados disponíveis no **World Health Organization**\n
#             Diferença corresponde ao valor da projeção para 2025 menos valor fato de 2020
#         '''

#         st.markdown("---")
#         st.markdown(
#             """
#             <style>
#             .small-font {
#              font-size: 12px;
#             }
#             </style>
#             """,
#             unsafe_allow_html=True
#         )
#         st.markdown(
#             '''<span class="small-font">
#             Para estimativa de Regressão foi utilizado para calcular o coeficientes do polinômio ajustado com a função polyfit do numpy para grau 1 que utiliza o método dos mínimos quadrados, que minimiza o erro quadrático.
#              </span>''',
#             unsafe_allow_html=True
#         )
#         st.markdown(
#             '<span class="small-font">E = Σᵢ(yᵢ - p(xᵢ))² </span>',
#         unsafe_allow_html=True
#         )
#         st.markdown(
#             '<span class="small-font">p(x) = p[0] * xᵈᵉᵍ + p[1] * xᵈᵉᵍ⁻¹ + ... + p[deg-1] * x + p[deg] </span>',
#         unsafe_allow_html=True
#         )

#     with tab7:
        
#         '''
#         #### Resultados da análise
        
#         Primeiramente, é fundamental ressaltar que a situação geopolítica tem um impacto significativo no comércio internacional, e o setor vinícola não é exceção. 
#         Tensões políticas e conflitos podem conduzir a restrições comerciais, tarifas altas e, em algumas situações, até a embargos totais. 
#         Além disso, essas circunstâncias podem desestabilizar a economia de um país, influenciando a demanda por produtos importados como o vinho.

#         Com a análise dos dados de exportação, principalmente financeiros, notamos que, apesar da crise global em 2019-2020 devido à Covid-19,
#         observamos um cenário favorável para a economia em 2021, principalmente em alguns países na exportação de vinhos.\n
#         Montamos uma lista dos países mais favoráveis para a exportação de vinhos brasileiros nos próximos anos:

#         - **Paraguai**        
#         - **Estados Unidos**
#         - **China**
#         - **Haiti**
#         - **Reino Unido**
#         - **Rússia**

#         Excluímos os países analisados que são responsáveis pelo maior volume importado mantendo apenas os Estados Unidos.\n
#         Mesmo com uma queda na exportacao no período Estados Unidos e Rússia detém mercados muito importantes para exportação em um nível global.

#         No contexto do conflito russo-ucraniano que se iniciou em 2022, a exportação de vinhos para esses países pode se tornar uma tarefa complexa. 
#         Desafios logísticos como restrições de transporte e bloqueios podem se apresentar como obstáculos significativos. 
#         No entanto, o Brasil poderia encontrar uma janela de oportunidade caso estes países busquem alternativas aos vinhos europeus devido a alianças políticas.

#         Entretanto, informações recentes divulgadas pela imprensa portuguesa apontam que a Rússia triplicou a importação de vinhos portugueses, 
#         o que pode sugerir um alinhamento estratégico com menor abertura para o mercado brasileiro. 

#         Porem a Rússia como Reino Unido tem um aumento projetado no consumo de álcool para 2025, 
#         e ambos estão em crescimento no consumo de vinho nos últimos anos, sendo mercados com importância global para o comercio.
 
#         A China, apesar de ter uma queda na exportação de vinho em 2021, é um país com um mercado de grande potencial a ser explorado. 
#         Apresenta um aumento histórico no PIB e queda histórica na inflação, mesmo durante a Covid-19.

#         Expandindo a análise para outras esferas da geopolítica, as tensões entre China e EUA também podem criar tanto oportunidades quanto desafios para o Brasil. 
#         Os EUA têm mostrado uma tendência de crescimento na importação e consumo de bebidas alcoólicas, incluindo o vinho. 
#         Por outro lado, um documentário recente destacou o crescimento do setor vinícola chinês com foco na exportação. 
#         Considerando este cenário, pode ser estrategicamente mais vantajoso para o vinho brasileiro competir por espaço em 
#         mercados de países não diretamente envolvidos na tensão sino-americana, 
#         ou em países próximos como Chile, Uruguai e Argentina que importam uma quantidade alta de vinhos muito consumidos no 
#         Brasil mas podemos explorar a exportacao pela facilidade de logística utilizando a fronteira.

#         O Paraguai é um país de fronteira e com um paladar que propicia a exportação de vinhos brasileiros, 
#         que indica porque é um país que esta no topo de importacao de vinhos brasileiros.

#         O Haiti é um país que, apesar de ter um clima mais quente, teve o maior aumento na exportação de vinhos brasileiros. 
#         É o país com o maior aumento histórico do PIB da lista, mesmo com a Covid-19. 
#         De acordo com a OMS e a OIV é projetado um aumento no consumo de álcool e um aumento histórico no consumo de vinho, 
#         tendo assim o ótimo mercado a ser explorado.

#         Por fim, é essencial lembrar que estes são apenas cenários hipotéticos e a realidade pode divergir dependendo de uma ampla variedade de fatores. 
#         A geopolítica global é uma tapeçaria complexa e imprevisível, é preciso que estejamos preparados para nos adaptar rapidamente às mudanças no mercado.
#         '''

# if __name__ == "__main__":
#     main()