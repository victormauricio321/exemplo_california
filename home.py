import geopandas as gpd
import numpy as np
import pandas as pd
import pydeck as pdk
import shapely
import streamlit as st

from joblib import load

from notebooks.src.config import DADOS_GEO_MEDIAN, DADOS_LIMPOS, MODELO_FINAL


@st.cache_data
def carregar_dados_limpos():
    return pd.read_parquet(DADOS_LIMPOS)


@st.cache_data
def carregar_dados_geo():
    gdf_geo = gpd.read_parquet(DADOS_GEO_MEDIAN)

    # Explode MultiPolygons into individual polygons
    gdf_geo = gdf_geo.explode(ignore_index=True)

    # Function to check and fix invalid geometries
    def fix_and_orient_geometry(geometry):
        if not geometry.is_valid:
            geometry = geometry.buffer(0)  # Fix invalid geometry
        # Orient the polygon to be counter-clockwise if it's a Polygon or MultiPolygon
        if isinstance(
            geometry, (shapely.geometry.Polygon, shapely.geometry.MultiPolygon)
        ):
            geometry = shapely.geometry.polygon.orient(geometry, sign=1.0)
        return geometry

    # Apply the fix and orientation function to geometries
    gdf_geo["geometry"] = gdf_geo["geometry"].apply(fix_and_orient_geometry)

    # Extract polygon coordinates
    def get_polygon_coordinates(geometry):
        return (
            [[[x, y] for x, y in geometry.exterior.coords]]
            if isinstance(geometry, shapely.geometry.Polygon)
            else [
                [[x, y] for x, y in polygon.exterior.coords]
                for polygon in geometry.geoms
            ]
        )

    # Apply the coordinate conversion and store in a new column
    gdf_geo["geometry"] = gdf_geo["geometry"].apply(get_polygon_coordinates)

    return gdf_geo


@st.cache_resource    
def carregar_modelo():
    return load(MODELO_FINAL)



df = carregar_dados_limpos()
gdf_geo = carregar_dados_geo()
modelo = carregar_modelo()





st.title('Previsão de preços de imóveis')

condados = sorted(gdf_geo['name'].unique()) # passar para o tipo lista, pois o 'st.selectbox()' recebe listas

coluna1, coluna2 = st.columns(2) # realizando unpack de tupla (CRIANDO AS COLUNAS DA ESQUERDA E DA DIREITA DO SITE)

with coluna1:

    with st.form(key='formulario'):

    
        selecionar_condado = st.selectbox('Condado', condados)
        
        
        # as 13 features (colunas que não são a target), são os valores que preciso entrar para que ele me dê a previsão
        # criando as variáveis para inserir os valores necessários para realizar previsões com o nosso modelo:
        
        longitude = gdf_geo.query('name == @selecionar_condado')['longitude'].values # pesquisar no GeoDataFrame chamado 'gdf_geo' utilizando o '.query()', na coluna 'name', valor igual a variável 'selecionar_condado', e filtrar pelo 'longitude' e retornar os valores pelo atributo '.values'
        latitude = gdf_geo.query('name == @selecionar_condado')['latitude'].values
        
        housing_median_age = st.number_input('Idade do imóvel', value=10, min_value=1, max_value=50)
        
        total_rooms = gdf_geo.query('name == @selecionar_condado')['total_rooms'].values
        total_bedrooms = gdf_geo.query('name == @selecionar_condado')['total_bedrooms'].values
        population = gdf_geo.query('name == @selecionar_condado')['population'].values
        households = gdf_geo.query('name == @selecionar_condado')['households'].values
        
        median_income = st.slider('Renda média (milhares de US$)', 5.0, 100.0, 45.0, 5.0)
        
        ocean_proximity = gdf_geo.query('name == @selecionar_condado')['ocean_proximity'].values
        
        bins_income = [0, 1.5, 3, 4.5, 6, np.inf]
        median_income_cat = np.digitize(median_income / 10, bins=bins_income) # 'np.digitize()' retorna um intervalo pré definido a partir de um array
        
        rooms_per_household = gdf_geo.query('name == @selecionar_condado')['rooms_per_household'].values
        bedrooms_per_rooms = gdf_geo.query('name == @selecionar_condado')['bedrooms_per_rooms'].values
        population_per_household = gdf_geo.query('name == @selecionar_condado')['population_per_household'].values
        
        
        
        # agora podemos reunir tudo isso em um DataFrame
        # e a melhor forma de montar um DataFrame é através de um dicionário (por ter 'chave' e 'valor')
        # as chaves do dicionário vão virar o nome das colunas do DataFrame
        
        
        entrada_modelo = {
            'longitude': longitude,
            'latitude': latitude,
            'housing_median_age': housing_median_age,
            'total_rooms': total_rooms,
            'total_bedrooms': total_bedrooms,
            'population': population,
            'households': households,
            'median_income': median_income / 10,
            'ocean_proximity': ocean_proximity,
            'median_income_cat': median_income_cat,
            'rooms_per_household': rooms_per_household,
            'bedrooms_per_rooms': bedrooms_per_rooms,
            'population_per_household': population_per_household,
        }
        
        # criando o DataFrame
        
        df_entrada_modelo = pd.DataFrame(entrada_modelo) # 'index=[0]' para o streamlit não acusar um erro dizendo que falta um índice para o DataFrame
        
        botao_previsao = st.form_submit_button('Prever preço')
    
    if botao_previsao:
        preco = modelo.predict(df_entrada_modelo)
        st.metric(label='Preço previsto (US$): ', value=f'{preco[0][0]:.2f}')
    

with coluna2:

    view_state = pdk.ViewState(
        latitude=float(latitude[0]),
        longitude=float(longitude[0]),
        zoom=5,
        min_zoom=5,
        max_zoom=15,
    )

    # criando camadas para o nosso mapa
    # o sistema de cores que o pydeck utiliza é o RGB
    polygon_layer = pdk.Layer(
        'PolygonLayer', # nome da camada
        data=gdf_geo[['name', 'geometry']], # de onde está vindo os dados (carregar_dados_geo()), do dataframe DADOS_GEO_MEDIAN
        get_polygon='geometry', # onde estarão as informações para fazer a figura
        get_fill_color=[0, 0, 255, 100], # R, G, B e um alpha de transparência
        get_line_color=[255, 255, 255],
        get_line_width=700,
        pickable=True, # passar a propriedade de transformar algo em selecionável
        auto_highlight=True, # botar destaque quando passar o mouse por cima
        
    )

    condado_selecionado = gdf_geo.query('name == @selecionar_condado')
    
    highlight_layer = pdk.Layer(
        'PolygonLayer',
        data=condado_selecionado[['name', 'geometry']],
        get_polygon='geometry',
        get_fill_color=[255, 0, 0, 100], # R, G, B e um alpha de transparência
        get_line_color=[0, 0, 0],
        get_line_width=1000,
        pickable=True,
        auto_highlight=True,
    )
    
    # fazer aparecer o nome do condado ao passar o mouse em cima
    tooltip = {
        'html': '<b>Condado:</b> {name}',
        'style': {'backgroundColor': 'steelblue', 'color': 'white', 'fontsize': '10px'}
    }

    mapa = pdk.Deck(
        initial_view_state=view_state,
        map_style="light",
        layers=[polygon_layer, highlight_layer], # passando a camada criada 'polygon_layer' para o mapa
        tooltip=tooltip,
    )

    st.pydeck_chart(mapa)
