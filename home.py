import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st

from joblib import load

from notebooks.src.config import DADOS_GEO_MEDIAN, DADOS_LIMPOS, MODELO_FINAL


@st.cache_data
def carregar_dados_limpos():
    return pd.read_parquet(DADOS_LIMPOS)


@st.cache_data
def carregar_dados_geo():
    return gpd.read_parquet(DADOS_GEO_MEDIAN)


@st.cache_resource    
def carregar_modelo():
    return load(MODELO_FINAL)



df = carregar_dados_limpos()
gdf_geo = carregar_dados_geo()
modelo = carregar_modelo()





st.title('Previsão de preços de imóveis')


# as 13 features (colunas que não são a target), são os valores que preciso entrar para que ele me dê a previsão
# criando as variáveis para inserir os valores necessários para realizar previsões com o nosso modelo:

longitude = st.number_input('Longitude', value=-122.33)
latitude = st.number_input('Latitude', value=37.88)

housing_median_age = st.number_input('Idade do imóvel', value=10)

total_rooms = st.number_input('Total de cômodos', value=800)
total_bedrooms = st.number_input('Total de quartos', value=100)
population = st.number_input('População', value=300)
households = st.number_input('Domicílios', value=100)

median_income = st.slider('Renda média (múltiplos de US$ 10k', 0.5, 15.0, 4.5, 0.5)

ocean_proximity = st.selectbox('Proximidade do oceano', df['ocean_proximity'].unique())

median_income_cat = st.number_input('Categoria de Renda', value=4) # vai de 1 a 5

rooms_per_household = st.number_input('Quartos por domicílio', value=7)
bedrooms_per_rooms = st.number_input('Quartos por cômodo', value=0.2)
population_per_household = st.number_input('Pessoas por domicílio', value=2)



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
    'median_income': median_income,
    'ocean_proximity': ocean_proximity,
    'median_income_cat': median_income_cat,
    'rooms_per_household': rooms_per_household,
    'bedrooms_per_rooms': bedrooms_per_rooms,
    'population_per_household': population_per_household,
}

# criando o DataFrame

df_entrada_modelo = pd.DataFrame(entrada_modelo, index=[0]) # 'index=[0]' para o streamlit não acusar um erro dizendo que falta um índice para o DataFrame

botao_previsao = st.button('Prever preço')

if botao_previsao:
    preco = modelo.predict(df_entrada_modelo)
    st.write(f'Preço previsto: US$ {preco[0][0]:.2f}')


































