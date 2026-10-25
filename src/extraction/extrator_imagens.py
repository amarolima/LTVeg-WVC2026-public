#========== INICIO DAS IMPORTAÇÕES ==========

#Importanto bilbiotecas de sistema
from qgis.utils import iface
from PyQt5.QtCore import QSize
import processing
import os
from datetime import datetime

#Importanto bibliotecas do QGIS
from qgis.core import *

from qgis.core import (
    QgsProject, 
    QgsVectorLayer,
    QgsRasterLayer,
    QgsRectangle,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsPointXY,
    QgsRasterLayer
)

#Importando bilbioteca Pandas para leitura de planilhas do Excel
import pandas as pd

#Importanto bibliotecas EXIF para gravação de metadados
from PIL import Image
#import piexif

print('Fim da importação das bibliotecas!')
#========== FIM DAS IMPORTAÇÕES ==========

# ========== CONFIGURAÇÕES E CAMINHOS ==========
# Caminho da planilha contendo as coordenadas dos vértices das linhas de transmissão
planilha_diretorio = r"<repo>\data\metadata\coordenadas_extraidas.xlsx"

# Diretório para salvar as imagens recortadas
output_dir = r"<pasta_de_saida>\imagens"

#Diretório para salvar as linhas sorteadas 
diretorio = r"<pasta_de_saida>\coordenadas_sorteadas"

# Tamanho da imagem em pixels
image_width = 650
image_height = 650

# Resolução em metros por pixel
# Exemplo: 1.0 = 1 metro por pixel
resolution = 0.25

#Número de imagens em cada grupo
n = 100

#Altura da câmera em metros
altura_camera = 50

#Definindo variável para inicializar o método construtor
__name__ = '__main__'

print('Fim das configurações e caminhos!')
#========== FIM DAS CONFIGURAÇÕES E CAMINHOS ==========

# ========== LEITURA DA PLANILHA DE COORDENADAS E AGRUPAMENTO DE LINHAS ==========
#Lendo a planilha principal
principal = pd.read_excel(planilha_diretorio)
#Conferindo as primeiras linhas
print("Imprimindo primeiras linhas da planilha completa para conferência...")
print(principal.head())

#Criando os grupos

#Linhas do G1 - 800kV
G1 = principal[principal["Tensao"] == 800]
print("Tensão de 800kV...")
print(G1.head())

#Linhas do G2 - 765 e 600kV
G2 = principal[(principal["Tensao"] == 765) | (principal["Tensao"] == 600)]
print("Tensão de 765 e 600kV...")
print(G2.head())

#Linhas do G3 - 500, 440 e 345kV
G3 = principal[(principal["Tensao"] == 500) | (principal["Tensao"] == 440) | (principal["Tensao"] == 345)]
print("Tensão de 500, 440 e 345kV...")
print(G3.head())

#Linhas do G4 - 230 e 138kV
G4 = principal[(principal["Tensao"] == 230) | (principal["Tensao"] == 138)]
print("Tensão de 230 e 138kV...")
print(G4.head())

print('Fim da separação por grupos!')
#========== FIM DA LEITURA DA PLANILHA DE COORDENADAS E AGRUPAMENTO DE LINHAS ==========


# ========== FUNÇÕES ==========

def sorteia_vertices(nome_grupo,grupo,n,endereco):
    #Sorteia n linhas no grupo informado com reprodutibilidade e salva uma planilha em Excel com os dados sorteados
    #Se quiser sortear sempre as mesmas linhas, usar random_state = 123 como argumento do comando amostra = grupo.sample(n)

    amostra = grupo.sample(n)
    
    #Obtendo data e hora do sistema
    data_hora = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    #Salva as linhas sorteadas numa planilha Excel no endereço infomado
    amostra.to_excel(f"{endereco}\{nome_grupo}_{n}_{data_hora}.xlsx", index=False)
    return amostra

def carregar_camada_google_satellite():
    #Carrega a camada Google Satellite no QGIS
    # URL do serviço XYZ do Google Satellite
    url = "type=xyz&url=https://mt1.google.com/vt/lyrs%3Ds%26x%3D%7Bx%7D%26y%3D%7By%7D%26z%3D%7Bz%7D&zmax=19&zmin=0"
    
    raster_layer = QgsRasterLayer(url, "Google Satellite", "wms")
    
    if not raster_layer.isValid():
        print("Erro ao carregar camada Google Satellite")
        return None
    
    QgsProject.instance().addMapLayer(raster_layer)
    print("Camada Google Satellite carregada com sucesso")
    return raster_layer

def criar_extent_centrado(ponto_central, largura_metros, altura_metros, crs):
    # Cria um extent (retângulo) centrado no ponto X_coord_pre e Y_coord_pre
    x = ponto_central.x()
    y = ponto_central.y()
    
    half_width = largura_metros / 2
    half_height = altura_metros / 2
    
    extent = QgsRectangle(
        x - half_width,
        y - half_height,
        x + half_width,
        y + half_height
    )
    
    return extent


def exportar_imagem_recortada(extent, crs, caminho_saida, largura_px, altura_px):
    #Exporta a visualização atual do mapa como imagem
    # Configura o canvas (tela)
    canvas = iface.mapCanvas()
    canvas.setDestinationCrs(crs)
    canvas.setExtent(extent)
    canvas.zoomScale(altura_camera)
    canvas.refresh()
    
    # Aguarda o carregamento
    canvas.waitWhileRendering()
    
    # Configura as dimensões da imagem
    settings = canvas.mapSettings()
    settings.setOutputSize(QSize(largura_px, altura_px))
    settings.setExtent(extent)
    
    # Renderiza e salva a imagem
    from qgis.core import QgsMapRendererParallelJob
    job = QgsMapRendererParallelJob(settings)
    job.start()
    job.waitForFinished()
    
    image = job.renderedImage()
    #Salvando a imagem recortada
    image.save(caminho_saida)
    


def processar_linhas_transmissao(pontos_G1, pontos_G2, pontos_G3, pontos_G4, output_dir, n):
    #Processa as linhas e gera imagens para vértices selecionados
    #Testa se existe o diretório, se não existir, cria o diretório.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # crs_destino é o EPSG de onde a imagem será recortada (normalmente EPSG:3857 para Google Maps)
    crs_destino = QgsCoordinateReferenceSystem("EPSG:3857")

    #crs_origem é o EPSG do sistema de coordenadas do shapefile das linhas da EPE
    crs_origem = QgsCoordinateReferenceSystem("EPSG:4674")
    
    # Calcula altura e largura em metros da imagem
    largura_metros = image_width * resolution
    altura_metros = image_height * resolution    
    
    contador = 0
    # Itera sobre o total n de imagens para cada grupo
    for indice, row in pontos_G1.iterrows():
        #Iterando sobre o grupo G1
        ponto_extraido_G1 = QgsPointXY(row["X_coord_pre"], row["Y_coord_pre"])
        # Transformador de coordenadas
        transform = QgsCoordinateTransform(crs_origem, crs_destino, QgsProject.instance())
        ponto_transformado_G1 = transform.transform(ponto_extraido_G1)
        # Calcula extent
        extent_G1 = criar_extent_centrado(
            ponto_transformado_G1, 
            largura_metros, 
            altura_metros, 
            crs_destino
            )                
        # Nome do arquivo de saída
        # Numero_da_Linha_Original_Planilha_Grupo_Tensão
        nome_arquivo = f"{indice}_{"G1"}_{row["Tensao"]}.png"
        caminho_saida = os.path.join(output_dir, nome_arquivo)

        # Exporta imagem
        exportar_imagem_recortada(
            extent_G1, 
            crs_destino, 
            caminho_saida, 
            image_width, 
            image_height
        )
        contador += 1
    #-----------------------------------------------------------------------------------
    #Iterando sobre o grupo G2
    for indice, row in pontos_G2.iterrows():
        ponto_extraido_G2 = QgsPointXY(row["X_coord_pre"], row["Y_coord_pre"])
        # Transformador de coordenadas
        transform = QgsCoordinateTransform(crs_origem, crs_destino, QgsProject.instance())
        ponto_transformado_G2 = transform.transform(ponto_extraido_G2)
        # Calcula extent
        extent_G2 = criar_extent_centrado(
            ponto_transformado_G2, 
            largura_metros, 
            altura_metros, 
            crs_destino
        )
            
        # Nome do arquivo de saída
        # Numero_da_Linha_Original_Planilha_Grupo_Tensão
        nome_arquivo = f"{indice}_{"G2"}_{row["Tensao"]}.png"
        caminho_saida = os.path.join(output_dir, nome_arquivo)
            
        # Exporta imagem
        exportar_imagem_recortada(
            extent_G2, 
            crs_destino, 
            caminho_saida, 
            image_width, 
            image_height
        )
        contador += 1
    #-----------------------------------------------------------------------------------
    #Iterando sobre o grupo G3
    for indice, row in pontos_G3.iterrows():
        ponto_extraido_G3 = QgsPointXY(row["X_coord_pre"], row["Y_coord_pre"])
        # Transformador de coordenadas
        transform = QgsCoordinateTransform(crs_origem, crs_destino, QgsProject.instance())
        ponto_transformado_G3 = transform.transform(ponto_extraido_G3)
        # Calcula extent
        extent_G3 = criar_extent_centrado(
            ponto_transformado_G3, 
            largura_metros, 
            altura_metros, 
            crs_destino
        )
                
        # Nome do arquivo de saída
        # Numero_da_Linha_Original_Planilha_Grupo_Tensão
        nome_arquivo = f"{indice}_{"G3"}_{row["Tensao"]}.png"
        caminho_saida = os.path.join(output_dir, nome_arquivo)
                
        # Exporta imagem
        exportar_imagem_recortada(
            extent_G3, 
            crs_destino, 
            caminho_saida, 
            image_width, 
            image_height
        )
        contador += 1
    #-----------------------------------------------------------------------------------
    #Iterando sobre o grupo G4  
    for indice, row in pontos_G4.iterrows():
        ponto_extraido_G4 = QgsPointXY(row["X_coord_pre"], row["Y_coord_pre"])
        # Transformador de coordenadas
        transform = QgsCoordinateTransform(crs_origem, crs_destino, QgsProject.instance())
        ponto_transformado_G4 = transform.transform(ponto_extraido_G4)
        # Calcula extent
        extent_G4 = criar_extent_centrado(
            ponto_transformado_G4, 
            largura_metros, 
            altura_metros, 
            crs_destino
        )
                
        # Nome do arquivo de saída
        # Numero_da_Linha_Original_Planilha_Grupo_Tensão
        nome_arquivo = f"{indice}_{"G4"}_{row["Tensao"]}.png"
        caminho_saida = os.path.join(output_dir, nome_arquivo)
                
        # Exporta imagem
        exportar_imagem_recortada(
            extent_G4, 
            crs_destino, 
            caminho_saida, 
            image_width, 
            image_height
            )
        contador += 1
    
    
    print(f"\nProcessamento concluído! Total de imagens: {contador}")

print('Fim da leitura das funções!')
# ========== EXECUÇÃO PRINCIPAL ==========

def main():
    print("Iniciando processamento...")
    
    # Carrega a camada do Google no QGIS
    google_layer = carregar_camada_google_satellite()


    #Sorteando n vertices aleatórios dentros do grupo G1 e salvando uma planilha
    G1_sorteado = sorteia_vertices("G1",G1,n,diretorio)
    
    #Sorteando n vertices aleatórios dentros do grupo G2 e salvando uma planilha    
    G2_sorteado = sorteia_vertices("G2",G2,n,diretorio)
    
    #Sorteando n vertices aleatórios dentros do grupo G3 e salvando uma planilha
    G3_sorteado = sorteia_vertices("G3",G3,n,diretorio)
    
    #Sorteando n vertices aleatórios dentros do grupo G4 e salvando uma planilha
    G4_sorteado = sorteia_vertices("G4",G4,n,diretorio)
    
    # Processa as linhas
    processar_linhas_transmissao(G1_sorteado, G2_sorteado, G3_sorteado, G4_sorteado, output_dir, n)
    
    print("Script finalizado!")


# Executa o script
if __name__ == '__main__':
    main()