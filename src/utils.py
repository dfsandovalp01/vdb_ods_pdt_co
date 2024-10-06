from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
import os

import zipfile
import os
import sys
from PIL import Image
import pdfplumber
import pandas as pd
import gradio as gr

import pandas as pd
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def vdbPDF():
    embeddings_st = SentenceTransformerEmbeddings(
        # model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_name="hackathon-pln-es/paraphrase-spanish-distilroberta",
        # model_name="sentence-transformers/all-MiniLM-L6-v2"

        # device="cuda",
        model_kwargs={"device":"cpu"}
    )


    # from google.colab import drive
    # drive.mount('/content/drive')
    # current_dir = os.getcwd()
    current_dir = self.dir_root

    vdb_dir = os.path.join(current_dir, os.pardir, "data", "vdb", "chromaPdtGob_pSpDroberta")
    # print(os.path.exists(data_dir))
    vectorstore_chroma = Chroma(
        persist_directory=vdb_dir,#NOMBRE_INDICE_CHROMA,
        embedding_function=embeddings_st
    )

    return vectorstore_chroma

class lectorPDFnlp:
    def __init__(self, model_name="hackathon-pln-es/paraphrase-spanish-distilroberta",dir_root):
        model_name = self.model_name
        dir_root = self.dir_root
        dir_temp_img = os.path.join(self.dir_root, "data", "temp", "img")
        dir_temp_pdf = os.path.join(self.dir_root, "data", "temp", "pdf")

    def readPageVW(self, docname, page):
        page_view = None
        with pdfplumber.open(docname) as pdf:
            page_view = pdf.pages[page-1].to_image()
            # dir = os.path.join(self.dir_root, "data", "temp", "img")
            # dir = f"/content/images"
            # dir = dir.replace('Departamentos/','')#.replace('Departamentos/','')
            # Guarda la imagen como .png
            os.makedirs(self.dir_temp_img,exist_ok=True)
            page_view.save(f"{self.dir_temp_img}/page_{page}_doc_{docname.replace('Departamentos/','').replace('.pdf','.png')}")
        return page_view

    def readZip(self,idDoc,page):
        dir=os.path.join(self.dir_root,'Notebooks','Departamentos')
        # def readZip(idDoc,page,dir=os.path.join(os.getcwd(), "Departamentos")):
        # def readZip(idDoc,page,dir_temp_pdf=dir_temp_pdf):
        # for _ in [f for f in os.listdir(dir) if f.endswith('.pdf')]:
        #     os.remove(os.path.join(dir, _))
        docname = f'{dir}/{idDoc}.pdf'
        # docname = os.path.join(dir, f'{idDoc}.pdf')
        # print(f'docname: {docname}')
        if os.path.exists(docname):
            # os.remove(docname)
            page_view = readPageVW(docname, page)
            # print(f"{} ha sido eliminado.")
        else:
            # print(f"El archivo {i} no existe.")
            pages = []
            bdl_corpus = pd.DataFrame()
            page_view = None
            # ruta = '/content/drive/MyDrive/Automatizacion ODS 11a1/Pruebas/Departamentos/datos/Departamentos.zip'
            ruta = os.path.join(self.dir_root, "data", "comprimido", "Departamentos.zip")
            # Abrir el archivo .zip
            with zipfile.ZipFile(ruta, 'r') as zip_ref:
            # Listar el contenido del .zip
            lista_archivos = zip_ref.namelist()
            # print(lista_archivos)

            try:
                # i = i.replace('Departamentos/','')
                zip_ref.extract(docname)
                page_view = readPageVW(docname, page)

            except Exception as e:
                print(f'Error:  {e}')
                pass
        
            
        return page_view


    from sentence_transformers import SentenceTransformer, util

def model():
    return SentenceTransformer('hackathon-pln-es/paraphrase-spanish-distilroberta')

    

def search(query='',filter='',dpto='',rank='15', clusters='4'):
    
#departamentos
    lista_archivos_pdf = ['05.pdf', '08.pdf', '15.pdf', '17.pdf', '18.pdf', '19.pdf', '20.pdf', '23.pdf', '25.pdf', '27.pdf', '41.pdf', '52.pdf', '54.pdf', '63.pdf', '66.pdf',
    '68.pdf', '73.pdf', '76.pdf', '81.pdf', '85.pdf', '86.pdf', '88.pdf', '91.pdf', '94.pdf', '95.pdf', '97.pdf', '99.pdf']
    # data_dir = os.path.join(self.dir_root, "data")
    # print(data_dir)
    codigos_dane = pd.read_csv(os.path.join(sys.path[-1], "data", "Tabla_codigos_Dane.txt"), sep='|', dtype=str)
    codigoDepartamentoPdf = [_.replace('.pdf','') for _ in lista_archivos_pdf]
    codigos_dane_dpto = codigos_dane.groupby(['CodigoDepartamento', 'NombreDepartamento']).agg({'CodigoMunicipio':'count'}).reset_index()
    codigos_dane_dpto = codigos_dane_dpto[codigos_dane_dpto.CodigoDepartamento.isin(codigoDepartamentoPdf)]

    # query = "proyecciones de poblacion"
    # doc1 = None
    # if os.path.exists('/content/images'):
    if os.path.exists((os.path.join(sys.path[-1], "data", "temp", "img"))):
        # for _ in [f for f in os.listdir('/content/images') if f.endswith('.png')]:
        for _ in [f for f in os.listdir((os.path.join(sys.path[-1], "data", "temp", "img"))) if f.endswith('.png')]:
        os.remove((os.path.join(sys.path[-1], "data", "temp", "img", _)))
        # print(f"{i} ha sido eliminado.")
    if filter == "Filtro":
        doc1 = codigos_dane_dpto[codigos_dane_dpto.NombreDepartamento == dpto].reset_index(drop=True).loc[0,'CodigoDepartamento']
        doc1 = f'{doc1}.pdf'
        docs = vectorstore_chroma.similarity_search_with_score(query, k=rank, filter={"source": doc1})  # .unique()
    else:
        docs = vectorstore_chroma.similarity_search_with_score(query, k=rank)



    # scores = [round(1 - doc[1]/100,2) for doc in docs ]
    scores = [round(doc[1],2) for doc in docs ]
    text = [doc[0].page_content for doc in docs ]
    page = [doc[0].metadata['page'] for doc in docs ]
    source = [doc[0].metadata['source'] for doc in docs ]
    idDN = [doc[0].metadata['source'].replace('.pdf','') for doc in docs ]
    ids = list(range(0, len(text))) 
    # 

    df = pd.DataFrame({
        'id': ids,
        # 'cluster' : cluster_assignment,
        'source': source,
        'page': page,
        'scores': scores,
        'text': text,
        'idDANE': idDN
        })
    
    df = df.merge(codigos_dane_dpto[['CodigoDepartamento','NombreDepartamento']], 'left', left_on='idDANE', right_on='CodigoDepartamento')
    
    imgs = []
    for index, row in df.iterrows():
        # imgs.append(readZip(row['idDANE'], row['page']))
        readZip(row['idDANE'], row['page'])
    png_files = [f for f in os.listdir((os.path.join(sys.path[-1], "data", "temp", "img"))) if f.startswith('page_')]
    # png_files = [f for f in os.listdir('/content/images') if f.startswith('page_')]
    for file in png_files:
        # image_path = os.path.join('/content/images', file)
        image_path = os.path.join((os.path.join(sys.path[-1], "data", "temp", "img", file)))
        image = Image.open(image_path)
        image = [image, file]
        imgs.append(image)
        # print(row['idDANE'], row['NombreDepartamento'])

    ## Clusters
    labels = [f"{a}-{b}" for a, b in zip(page, df['NombreDepartamento'])]
    # labels = [f"{a}-{b}: {c}" for a, b, c in zip(df['NombreDepartamento'],page,scores)]
    text.append(query)
    labels.append('query')

    results_embeddings = model.encode(text,batch_size=64,show_progress_bar=True, device='cpu')  

    emb_results = np.array(results_embeddings)
    print(f'3: {len(emb_results)}')

    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    embeddings_2d = tsne.fit_transform(emb_results)
    print(f'4: {len(embeddings_2d)}')

    num_clusters = clusters
    clustering_model = KMeans(n_clusters=num_clusters)
    clustering_model.fit(emb_results)
    cluster_assignment = clustering_model.labels_

    
    
    
    

    plt.figure(figsize=(6, 4))
    colors = ['r', 'g', 'b', 'y', 'c', 'm']

    for index, embedding in enumerate(embeddings_2d):
        # print(f'{index} - {embedding}')
        plt.scatter(embedding[0],embedding[1],color=colors[cluster_assignment[index]])

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Grupos")

    

    labels_plt = labels.copy()
    labels_plt.append('query')
    print(len(labels_plt))

    for i, sentence in enumerate(labels):
        plt.annotate(sentence, (embeddings_2d[i,0],embeddings_2d[i,1]))

    
    plt.grid(False)
    ruta_img = os.path.join(current_dir, os.pardir, "data","temp","img")
    # ruta_img = '/content/images'
    plt.savefig(os.path.join(ruta_img, "grupos.png"))

    

    img_cl = []  
    png_files = [f for f in os.listdir(ruta_img) if f.endswith('grupos.png')]
    for file in png_files:
        # image_path = os.path.join(ruta_img, file)
        # image_path = os.path.join('/content/images', file)
        image_path = os.path.join(ruta_img, file)
        image = Image.open(image_path)
        image = [image, file]
        img_cl.append(image)

    df[['source','page']] = df[['source','page']].astype(str)
    df['LABELS'] = df['source'] + '-' + df['page']
    df['cluster'] = cluster_assignment[:-1]
    
    

    return df[['cluster','idDANE','page','scores','NombreDepartamento','text']], img_cl, imgs

    # pr = search('desarrollo equilibrado asentamientos, pueblos, ciudades','no filtro','x',10)
    # Define possible genres
    # clientes = clients.tolist()

def demoLaunch(fun=search()):
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(lines=5, placeholder="Escribe aquí tu consulta...", label="Consulta")
                gr.Examples(["dinámica población","desarrollo territorial equilibrado", "margen fiscal local"], inputs=[prompt])
            with gr.Column():
                filtros = gr.Radio(["Filtro", "Sin filtro"], label="Filtros")
                departamento = gr.Dropdown(choices=list(codigos_dane_dpto.NombreDepartamento), label="Departamento")
                salidas = gr.Number(minimum=1, maximum=15, value=10, label="Número de resultados")
                clusters = gr.Number(minimum=1, maximum=5, value=3, label="Número de Clusters")
                

        with gr.Row():
            consulta_btn = translate_btn = gr.Button(value="Consultar")

        with gr.Row():
            tabla = gr.Dataframe(type="pandas", label="Resultados")

        # with gr.Row():
        #     resol = gr.Slider(minimum=500, maximum=1500, value=100, label="Resolución")

        with gr.Row():
            galeria = gr.Gallery(label="Clusters", show_label=True, elem_id="gallery", scale=1,object_fit="contain", height=500)#"auto")

        with gr.Row():
            galeria1 = gr.Gallery(label="Hoja PDT", show_label=True, elem_id="gallery1", scale=2,object_fit="contain", height=1000)#"auto")


        translate_btn.click(fun, inputs=[prompt, filtros, departamento, salidas, clusters], outputs=[tabla, galeria, galeria1])
    
    return demo




