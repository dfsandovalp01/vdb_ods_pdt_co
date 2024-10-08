**MANUAL DE INSTALACIÓN HERRAMIENTA ODS 11.a.1.** 

Pre-requisitos: Se debe contar con las siguientes herramientas
1.	Anaconda o miniconda3
2.	Visual Studio Code
3.	GIT

Pasos:
1.	Clonar repositorio: Abrir una terminal de Anaconda PowerShell, ubicar la ruta de trabajo con el comando:
_cd ‘ruta/de/trabajo’_
Insertar el siguiente comando para clonar el repositorio:
git clone https://github.com/dfsandovalp01/vdb_ods_pdt_co.git

2.	Configurar entorno virtual: 
Desde el terminal de Anaconda, ubicarse en la ruta del repositorio y abrir Visual Studio Code:
_cd_ ‘ruta/de/trabajo/vdb_ods_pdt_co’
Configurar entorno virtual:
conda env create -f environment_export.yml
Activar el entorno virtual.
conda activate nlp_base
3.	Abrir Visual Studio Code y ejecutar Notebook:
Desde el terminal de Anaconda, mantenerse en la ruta de trabajo y ejecutar:
_code ._
Abrir el Notebook (Notebooks/notebook_ejecutable.ipynb) , seleccionar entorno virtual previamente configurado y ejecutar cada celda.

