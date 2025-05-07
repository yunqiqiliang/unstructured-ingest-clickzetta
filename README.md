## Unstructured Ingest  

For details, see the [Unstructured Ingest overview](https://docs.unstructured.io/ingestion/overview) in the Unstructured documentation.

## Yunqi Lakehouse

For details, see [Yunqi Lakehouse documentation](https://www.yunqi.tech/documents).


conda activate unstructured
conda deactivate
conda remove -n unstructured --all

conda create -n unstructured311 python=3.11
conda activate unstructured311

pip install -e .

conda install build
python -m build

pip install ipywidgets --upgrade