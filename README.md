# Consumer complaints classifier

Streamlit deployment for the DistilBERT model trained in `notebookf8b922d97c.ipynb`.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The app expects the trained model at `final_distilbert/`. You can override that path:

```powershell
$env:MODEL_DIR="D:\AI\AI_Trainnig\Consumer-Complaints\final_distilbert"
streamlit run streamlit_app.py
```

## Deploy

Recommended option: Hugging Face Spaces with the Streamlit SDK.

1. Create a new Space and choose Streamlit.
2. Upload this project, including `final_distilbert/`.
3. Keep `.gitattributes` so `model.safetensors` is handled by Git LFS.
4. Set the app file to `app.py` if the platform asks for an entry point.

`model.safetensors` is about 268 MB, so normal GitHub uploads may fail without Git LFS.
