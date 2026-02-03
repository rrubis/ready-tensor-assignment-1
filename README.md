# RAG-Based AI Assistant For Vector Database Offerings

## Project Description

This repository is an implementation of the re-usable RAG-Based Assistant. It is implemented in flexible way so that it can be used with any Text (.txt) and PDF (.pdf) document files. 

---

Here are the highlights of this implementation: <br/>

- A **Similarity Search** algorithm built using **SentenceTransformer** package.
  Additionally, the implementation contains the following features:

## 📁 Project Structure

```
rt-aaidc-project1-template/
├── src/
│   ├── app.py           # Main RAG application
│   |── vectordb.py      # Vector database wrapper
│   |── prompt_template.py      # Prompt template
│   └── paths.py      # Paths required by application
├── data/              # Currently supports .txt and .pdf files 
│   ├── *.txt          
│   ├── *.pdf
├── requirements.txt    # All dependencies included
├── .env.example        # Environment template
├── project_1_publications  # Data files metadata (license, title, publication description
├── README.md          # This guide
└── LICENSE.txt        
```

### To run locally
- Create your virtual environment and install dependencies listed in `requirements.txt`.
- Run the script `src/app.py`
```

## Requirements

Dependencies for the main model implementation in `src` are listed in the file `requirements.txt`.
You can install these packages by running the following command from the root of your project directory:

```python
pip install -r requirements.txt
```
## LICENSE

This project is provided under the Attribution-NonCommercial-ShareAlike 4.0 International. Please see the LICENSE file for more information.

## Contact Information

Repository created by Russ Rubis (rrubis@yahoo.com)
