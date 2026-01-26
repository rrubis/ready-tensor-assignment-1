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
│   └── vectordb.py      # Vector database wrapper
├── data/              # Currently supports .txt and .pdf files 
│   ├── *.txt          
│   ├── *.pdf
├── requirements.txt    # All dependencies included
├── .env.example       # Environment template
└── README.md          # This guide
└── LICENSE.txt        
```

### To run locally
- Create your virtual environment and install dependencies listed in `requirements.txt`.
- Move the two example files (`papers_schema.json` and `papers.csv`) in the `examples` directory into the `./inputs/schema` and `./inputs/data` folders, respectively (or alternatively, place your custom dataset files in the same locations).
- Run the script `src/create_db.py` to create the database. This will save the db in the path `./db/`.
- Run the script `src/serve.py` to start the inference service, which can be queried using the `/ping` and `/infer` endpoints. The service runs on port 8080.
```

## Requirements

Dependencies for the main model implementation in `src` are listed in the file `requirements.txt`.
You can install these packages by running the following command from the root of your project directory:

```python
pip install -r requirements.txt
```

## Contact Information

Repository created by Ready Tensor, Inc. (https://www.readytensor.ai/)
