#### llm model — chainlit + langchain

upload a pdf, then prompt questions about it, model will return chunks and the answer.

install deps:
```
pip install -r requirements.txt
```

create env and input openai key
```
OPENAI_API_KEY=
```

run model:
```
chainlit run app.py -w
```