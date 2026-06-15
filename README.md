# llm_zoomcamp_code
Notes &amp; Homeworks from the LLM Zoomcamp 2026


### Setting up environment
```{bash} 
python3 -m pip install --upgrade pip 
pip install uv 
uv init 
uv add requests minsearch openai jupyter python-dotenv 
```

### OpenAI API Key 
`platform.openai.com/login` 

Include the key using
```{bash}
cp .env.example .env
```
and pasting your key into the corresponding env variable.