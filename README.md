# llm_zoomcamp_code
Notes &amp; Homeworks from the LLM Zoomcamp 2026


### Setting up environment
```{bash} 
python3 -m pip install --upgrade pip 
pip install uv
git clone https://github.com/davidevdt/llm_zoomcamp_code.git 
cd llm_zoomcamp_code
uv sync 
```

### OpenAI API Key 
`platform.openai.com/login` 

Include the key using
```{bash}
cp .env.example .env
```
and pasting your key into the corresponding env variable.
