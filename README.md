
```
sudo apt install python3-virtualenv
virtualenv knowowl
source knowowl/bin/activate
```
```
pip install --upgrade pip pydantic
pip install -r requirements.txt  # Assuming this includes streamlit, ydata-profiling, and any other required packages not explicitly related to langchain
pip install --upgrade langchain langchain-community langchain-core langchain_community.prompts langchain-openai
pip cache purge  # Optional, based on whether you've encountered issues requiring cache purging
```
``` 
streamlit run app3.py 
```
#TODO
Ollama Pieces. Roles
