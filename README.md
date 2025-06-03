## DeepSense-SQL Instruction

## ⚡Environment

1. Config your local environment.

```txt
conda create -n deepsense-sql python=3.10 -y
conda activate deepsense-sql
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"
```

2. Edit openai config at **core/api_config.py**.

```txt
client = OpenAI(api_key='YOUR_OPENAI_API_KEY',base_url='YOUR_OPENAI_API_BASE')
MODEL_NAME = 'YOUR_MODEL_NAME'
```

## 📝Evaluation

1. Run SQL generation at   **run.py**.


2. Run evaluation EX at   **evaluation/evaluation_bird_ex.py**.


## 🌟 Project Structure

```txt
├─data # store datasets and databases
|  ├─bird
├─core
|  ├─agents.py       # define three agents class
|  ├─api_config.py   # OpenAI API ENV config
|  ├─chat_manager.py # manage the communication between agents
|  ├─const.py        # prompt templates and CONST values
|  ├─llm.py          # api call function and log print
|  ├─utils.py        # utils function
├─first-stage-ranker # pretrained model from MetaSQL for first stage candidate SQLs ranking
|  ├─...
├─evaluation # evaluation scripts
|  ├─evaluation_bird_ex.py
├─README.md
├─requirements.txt
├─run.py # main run script for SQL generation
```