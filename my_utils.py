def get_models_by_task(task):
    if task == 'Text Classification':
        return ["roberta-base", "xlm-roberta-base"]
    elif task == 'Text Generation':
        return ['TinyLlama/TinyLlama-1.1B-Chat-v1.0','openai-community/gpt2','openlm-research/open_llama_3b_v2','distilbert/distilgpt2','lightblue/suzume-llama-3-8B-multilingual', 'ai-forever/rugpt3large_based_on_gpt2', "ai-forever/rugpt3medium_based_on_gpt2", "ai-forever/rugpt3small_based_on_gpt2"]
    else:
        return []