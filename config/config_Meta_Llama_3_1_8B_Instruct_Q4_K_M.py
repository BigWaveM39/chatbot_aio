config = {
  "name": "Config for Chat ID 1722416667888",
  "load_params": {
    "n_ctx": 2048, # max token
    "n_batch": 512, 
    "rope_freq_base": 0,
    "rope_freq_scale": 0,
    "n_gpu_layers": -1,
    "use_mlock": True,
    "main_gpu": 0,
    "tensor_split": [0],
    "seed": -1,
    "f16_kv": True,
    "use_mmap": True,
    "no_kv_offload": False,
    "num_experts_used": 0
  },
  "inference_params": {
    "n_threads": 4,
    "n_predict": -1,
    "top_k": 40,
    "min_p": 0.05,
    "top_p": 0.95,
    "temp": 0.7, # accuracy [0, 1]
    "repeat_penalty": 1.1,
    "input_prefix": "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n",
    "input_suffix": "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    "antiprompt": ["<|start_header_id|>", "<|eot_id|>"],
    "pre_prompt": """
Sei un MetaHuman su Twitch chiamata BigJammy. Sei gentile e disponibile. I messaggi sono nel formato 'NomeUtente: Testo'.

Rispondi in modo breve, conciso e diretto per garantire rapidità con un tono empatia. Ogni nuovo utente riceve prima un breve messaggio di benvenuto e poi rispondi. 
Personalizza le risposte. Accogli calorosamente i nuovi utenti.

Esempi:
1. Se 'Mario' scrive 'Ciao!', rispondi: 'Ciao Mario! ' e poi rispondi a tutte le domande.

Riferisciti ai messaggi passati per conversazioni fluide e rapide.
Se non sai cosa dire, rispondi con un'opzione di aiuto.
Non elencare le risorse disponibili, ma rispondi in modo naturale.
""",
    "pre_prompt_suffix": "",
    "pre_prompt_prefix": "<|start_header_id|>system<|end_header_id|>\n\n",
    "seed": -1,
    "tfs_z": 1,
    "typical_p": 1,
    "repeat_last_n": 64,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "n_keep": 0,
    "logit_bias": {},
    "mirostat": 0,
    "mirostat_tau": 5,
    "mirostat_eta": 0.1,
    "memory_f16": True,
    "multiline_input": False,
    "penalize_nl": True
  }
}
