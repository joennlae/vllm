
Seed always: 4419

While the Quantization is broken: the FP16 model seems to work find in the https://github.com/vllm-project/vllm/pull/2293 branch.

### Comparision Table

Tested models: `casperhansen/mixtral-instruct-awq`, `mistralai/Mixtral-8x7B-Instruct-v0.1`, `TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ`

| Model | TP | Max Model Len | GPU Memory | Tok/Sec | No early stopping |
| --- | --- | --- | --- | --- | --- |
| `casperhansen/mixtral-instruct-awq` | 8 | 32768 | 0.98 | 43.5 | ✅ |
| `casperhansen/mixtral-instruct-awq` | 8 | 4096  | 0.98 | 49.0 | ✅ |
| `casperhansen/mixtral-instruct-awq` | 4 | 32768 | 0.98 | 43.3 | ✅ |
| `casperhansen/mixtral-instruct-awq` | 4 | 4096  | 0.98 | 42.0 | ✅ |
| `casperhansen/mixtral-instruct-awq` | 2 | 16384 | 0.98 | 35.7 | ✅ |
| `casperhansen/mixtral-instruct-awq` | 2 | 4096  | 0.98 | 35.5 | ✅ |
| --- | --- | --- | --- | --- | --- |
| `mistralai/Mixtral-8x7B-Instruct-v0.1` | 8 | 32768 | 0.98 | --   | X |
| `mistralai/Mixtral-8x7B-Instruct-v0.1` | 8 | 4096  | 0.98 | --   | X |
| `mistralai/Mixtral-8x7B-Instruct-v0.1` | 4 | 4096  | 0.98 | 31.0 | ✅ | 
| `mistralai/Mixtral-8x7B-Instruct-v0.1` | 4 | 2048  | 0.98 | 30.8 | ✅ |
| --- | --- | --- | --- | --- | --- |

I get the early stopping on TP=8 for seeds: `4419`, `4420`, `8000`...
`--enfore-eager` does not make a difference if removed
### Request

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MODEL_NAME",
    "temperature": 0.0,
    "max_tokens": 2048,
    "messages": [
      {
        "role": "user",
        "content": "Create a JSON-formatted profile for a fictional character named Alex Mercer, who is a detective in a futuristic city. The profile should include his personal details, professional background, skills, and a brief description of his latest case."
      }
    ]
  }'
```

### Unquantized

Command:
```bash
python -m vllm.entrypoints.openai.api_server --model="mistralai/Mixtral-8x7B-Instruct-v0.1"  --download-dir="/scratch/janniss/models" --max-model-len 32768 --gpu-memory-utilization 0.98 --enforce-eager --dtype auto -tp 8 --seed 4419
```

### AWQ

Command:
```bash
python -m vllm.entrypoints.openai.api_server --model="casperhansen/mixtral-instruct-awq"  --quantization awq --download-dir="/scratch/janniss/models" --max-model-len 32768 --gpu-memory-utilization 0.98 --enforce-eager --dtype auto -tp 8 --seed 4419
```

### GPTQ

Command:
```bash
python -m vllm.entrypoints.openai.api_server --model="TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ"  --quantization gptq --download-dir="/scratch/janniss/models" --max-model-len 32768 --gpu-memory-utilization 0.98 --enforce-eager --dtype auto -tp 8 --seed 4419
```


#### AWQ Expected Results

TP 8:
```json
{
  "id": "cmpl-8354fd9bda2644c79138829b871941b3",
  "object": "chat.completion",
  "created": 704058,
  "model": "casperhansen/mixtral-instruct-awq",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": " {\n  \"name\": \"Alex Mercer\",\n  \"age\": 35,\n  \"gender\": \"Male\",\n  \"height\": \"6'2\",\n  \"weight\": \"190 lbs\",\n  \"hair_color\": \"Black\",\n  \"eye_color\": \"Ice Blue\",\n  \"ethnicity\": \"Caucasian\",\n\n  \"professional_background\": [\n    {\n      \"position\": \"Detective\",\n      \"department\": \"Neo City Police Department\",\n      \"service\": \"15 years\",\n      \"badge_number\": \"2147\",\n      \"rank\": \"Senior Detective\"\n    },\n    {\n      \"position\": \"Military Police\",\n      \"branch\": \"United States Army\",\n      \"service\": \"5 years\",\n      \"rank\": \"Sergeant\"\n    }\n  ],\n\n  \"skills\": [\n    \"Mastery in forensic investigation\",\n    \"Expertise in cybercrime\",\n    \"Proficient in martial arts and firearms\",\n    \"Advanced knowledge of quantum computing\",\n    \"Fluent in three languages: English, Spanish, Japanese\"\n  ],\n\n  \"latest_case\": {\n    \"case_name\": \"The Quantum Heist\",\n    \"description\": \"Alex Mercer was called to investigate a series of high-profile thefts involving priceless artifacts. The thief left no trace, and the only clue was a quantum signature at each crime scene. Mercer had to delve into the world of quantum computing, AI, and advanced encryption to track down the mysterious thief. After weeks of relentless pursuit, Mercer discovered an underground organization known as 'The Quants' was behind the heists. Mercer, with his tech-savvy skills, managed to infiltrate their network, expose their plans, and bring the criminals to justice.\"\n  }\n}"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 59,
    "total_tokens": 508,
    "completion_tokens": 449
  }
}
```

TP 4:
```json
{
  "id": "cmpl-d770350c7c5b4592ae565e5bf8ead337",
  "object": "chat.completion",
  "created": 704800,
  "model": "casperhansen/mixtral-instruct-awq",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": " {\n  \"name\": \"Alex Mercer\",\n  \"age\": 35,\n  \"gender\": \"Male\",\n  \"height\": \"6'2\",\n  \"weight\": \"190 lbs\",\n  \"hair_color\": \"Black\",\n  \"eye_color\": \"Ice Blue\",\n  \"ethnicity\": \"Caucasian\",\n\n  \"professional_background\": [\n    {\n      \"position\": \"Detective\",\n      \"department\": \"Neo City Police Department\",\n      \"service\": \"15 years\",\n      \"badge_number\": \"2147\",\n      \"rank\": \"Senior Detective\"\n    },\n    {\n      \"position\": \"Military Police\",\n      \"branch\": \"United States Army\",\n      \"service\": \"5 years\",\n      \"rank\": \"Sergeant\"\n    }\n  ],\n\n  \"skills\": [\n    \"Masterful investigative skills\",\n    \"Expertise in cyber forensics\",\n    \"Proficient in martial arts and firearms\",\n    \"Fluent in three languages (English, Spanish, Japanese)\",\n    \"Advanced knowledge of quantum computing\"\n  ],\n\n  \"latest_case\": {\n    \"case_name\": \"The Quantum Heist\",\n    \"description\": \"Alex Mercer was called to investigate a series of high-profile thefts involving advanced quantum technology. The thief left no traces behind, making it nearly impossible to track them down. However, using his exceptional cyber forensic skills, Alex managed to trace the source of the crimes to an underground organization known as 'The Dark Quantum'. After weeks of surveillance and covert operations, Alex led a successful raid on their hideout, apprehending the criminals and recovering the stolen technology.\",\n    \"status\": \"Closed\"\n  }\n}"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 59,
    "total_tokens": 490,
    "completion_tokens": 431
  }
}
```

TP 2:
```json
{
  "id": "cmpl-0831156f5b37412a9de566c2f49984f5",
  "object": "chat.completion",
  "created": 705043,
  "model": "casperhansen/mixtral-instruct-awq",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": " {\n  \"name\": \"Alex Mercer\",\n  \"age\": 35,\n  \"gender\": \"Male\",\n  \"height\": \"6'2\",\n  \"weight\": \"190 lbs\",\n  \"hair_color\": \"Black\",\n  \"eye_color\": \"Ice Blue\",\n  \"ethnicity\": \"Caucasian\",\n\n  \"professional_background\": [\n    {\n      \"position\": \"Detective\",\n      \"department\": \"Neo City Police Department\",\n      \"service\": \"15 years\",\n      \"badge_number\": \"2147\",\n      \"rank\": \"Senior Detective\"\n    },\n    {\n      \"position\": \"Military Police\",\n      \"branch\": \"United States Army\",\n      \"service\": \"5 years\",\n      \"rank\": \"Sergeant\"\n    }\n  ],\n\n  \"skills\": [\n    \"Mastery in forensic investigation\",\n    \"Expertise in cybercrime\",\n    \"Proficient in martial arts and firearms\",\n    \"Advanced knowledge of quantum computing\",\n    \"Fluent in three languages: English, Spanish, Japanese\"\n  ],\n\n  \"latest_case\": {\n    \"case_name\": \"The Quantum Heist\",\n    \"description\": \"A series of high-profile thefts targeting advanced technology companies, leaving no trace or evidence behind. Mercer's extensive background in cybercrime and quantum computing led to the arrest of an elusive hacker known only as 'The Architect'. The case involved unraveling complex encryptions and chasing virtual footprints across the darknet. Mercer successfully retrieved stolen technology worth billions and apprehended the culprit, preventing further damage to Neo City's infrastructure.\"\n  }\n}"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 59,
    "total_tokens": 483,
    "completion_tokens": 424
  }
}
```

### Unquantized Expected Results

ABORTS MID-ANSWER (always after numbers...)
TP 8:
```json
{
  "id": "cmpl-dfc188967e704fbe8dff9c0b6a785a8b",
  "object": "chat.completion",
  "created": 705358,
  "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": " {\n  \"name\": \"Alex Mercer\",\n  \"age\": 35,\n  \"gender\": \"Male\",\n  \"eye_color\": \"Ice Blue\",\n  \"hair_color\": \"Raven Black\",\n  \"height\": \"6'2\\\"\",\n  \"weight\": \"190 lbs\",\n\n  \"professional_background\": {\n    \"title\": \"Detective\",\n    \"department\": \"Centural City Police Department\",\n    \"service_duration\": \"10 years\",\n    \"badge_number\": \"2147\",\n    \"notable_achievements\": [\n      \"Solved the notorious 'Neon Gang' case in 2"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 59,
    "total_tokens": 228,
    "completion_tokens": 169
  }
}
```

TP 4: (works)
```json
{
  "id": "cmpl-81c20782c1f0445c8173d13d83afb55b",
  "object": "chat.completion",
  "created": 705538,
  "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": " {\n  \"name\": \"Alex Mercer\",\n  \"age\": 35,\n  \"gender\": \"Male\",\n  \"eye_color\": \"Ice Blue\",\n  \"hair_color\": \"Raven Black\",\n  \"height\": \"6'2\\\"\",\n  \"weight\": \"190 lbs\",\n\n  \"professional_background\": {\n    \"title\": \"Detective\",\n    \"department\": \"Centural City Police Department\",\n    \"service\": \"15 years\",\n    \"rank\": \"Senior Detective\",\n    \"badge_number\": \"718\",\n    \"specializations\": [\n      \"Homicide Investigation\",\n      \"Cyber Crime\",\n      \"Forensic Science\"\n    ]\n  },\n\n  \"skills\": [\n    {\n      \"name\": \"Keen Observation\",\n      \"level\": \"Expert\"\n    },\n    {\n      \"name\": \"Interrogation Techniques\",\n      \"level\": \"Master\"\n    },\n    {\n      \"name\": \"Firearms Proficiency\",\n      \"level\": \"Advanced\"\n    },\n    {\n      \"name\": \"Computer Hacking\",\n      \"level\": \"Intermediate\"\n    },\n    {\n      \"name\": \"Martial Arts\",\n      \"level\": \"Proficient\"\n    }\n  ],\n\n  \"latest_case\": {\n    \"name\": \"The Quantum Murders\",\n    \"status\": \"Ongoing\",\n    \"case_details\": {\n      \"type\": \"Serial Homicide\",\n      \"modus_operandi\": \"Victims are found disintegrated at a quantum level.\",\n      \"lead_suspect\": \"Unknown\",\n      \"evidence\": [\n        \"Unidentified quantum particles at crime scenes\",\n        \"Encrypted messages sent to local media outlets\"\n      ],\n      \"challenges\": [\n        \"Crime scenes lack physical evidence\",\n        \"Encrypted messages cannot be deciphered\"\n      ]\n    }\n  }\n}"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 59,
    "total_tokens": 552,
    "completion_tokens": 493
  }
}
```