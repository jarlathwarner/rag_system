# RAG System to retrieve threat guidance on different malware and network attacks
1. Download dataset from HuggingFace.
2. Authenticate using 
```
hf auth login
```

## System Objective
Enter a query regarding different attacks.
E.g., 
```
what piece of malware has an ID of S000?
```
```
how do I detect <some malware name> and what platforms does it target?
```

## ToDo
[] set up inference serving using vLLM

[] profile using Nvidia NSight

[] incorporate RAG metrics
