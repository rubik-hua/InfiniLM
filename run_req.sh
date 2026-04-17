
unset http_proxy https_proxy all_proxy ALL_PROXY  
# curl -N -H "Content-Type: application/json"      -X POST http://127.0.0.1:8103/v1/chat/completions      -d '{
#        "model": "/home/ubuntu/models/Qwen/Qwen3-0.6B",
#        "messages": [
#          {"role": "user", "content": "What is the capital of France?"}
#        ],
#        "temperature": 1.0,
#        "top_k": 50,
#        "top_p": 0.8,
#        "max_tokens": 32,
#        "stream": false
#      }'


curl -N -H "Content-Type: application/json" \
     -X POST http://127.0.0.1:8103/chat/completions \
     -d '{
       "model": "/home/ubuntu/models/Qwen/Qwen3-0.6B/",
       "messages": [
         {"role": "user", "content": "山东最高的山是？"}
       ],
       "temperature": 1.0,
       "top_k": 50,
       "top_p": 0.8,
       "max_tokens": 4,
       "stream": false
     }'


