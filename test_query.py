import requests
import json

r = requests.post("http://localhost:8080/query", json={"question": "RAG是什么"})
data = r.json()

with open("test_result.txt", "w", encoding="utf-8") as f:
    f.write("=== 回答 ===\n")
    f.write(data["answer"])
    f.write("\n\n=== 引用来源 ===\n")
    for s in data["sources"]:
        f.write(f"[{s['source']}] 相似度:{s['score']:.3f}\n")
        f.write(f"  {s['text'][:150]}...\n\n")

print("done")
