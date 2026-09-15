class CodeRetriever:
    def __init__(self, memory):
        self.memory = memory

    def add_code(self, code, module_name):
        text = f"[MODULE: {module_name}]\n{code}"
        self.memory.add_to_domain(text, "coding", quality="high")

    def retrieve(self, query, k=2):
        results = self.memory.search_domain(query, "coding", k=k)
        return "\n\n".join([
            r["text"] for r in results
            if "[MODULE:" in r["text"]
        ])