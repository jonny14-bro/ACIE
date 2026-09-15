class ProjectState:
    def __init__(self):
        self.modules = {}
        self.todo = []
        self.completed = []
        self.order = []

    def add_task(self, task):
        self.todo.append({"task": task, "done": False})

    def mark_done(self, task):
        for t in self.todo:
            if t["task"] == task:
                t["done"] = True
                self.completed.append(task)

    def add_module(self, name, code):
        self.modules[name] = code
        self.order.append(name)

    def get_completed_modules(self):
        return list(self.modules.keys())

    def get_completed_class_names(self) -> list[str]:
        """Return actual class names (e.g. 'Server') from all completed modules."""
        import re
        names = []
        for code in self.modules.values():
            for line in code.splitlines():
                m = re.match(r"^class (\w+)", line.strip())
                if m:
                    names.append(m.group(1))
        return names

    def get_context(self, max_chars: int = 2000) -> str:
        """
        Return the most recent completed modules as context for the next step.
        Shows ALL completed modules but caps total chars to avoid context overflow.
        Earlier modules are summarised as class signatures only (no body)
        to keep the model aware of them without blowing the context window.
        """
        if not self.modules:
            return ""

        names = list(self.modules.keys())
        context = ""

        # Full code for the last 2 modules (most relevant for continuity)
        full_names = names[-2:]
        summary_names = names[:-2]

        # Summarise older modules: just class/def signatures, no bodies
        if summary_names:
            context += "# Previously completed modules (signatures only):\n"
            for name in summary_names:
                code = self.modules[name]
                sigs = []
                for line in code.splitlines():
                    stripped = line.strip()
                    if stripped.startswith("class ") or (stripped.startswith("def ") and not line.startswith(" ")):
                        sigs.append(stripped.split(":")[0] + ": ...")
                if sigs:
                    context += f"# {name}.py: {', '.join(sigs)}\n"
            context += "\n"

        # Full code for recent modules
        for name in full_names:
            snippet = self.modules[name]
            context += f"\n# FILE: {name}.py\n{snippet}\n"

        # Hard cap to avoid context overflow
        if len(context) > max_chars:
            context = context[-max_chars:]

        return context
