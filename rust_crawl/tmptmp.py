import json

with open("cache/state.json", "r") as f:
    state = json.load(f)

print(*state["que"], sep="\n")