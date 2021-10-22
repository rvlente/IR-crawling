import orjson
from collections import defaultdict
from tqdm import tqdm

with open("cache/state.json", "r") as f:
    state = orjson.loads(f.read())


new_que_inner = defaultdict(list)

for heapurl in tqdm(state["que"]):
    new_que_inner[str(heapurl["host_count"])].append(heapurl["url"])

# new_que_inner = dict(**new_que_inner)

new_que = {
    "que": new_que_inner
}

state["que"] = new_que

with open("cache/state3.json", "wb") as f:
    f.write(orjson.dumps(state))