#!/usr/bin/env python3
"""API-format workflow -> UI-format workflow (the kind ComfyUI saves and opens by drag and drop).
usage: make_ui_workflow.py <object_info.json> <api.json> <ui.json>
Node layout is by dependency depth. Widget order comes from the node definitions."""
import json, sys
oi = json.load(open(sys.argv[1])); api = json.load(open(sys.argv[2]))["prompt"]
PRIMS = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO", "COMFY_DYNAMICCOMBO_V3"}

def is_widget(spec):
    return isinstance(spec[0], list) or spec[0] in PRIMS

def is_link(v):
    return isinstance(v, list) and len(v) == 2 and isinstance(v[0], str) and v[0] in api

depth = {}
def d(nid):
    if nid not in depth:
        depth[nid] = 0
        depth[nid] = 1 + max([d(v[0]) for v in api[nid]["inputs"].values() if is_link(v)], default=-1)
    return depth[nid]
for nid in api: d(nid)

links, nodes, out_links = [], [], {}
order = sorted(api, key=lambda n: (depth[n], int(n)))
col_y = {}
for idx, nid in enumerate(order):
    node = api[nid]; info = oi[node["class_type"]]
    specs = {**info["input"].get("required", {}), **info["input"].get("optional", {})}
    names = info["input_order"].get("required", []) + info["input_order"].get("optional", [])
    inputs, widgets, stop = [], [], False
    for name in names:
        spec = specs[name]; value = node["inputs"].get(name)
        widget = is_widget(spec)
        typ = "COMBO" if isinstance(spec[0], list) else spec[0]
        if is_link(value):
            lid = len(links) + 1
            src = api[value[0]]; src_type = oi[src["class_type"]]["output"][value[1]]
            links.append([lid, int(value[0]), value[1], int(nid), len(inputs), src_type if isinstance(src_type, str) else "COMBO"])
            out_links.setdefault((value[0], value[1]), []).append(lid)
            entry = {"name": name, "type": typ, "link": lid}
            if widget: entry["widget"] = {"name": name}
            inputs.append(entry)
        elif not widget:
            inputs.append({"name": name, "type": typ, "link": None})
        if widget and not stop:
            if typ == "COMFY_DYNAMICCOMBO_V3":
                stop = True  # leave this and later widgets at their defaults
                continue
            opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
            if value is None or is_link(value):
                value = opts.get("default", spec[0][0] if isinstance(spec[0], list) and spec[0] else (opts.get("options") or [""])[0] if typ == "COMBO" else 0 if typ in ("INT", "FLOAT") else "")
            widgets.append(value)
            if opts.get("control_after_generate") or name in ("seed", "noise_seed"):
                widgets.append("fixed")
    outputs = [{"name": (info.get("output_name") or info["output"])[i], "type": t if isinstance(t, str) else "COMBO", "links": None, "slot_index": i}
               for i, t in enumerate(info["output"])]
    rows = max(len(inputs) + len(widgets), len(outputs), 1)
    big = any(isinstance(w, str) and (len(w) > 60 or "\n" in w) for w in widgets)
    size = [400, 60 + 26 * rows + (160 if big else 0)]
    x = 60 + depth[nid] * 460; y = col_y.get(depth[nid], 60); col_y[depth[nid]] = y + size[1] + 50
    nodes.append({"id": int(nid), "type": node["class_type"], "pos": [x, y], "size": size, "flags": {}, "order": idx, "mode": 0,
                  "inputs": inputs, "outputs": outputs, "properties": {"Node name for S&R": node["class_type"]}, "widgets_values": widgets})
for n in nodes:
    for o in n["outputs"]:
        ids = out_links.get((str(n["id"]), o["slot_index"]))
        if ids: o["links"] = ids
ui = {"last_node_id": max(int(n) for n in api), "last_link_id": len(links), "nodes": nodes, "links": links, "groups": [], "config": {}, "extra": {}, "version": 0.4}
json.dump(ui, open(sys.argv[3], "w"), indent=1)
print(sys.argv[3], len(nodes), "nodes", len(links), "links")
