"""
mineskaffold.py
Lean pandas + stdlib scaffold for theme mining / tagging under 128 k-token API cap.
Edit HOST / TOKEN, tweak ratios if needed.
"""

import json, math, time, pathlib, http.client, collections
import pandas as pd
from difflib import SequenceMatcher

# ────────── helpers ──────────
def n_tokens(txt, chars_per_tok=4):            # fast ≈ token counter
    return math.ceil(len(txt) / chars_per_tok)

def similar(a, b, th=0.82):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() >= th

def merge(counter, th=0.82):                   # cheap O(N²) synonym merge
    canon = {}
    for t in counter:
        k = next((c for c in canon if similar(t, c, th)), t)
        canon.setdefault(k, []).append(t)
    out = collections.Counter()
    for k, grp in canon.items():
        out[k] = sum(counter[g] for g in grp)
    return out

# ────────── core class ──────────
class ThemeMiner:
    MAX_TOK = 128_000; RESERVED = 28_000; IN = MAX_TOK - RESERVED; OUT = 1_000
    SYS_ANALYST = "You are an analyst. Return only valid JSON."
    SYS_CLASS   = "You are a classifier. JSON only."

    def __init__(self, host, endpoint="/v1/chat/completions",
                 model="gpt-4o-mini", token=None):
        self.host, self.endpoint, self.model = host, endpoint, model
        self.headers = {"Content-Type": "application/json"}
        if token: self.headers["Authorization"] = f"Bearer {token}"

    # I/O --------------------------------------------------------------
    def load(self, path):
        p = pathlib.Path(path)
        if p.suffix.lower() in {".csv", ".txt"}:   return pd.read_csv(p)
        if p.suffix.lower() in {".xls", ".xlsx"}:  return pd.read_excel(p, engine="openpyxl")
        raise ValueError("unsupported file type")

    def _post(self, body):
        c = http.client.HTTPSConnection(self.host, timeout=90)
        c.request("POST", self.endpoint, body, headers=self.headers)
        r = c.getresponse()
        if r.status != 200: raise RuntimeError(f"API {r.status}: {r.reason}")
        out = r.read().decode(); c.close(); return out

    # theme discovery --------------------------------------------------
    def discover(self, df, text_col, epochs=3, top_k=100, th=0.82):
        counts = collections.Counter()
        for e in range(epochs):
            for chunk in self._chunks(df.sample(frac=1, random_state=e)[text_col]):
                counts.update(self._theme_call(chunk)); time.sleep(0.05)
        merged = merge(counts, th)
        return (pd.DataFrame(merged.items(), columns=["theme", "count"])
                .nlargest(top_k, "count").reset_index(drop=True))

    def _theme_call(self, series):
        prompt = (f"Input:\n```{'\n\n'.join(series)}```\n\n"
                  "Task: Identify recurring themes (concepts / topics). "
                  "Return JSON array (keys 'theme','count') ≤100 items sorted desc.")
        payload = {"model": self.model,
                   "messages": [
                       {"role": "system", "content": self.SYS_ANALYST},
                       {"role": "user",   "content": prompt}],
                   "max_tokens": self.OUT, "temperature": 0}
        data = json.loads(json.loads(self._post(json.dumps(payload)))
                          ["choices"][0]["message"]["content"])
        return collections.Counter({d["theme"]: int(d["count"]) for d in data})

    # classification ---------------------------------------------------
    def classify(self, df, text_col, themes_df, allow_none=True):
        theme_block = "\n".join(f"{i+1}. {t}" for i, t in enumerate(themes_df.iloc[:, 0]))
        tb_tokens   = n_tokens(theme_block) + 40
        assign, batch, tok = {}, [], tb_tokens

        for idx, txt in df[text_col].items():
            t = n_tokens(txt)
            if t > self.IN - tb_tokens: raise ValueError(f"row {idx} too big")
            if tok + t > self.IN: self._flush(batch, theme_block, assign); batch, tok = [], tb_tokens
            batch.append((idx, txt)); tok += t
        self._flush(batch, theme_block, assign)

        out = df.copy()
        out["theme"] = out.index.map(lambda i: assign.get(i, "NONE" if allow_none else None))
        return out

    def _flush(self, batch, theme_block, assign):
        if not batch: return
        records = "\n".join(f"{i}. {t}" for i, t in batch)
        user = (f"Themes\n------\n{theme_block}\n\n"
                "Task\n----\nFor each record choose best matching theme verbatim; else NONE.\n\n"
                "Output\n------\nJSON array {{\"idx\":<row_id>,\"theme\":\"<theme|NONE>\"}}.\n\n"
                f"Records\n-------\n{records}")
        payload = {"model": self.model,
                   "messages": [
                       {"role": "system", "content": self.SYS_CLASS},
                       {"role": "user",   "content": user}],
                   "max_tokens": self.OUT, "temperature": 0}
        res = json.loads(json.loads(self._post(json.dumps(payload)))
                         ["choices"][0]["message"]["content"])
        for d in res: assign[d["idx"]] = d["theme"]

    # chunk helper -----------------------------------------------------
    def _chunks(self, series):
        start, tok = 0, 0
        for i, t in enumerate(series):
            nt = n_tokens(t)
            if nt > self.IN: raise ValueError(f"row {i} too big")
            if tok + nt > self.IN: yield series.iloc[start:i]; start, tok = i, 0
            tok += nt
        yield series.iloc[start:]

# demo -----------------------------------------------------------------
if __name__ == "__main__":
    miner   = ThemeMiner("llm.internal.corp", token="TOKEN")
    df_raw  = miner.load("comments.csv")
    themes  = miner.discover(df_raw, "comment_text")          # phase 1
    themes.to_csv("themes.csv", index=False)
    df_tag  = miner.classify(df_raw, "comment_text", themes)  # phase 2
    df_tag.to_parquet("comments_tagged.parquet")
