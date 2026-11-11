#!/usr/bin/env python3
"""Purple AI programming fine-tune: continue from the existing PurpleGPT checkpoint."""
import json, os, random, re, time
from pathlib import Path
import requests
import torch
from torch import nn
from torch.nn import functional as F
from huggingface_hub import snapshot_download

OUT = Path(os.environ["PURPLE_OUT"]).resolve()
WORK = Path(os.environ["PURPLE_WORK"]).resolve()
DATA = WORK / "programming_corpus.txt"
HF_REPO = "Kamiljdds/purpleaiV1.0"
REPOS = [
    "psf/requests", "pallets/flask", "expressjs/express",
    "rust-lang/rust-by-example", "fmtlib/fmt", "square/okio",
    "sinatra/sinatra", "lua/lua"
]
ALLOWED = {"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC"}
EXTS = {".py",".js",".ts",".tsx",".jsx",".java",".kt",".swift",".c",".h",".cc",".cpp",".hpp",".cs",".go",".rs",".rb",".php",".lua",".sql",".html",".css",".sh",".md"}
SKIP = {"node_modules","vendor","dist","build","target",".git","third_party","testdata","__pycache__"}

def download_code():
    headers = {"User-Agent":"PurpleAI-training-data-fetcher"}
    total, skipped = 0, []
    with DATA.open("w", encoding="utf-8") as out:
        for repo in REPOS:
            try:
                meta = requests.get("https://api.github.com/repos/"+repo, headers=headers, timeout=30)
                meta.raise_for_status()
                info = meta.json()
                license_id = ((info.get("license") or {}).get("spdx_id") or "UNKNOWN")
                if license_id not in ALLOWED:
                    print(f"POMINIĘTO {repo}: licencja {license_id}")
                    skipped.append(repo); continue
                branch = info["default_branch"]
                tree_r = requests.get(f"https://api.github.com/repos/{repo}/git/trees/{branch}?recursive=1", headers=headers, timeout=60)
                tree_r.raise_for_status()
                tree = tree_r.json().get("tree", [])
                chosen = []
                for item in tree:
                    p = item.get("path","")
                    parts = p.lower().split("/")
                    if item.get("type") != "blob" or any(x in SKIP for x in parts): continue
                    if Path(p).suffix.lower() not in EXTS: continue
                    if item.get("size", 0) < 80 or item.get("size", 0) > 100000: continue
                    chosen.append(p)
                random.Random(42).shuffle(chosen)
                got = 0
                for p in chosen[:35]:
                    try:
                        url = f"https://raw.githubusercontent.com/{repo}/{branch}/{p}"
                        r = requests.get(url, headers=headers, timeout=25)
                        if r.status_code != 200: continue
                        s = r.content.decode("utf-8", errors="ignore").replace("\x00","")
                        if len(s.strip()) < 50: continue
                        if re.search(r"(?i)(api[_-]?key|password|secret|access[_-]?token)\s*[:=]\s*['\"][^'\"]{6,}", s): 
                            continue
                        out.write(f"\n\n### REPO: {repo} | LICENSE: {license_id} | FILE: {p}\n{s[:90000]}\n")
                        got += 1; total += 1
                    except Exception: pass
                print(f"{repo}: pobrano {got} plików (licencja {license_id})")
            except Exception as e:
                print(f"Nie udało się pobrać {repo}: {e}")
                skipped.append(repo)
    if total < 5:
        raise RuntimeError("Pobrano za mało kodu z GitHuba. Sprawdź połączenie i spróbuj ponownie.")
    print(f"Łącznie plików źródłowych: {total}; pominięte repozytoria: {', '.join(skipped) or 'brak'}")
    return total

class AttentionHead(nn.Module):
    def __init__(self, embed, head_size, block_size):
        super().__init__()
        self.key=nn.Linear(embed,head_size,bias=False); self.query=nn.Linear(embed,head_size,bias=False)
        self.value=nn.Linear(embed,head_size,bias=False)
        self.register_buffer("mask",torch.tril(torch.ones(block_size,block_size)))
        self.dropout=nn.Dropout(0.0)
    def forward(self,x):
        _,n,_=x.shape; k,q=self.key(x),self.query(x)
        w=q@k.transpose(-2,-1)*(k.shape[-1]**-0.5)
        w=w.masked_fill(self.mask[:n,:n]==0,float("-inf"))
        return self.dropout(F.softmax(w,dim=-1))@self.value(x)
class MultiHeadAttention(nn.Module):
    def __init__(self,e,h,b):
        super().__init__(); d=e//h
        self.heads=nn.ModuleList([AttentionHead(e,d,b) for _ in range(h)])
        self.proj=nn.Linear(e,e); self.dropout=nn.Dropout(0.0)
    def forward(self,x): return self.dropout(self.proj(torch.cat([h(x) for h in self.heads],dim=-1)))
class Block(nn.Module):
    def __init__(self,e,h,b):
        super().__init__(); self.attn=MultiHeadAttention(e,h,b)
        self.ff=nn.Sequential(nn.Linear(e,4*e),nn.GELU(),nn.Linear(4*e,e),nn.Dropout(0.0))
        self.ln1=nn.LayerNorm(e); self.ln2=nn.LayerNorm(e)
    def forward(self,x):
        x=x+self.attn(self.ln1(x)); return x+self.ff(self.ln2(x))
class PurpleGPT(nn.Module):
    def __init__(self,b,e,h,l,v):
        super().__init__(); self.token_embedding=nn.Embedding(v,e); self.position_embedding=nn.Embedding(b,e)
        self.blocks=nn.Sequential(*[Block(e,h,b) for _ in range(l)])
        self.norm=nn.LayerNorm(e); self.head=nn.Linear(e,v)
    def forward(self,idx):
        pos=torch.arange(idx.shape[1],device=idx.device)
        return self.head(self.norm(self.blocks(self.token_embedding(idx)+self.position_embedding(pos))))

def get_vocab(saved, config, vocab_size, snap):
    stoi = saved.get("stoi") or saved.get("vocab") or config.get("stoi") or config.get("vocab")
    itos = saved.get("itos") or config.get("itos")
    chars = saved.get("chars") or config.get("chars")
    if isinstance(stoi,dict):
        stoi={str(k):int(v) for k,v in stoi.items()}
        if itos is None: itos={v:k for k,v in stoi.items()}
    elif isinstance(chars,(list,tuple,str)):
        chars=list(chars); stoi={str(c):i for i,c in enumerate(chars)}; itos={i:str(c) for i,c in enumerate(chars)}
    if isinstance(itos,(list,tuple)): itos={i:str(c) for i,c in enumerate(itos)}
    if isinstance(itos,dict): itos={int(k):str(v) for k,v in itos.items()}
    if not isinstance(stoi,dict) or not isinstance(itos,dict):
        for vf in Path(snap).glob("**/vocab.json"):
            try:
                v=json.loads(vf.read_text(encoding="utf-8"))
                if isinstance(v,dict) and isinstance(v.get("stoi"),dict):
                    stoi={str(k):int(x) for k,x in v["stoi"].items()}
                    itos={int(x):str(k) for k,x in stoi.items()}; break
                if isinstance(v,dict) and all(isinstance(x,int) for x in v.values()):
                    stoi={str(k):int(x) for k,x in v.items()}; itos={x:k for k,x in stoi.items()}; break
                if isinstance(v,list):
                    stoi={str(c):i for i,c in enumerate(v)}; itos={i:str(c) for i,c in enumerate(v)}; break
            except Exception: pass
    if not isinstance(stoi,dict) or not isinstance(itos,dict) or len(itos)<vocab_size:
        raise RuntimeError("Nie znaleziono poprawnej mapy znaków w checkpointcie ani vocab.json. Trening zatrzymany — nie tworzę losowego modelu.")
    return stoi,itos

def main():
    WORK.mkdir(parents=True,exist_ok=True); OUT.mkdir(parents=True,exist_ok=True)
    print("Pobieram bazowy model Purple AI z Hugging Face...")
    snap=snapshot_download(repo_id=HF_REPO, allow_patterns=["*.pt","*.pth","*.bin","*.json","*.safetensors"])
    candidates=list(Path(snap).rglob("purple_ai_latest.pt")) or list(Path(snap).rglob("*.pt")) or list(Path(snap).rglob("*.pth"))
    if not candidates: raise RuntimeError("Repo HF nie zawiera pliku checkpointu .pt/.pth.")
    base=candidates[0]
    try: saved=torch.load(base,map_location="cpu",weights_only=True)
    except TypeError: saved=torch.load(base,map_location="cpu")
    if not isinstance(saved,dict): raise RuntimeError("Checkpoint ma nieobsługiwany format.")
    state=next((saved[k] for k in ("model","model_state_dict","state_dict","model_weights") if isinstance(saved.get(k),dict)),None)
    if state is None and saved and all(torch.is_tensor(v) for v in saved.values()): state=saved
    if state is None: raise RuntimeError("Nie znaleziono wag modelu w checkpointcie.")
    if any(k.startswith("module.") for k in state): state={k.removeprefix("module."):v for k,v in state.items()}
    tw=state.get("token_embedding.weight"); pw=state.get("position_embedding.weight")
    if tw is None or pw is None: raise RuntimeError("Wagi nie pasują do architektury PurpleGPT z api/app.py.")
    vocab_size,embed=map(int,tw.shape); block_size=int(pw.shape[0])
    cfg=saved.get("config") or {}
    layer_ids={int(k.split(".")[1]) for k in state if k.startswith("blocks.") and k.split(".")[1].isdigit()}
    head_ids={int(k.split(".")[4]) for k in state if len(k.split("."))>5 and k.startswith("blocks.") and k.split(".")[2:4]==["attn","heads"] and k.split(".")[4].isdigit()}
    layers=int(cfg.get("layers",max(layer_ids)+1 if layer_ids else 0))
    heads=int(cfg.get("heads",max(head_ids)+1 if head_ids else 0))
    if not layers or not heads or embed%heads: raise RuntimeError("Nie można bezpiecznie odtworzyć liczby warstw/głów z checkpointu.")
    tokenizer=str(cfg.get("tokenizer",saved.get("tokenizer","legacy-char"))).lower()
    if tokenizer in ("utf8-bytes-v1","byte","bytes","utf8"):
        if vocab_size!=256: raise RuntimeError("Tokenizer bajtowy ma nieprawidłowy rozmiar.")
        tokenizer="utf8-bytes-v1"; stoi=itos=None
    else:
        tokenizer="legacy-char"; stoi,itos=get_vocab(saved,cfg,vocab_size,snap)
    device="cuda" if torch.cuda.is_available() else "cpu"
    if device!="cuda": raise RuntimeError("PyTorch nie widzi karty NVIDIA/CUDA. Nic nie trenowałem. Uruchom ponownie po instalacji wersji PyTorch CUDA.")
    print(f"Model bazowy: {base.name} | {vocab_size=} {block_size=} {embed=} {heads=} {layers=} | GPU: {torch.cuda.get_device_name(0)}")
    if not DATA.exists(): download_code()
    raw=DATA.read_text(encoding="utf-8",errors="ignore")
    if tokenizer=="utf8-bytes-v1": ids=list(raw.encode("utf-8"))
    else:
        unk=stoi.get("<unk>",stoi.get("[UNK]",0)); ids=[stoi.get(c,unk) for c in raw]
    if len(ids)<block_size+2: raise RuntimeError("Dataset jest za krótki dla kontekstu modelu.")
    data=torch.tensor(ids,dtype=torch.long)
    model=PurpleGPT(block_size,embed,heads,layers,vocab_size)
    model.load_state_dict(state,strict=True); model.to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5)
    batch_size=2
    steps=20000
    print(f"Rozpoczynam fine-tuning: {steps} kroków, batch={batch_size}, kontekst={block_size}.")
    model.train(); start=time.time()
    for step in range(steps):
        starts=torch.randint(0,len(data)-block_size-1,(batch_size,))
        x=torch.stack([data[i:i+block_size] for i in starts]).to(device)
        y=torch.stack([data[i+1:i+block_size+1] for i in starts]).to(device)
        logits=model(x); loss=F.cross_entropy(logits.reshape(-1,vocab_size),y.reshape(-1))
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
        if (step+1)%100==0 or step==0:
            print(f"krok {step+1}/{steps} | loss={loss.item():.4f} | {((time.time()-start)/max(1,step+1)):.2f}s/krok",flush=True)
        if (step+1)%500==0 or step==steps-1:
            config={"tokenizer":tokenizer,"vocab_size":vocab_size,"block_size":block_size,"embed":embed,"heads":heads,"layers":layers}
            pack={"step":step+1,"model":model.state_dict(),"config":config,"tokenizer":tokenizer}
            if stoi is not None:
                pack["stoi"]=stoi; pack["itos"]={str(k):v for k,v in itos.items()}; pack["vocab"]=stoi
            torch.save(pack,OUT/"purple_ai_latest.pt")
            if stoi is not None:
                (OUT/"vocab.json").write_text(json.dumps({"stoi":stoi,"itos":{str(k):v for k,v in itos.items()}},ensure_ascii=False),encoding="utf-8")
            else:
                (OUT/"vocab.json").write_text(json.dumps({"type":"utf8-bytes-v1","vocab_size":256}),encoding="utf-8")
            (OUT/"config.json").write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding="utf-8")
            print(f"Zapisano checkpoint: {OUT/'purple_ai_latest.pt'}",flush=True)
    print("\nGOTOWE. Oryginalne wagi HF nie zostały zmienione.")
    print(f"Wyniki: {OUT}")

if __name__=="__main__":
    main()
