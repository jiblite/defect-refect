"""Export the trained detector (CNN), the second-stage effect deciders and the settings readers
into one little-endian binary file the app reads (Resources/models.bin).
Layout: magic 'DRFX', version u32, then sections. All floats f32, ints i32."""
import sys, pickle, struct, numpy as np, torch
sys.path.insert(0, sys.argv[1])            # folder with model.py
import os
if os.environ.get("DR_MODEL27"): from model27 import Net, FX
else: from model import Net, FX
out=open(sys.argv[2],'wb')
def u32(v): out.write(struct.pack('<I',v))
def i32(v): out.write(struct.pack('<i',v))
def f32a(a): a=np.asarray(a,np.float32).ravel(); u32(len(a)); out.write(a.tobytes())
def s(t): b=t.encode(); u32(len(b)); out.write(b)
out.write(b'DRFX'); u32(1)
# ---- CNN with batch norms folded into the convolutions
net=Net(); net.load_state_dict(torch.load(sys.argv[3],map_location='cpu')); net.eval()
def fold(conv,bn):
    w=conv.weight.detach().numpy(); b=conv.bias.detach().numpy() if conv.bias is not None else np.zeros(w.shape[0])
    g=bn.weight.detach().numpy()/np.sqrt(bn.running_var.numpy()+bn.eps)
    return w*g.reshape(-1,*([1]*(w.ndim-1))), (b-bn.running_mean.numpy())*g+bn.bias.detach().numpy()
convs=[]
for blk in net.c:
    for ci,bi in ((0,1),(3,4)): convs.append(fold(blk[ci],blk[bi]))
u32(len(convs))
for w,b in convs: u32(w.shape[0]); u32(w.shape[1]); f32a(w); f32a(b)
pools=[(2,2),(2,1),(2,1),(2,1),(2,1)]; u32(len(pools))
for p in pools: u32(p[0]); u32(p[1])
t1=fold(net.t[0],net.t[1]); t2=fold(net.t[3],net.t[4])
for (w,b),(k,pad,dil) in ((t1,(5,2,1)),(t2,(5,4,2))): u32(w.shape[0]); u32(w.shape[1]); u32(k); u32(pad); u32(dil); f32a(w); f32a(b)
for lin in (net.cls,net.reg): u32(lin.weight.shape[0]); u32(lin.weight.shape[1]); f32a(lin.weight.detach().numpy()); f32a(lin.bias.detach().numpy())
u32(len(FX))
for f in FX: s(f)
# ---- gradient-boosted trees
def trees(m):
    base=np.asarray(m._baseline_prediction).ravel(); f32a(base)
    u32(len(m._predictors))
    for it in m._predictors:
        for p in it:
            n=p.nodes; u32(len(n))
            for r in n:
                i32(int(r['feature_idx'])); out.write(struct.pack('<f',float(r['num_threshold'])))
                i32(int(r['left'])); i32(int(r['right'])); out.write(struct.pack('<B',int(r['is_leaf']))); out.write(struct.pack('<B',int(r['missing_go_to_left'])))
                out.write(struct.pack('<f',float(r['value'])))
st=pickle.load(open(sys.argv[4],'rb')); u32(len(FX))
for f in FX: s(f); trees(st[f])
sm=pickle.load(open(sys.argv[5],'rb')); u32(len(sm))
for fx,params in sm.items():
    params={k:v for k,v in params.items() if v[0] in ('log','lin','cat')}   # 'signedlog' readers are not used by the app
    s(fx); u32(len(params))
    for k,(kind,m,cats) in params.items():
        s(k); s(kind); u32(len(cats or [])); [s(c) for c in (cats or [])]; trees(m)
out.close()
