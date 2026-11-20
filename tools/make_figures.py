"""Regenerate results/figures/fig_drift_v2.png and fig_vision_use.png from results/.

Inputs: results/summary.json (drift, CIs, noise floor) and results/blank_ci.json
(blank-camera drift, first 8 steps, mean and 95% episode-bootstrap CI on the 60-frame
diagnostic subset; derived from the per-frame diagnostics of the run).
Run from the repository root:  python tools/make_figures.py
"""
import json
from pathlib import Path
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
O = R / "figures"
BG,INK,GREY,RULE,ACC="#F6F3EC","#1F2328","#6B7078","#DDD8CC","#D8572A"
mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Inter","DejaVu Sans"],"figure.facecolor":BG,"axes.facecolor":BG,"savefig.facecolor":BG,
 "text.color":INK,"axes.labelcolor":INK,"xtick.color":INK,"ytick.color":INK,"axes.edgecolor":INK,
 "axes.spines.top":False,"axes.spines.right":False})
S=json.load(open(R/"summary.json")); res=S["res"]; fl=S["floor"]["h8"]
_b=json.load(open(R/"blank_ci.json")); BE=tuple(_b["blank_endo"]); BA=tuple(_b["blank_all"])
# ---------------- fig 1
def fig_drift(slide=False):
    fig,ax=plt.subplots(figsize=(7.4,5.7) if slide else (9.6,7.0))
    xs=[1,2,3,4,5]
    st={"SMOKE":(ACC,3.2,"-"),"BLOOD":(INK,1.8,"-"),"FOG":("#4A5560",1.8,"-"),"GLARE":("#7D858F",1.8,"-"),
        "OCCLUSION":("#A29E92",1.8,"-"),"GAUSS":("#4A5560",1.6,(0,(4,3))),"BRIGHTNESS":("#8A8F98",1.6,(0,(1,2.5)))}
    nm={"SMOKE":"Smoke","BLOOD":"Blood","FOG":"Fog","GLARE":"Glare","OCCLUSION":"Occlusion","GAUSS":"Gaussian noise","BRIGHTNESS":"Brightness"}
    ax.fill_between([0.85,5.05],fl["lo"],fl["hi"],color="#E6DFCF",lw=0,zorder=0)
    ax.hlines(fl["mean"],0.85,5.05,color=GREY,lw=1.2,ls=(0,(5,3)),zorder=1)
    ax.hlines(BE[0],0.85,5.05,color=INK,lw=1.6,zorder=1)
    ends=[]
    for k,(c,lw,ls) in st.items():
        r=res["h8|"+k]; D=np.array(r["D"])
        if k=="SMOKE": ax.fill_between(xs,r["Dlo"],r["Dhi"],color=ACC,alpha=0.18,lw=0,zorder=2)
        ax.plot(xs,D,color=c,lw=lw,ls=ls,marker="o" if k=="SMOKE" else None,ms=7,zorder=4 if k=="SMOKE" else 3)
        ends.append([D[-1],k])
    ends.sort(); y=[e[0] for e in ends]; gap=0.085
    for _ in range(200):
        for i in range(1,len(y)):
            if y[i]-y[i-1]<gap:
                m=(y[i]+y[i-1])/2; y[i-1]=m-gap/2; y[i]=m+gap/2
    for (d,k),yy in zip(ends,y):
        c=st[k][0]; c=c if k!="OCCLUSION" else "#7E7A6E"
        ax.plot([5.04,5.17],[d,yy],color=c,lw=0.8,zorder=2)
        ax.text(5.2,yy,nm[k],va="center",ha="left",fontsize=15,color=c,fontweight="bold" if k=="SMOKE" else "normal")
    ax.text(0.92,1.215,f"re-seed the clean image: {fl['mean']:.2f} mm  (95% CI {fl['lo']:.2f} to {fl['hi']:.2f})",fontsize=14,color=GREY,va="bottom",ha="left")
    ax.text(0.92,BE[0]+0.012,f"endoscope fully blanked: {BE[0]:.2f} mm",fontsize=14,color=INK,va="bottom",ha="left",fontweight="bold")
    ax.set_xlim(0.85,6.45); ax.set_ylim(0,1.38); ax.set_xticks(xs)
    ax.set_xticks(xs); ax.tick_params(labelsize=15,length=4)
    ax.set_xlabel("artifact severity (1 mild, 5 heavy)",fontsize=15,labelpad=8)
    ax.set_ylabel("drift of predicted tool motion (mm)",fontsize=15,labelpad=8)
    ax.spines["bottom"].set_bounds(1,5)
    ax.yaxis.grid(True,color=RULE,lw=0.8); ax.set_axisbelow(True)
    if not slide: fig.text(0.02,0.975,"Degrading the endoscope view moves the prediction\nno more than re-seeding the model",fontsize=21,fontweight="bold",va="top",ha="left",linespacing=1.15)
    if not slide: fig.text(0.02,0.012,"GR00T-H-N1.7, 400 training frames, first 8 steps, endoscope view corrupted. Smoke band: 95% CI over episodes.\nGrey dashed: Gaussian noise and brightness controls. Blank-endoscope value from a 60-frame subset.",fontsize=10.5,color=GREY,va="bottom")
    if slide:
        for t in list(ax.texts):
            if "re-seed the clean" in t.get_text(): t.set_text("re-seed: 1.10 mm (95% CI 1.00 to 1.19)"); t.set_fontsize(13)
            elif "fully blanked" in t.get_text(): t.set_fontsize(13); t.set_bbox(dict(fc=BG,ec="none",pad=1.5,alpha=0.9))
            elif t.get_text() in ("Gaussian noise","Occlusion","Smoke","Fog","Blood","Glare","Brightness"): t.set_fontsize(14)
        ax.set_xlim(0.85,7.0)
        fig.subplots_adjust(left=0.13,right=0.98,top=0.97,bottom=0.15)
    else: fig.subplots_adjust(left=0.1,right=0.98,top=0.85,bottom=0.2)
    fig.savefig(str(O/"fig_drift_v2.png"),dpi=200); plt.close(fig)

# ---------------- fig 2
def fig_use(slide=False):
    items=[("re-seed (noise floor)",fl["mean"],fl["lo"],fl["hi"],"#B9B4A6"),
           ("blank endoscope",*BE,"#4A5560"),
           ("smoke, s3",*[res["h8|SMOKE"][k][2] for k in("D","Dlo","Dhi")],"#4A5560"),
           ("fog, s3",*[res["h8|FOG"][k][2] for k in("D","Dlo","Dhi")],"#4A5560"),
           ("occlusion, s3",*[res["h8|OCCLUSION"][k][2] for k in("D","Dlo","Dhi")],"#4A5560"),
           ("blood, s3",*[res["h8|BLOOD"][k][2] for k in("D","Dlo","Dhi")],"#4A5560"),
           ("glare, s3",*[res["h8|GLARE"][k][2] for k in("D","Dlo","Dhi")],"#4A5560"),
           ("all cameras blank",*BA,ACC)]
    fig,ax=plt.subplots(figsize=(7.4,5.7) if slide else (9.6,7.0))
    n=len(items)
    for i,(l,v,lo,hi,c) in enumerate(items):
        yy=n-1-i
        ax.barh(yy,v,color=c,height=0.62,zorder=2)
        ax.plot([lo,hi],[yy,yy],color=INK,lw=1.6,zorder=3); 
        for e in (lo,hi): ax.plot([e,e],[yy-0.11,yy+0.11],color=INK,lw=1.6,zorder=3)
        ax.text(hi+0.05,yy,f"{v:.2f}",va="center",fontsize=16,fontweight="bold" if c==ACC else "normal",color=ACC if c==ACC else INK,bbox=dict(fc=BG,ec="none",pad=1.5),zorder=5)
    ax.set_yticks(range(n)); ax.set_yticklabels([x[0] for x in items][::-1],fontsize=15)
    ax.axvline(fl["mean"],color=GREY,lw=1.1,ls=(0,(5,3)),zorder=1)
    ax.set_xlim(0,2.9); ax.set_ylim(-0.6,n-0.4)
    ax.set_xlabel("drift of predicted tool motion, first 8 steps (mm)",fontsize=15,labelpad=8)
    ax.tick_params(axis="x",labelsize=14); ax.tick_params(axis="y",length=0)
    ax.spines["left"].set_visible(False); ax.xaxis.grid(True,color=RULE,lw=0.8); ax.set_axisbelow(True)
    if not slide: fig.text(0.02,0.975,"What actually moves the robot's\npredicted motion",fontsize=21,fontweight="bold",va="top",linespacing=1.15)
    if not slide: fig.text(0.02,0.012,"GR00T-H-N1.7. Artifacts: 400 frames, mean and 95% CI over episodes. Blank-camera rows: 60-frame subset.\nDashed line: re-seed noise floor. Smoke, fog, occlusion, blood, glare: endoscope view only, severity 3.",fontsize=10.5,color=GREY,va="bottom")
    if slide:
        ax.set_yticklabels([x[0] for x in items][::-1],fontsize=14); ax.set_xlabel("drift, first 8 steps (mm)",fontsize=14)
        ax.set_xlim(0,2.9)
        fig.subplots_adjust(left=0.34,right=0.97,top=0.97,bottom=0.16)
    else: fig.subplots_adjust(left=0.27,right=0.97,top=0.85,bottom=0.2)
    fig.savefig(str(O/"fig_vision_use.png"),dpi=200); plt.close(fig)
fig_drift(); fig_use()
