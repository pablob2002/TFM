import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib as mpl, matplotlib.ticker as mticker
import seaborn as sns
import powerlaw
from lifelines import KaplanMeierFitter
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde, ks_2samp, percentileofscore

DATA="/sessions/funny-dreamy-hypatia/mnt/TFM/Datasets"
df=pd.read_json(f"{DATA}/voluntariosdanavalencia_old.json")
df['sender_id']=pd.to_numeric(df['sender_id'],errors='coerce').astype('Int64')
df['sender_username']=df.get('sender_username')
df['date']=pd.to_datetime(df['date'],errors='coerce')
if df['date'].dt.tz is not None: df['date']=df['date'].dt.tz_localize(None)
df=df.dropna(subset=['sender_id','date']).sort_values('date')
cutoff=df['date'].max()

# active users: >=40 msgs and >=5 days
g=df.groupby('sender_id')['date']; cnt=g.count(); span=(g.max()-g.min()).dt.total_seconds()/86400
active=cnt[(cnt>=40)&(span>=5)].index.tolist()

# coordinators by username
coord_users=['Marc_Nuba','LauAlejandro','Camelocota','MartaBalaguer','tipisibiza','Manuel_deRonde','CharlieWatchtower','ZRNATXO','docebocados','Esthx','AlcarolaGo']
coord_ids=[]
for un in coord_users:
    m=df[df['sender_username']==un]['sender_id'].unique()
    if len(m): coord_ids.append(int(m[0]))

def alpha_tpl_km(user_dates):
    wd=user_dates.sort_values().diff().dt.total_seconds().dropna()
    wd=wd[(wd>0)&np.isfinite(wd)].values
    if len(wd)<10: return np.nan
    tau=max((cutoff-user_dates.iloc[-1]).total_seconds(),0.0)
    kd=np.concatenate([wd,wd,[tau]]); ko=np.concatenate([np.ones(len(wd)),np.ones(len(wd)),[0]])
    kmf=KaplanMeierFitter(); kmf.fit(kd,event_observed=ko.astype(bool))
    fit=powerlaw.Fit(wd,discrete=True,verbose=False); xmin=float(getattr(fit.power_law,'xmin',np.nan))
    a0=float(getattr(fit.truncated_power_law,'alpha',2.0))
    kt=kmf.survival_function_.index.values; ks=kmf.survival_function_['KM_estimate'].values
    mask=(kt>=xmin)&(ks>0)&(kt>0)
    if not np.isfinite(xmin) or mask.sum()<5: return np.nan
    sx=float(np.interp(xmin,kt,ks))
    if sx<=0: return np.nan
    tt=kt[mask]; ss=ks[mask]/sx
    def tpl(t,a,l): return (t/xmin)**(-(a-1))*np.exp(-(t-xmin)*l)
    try:
        p,_=curve_fit(tpl,tt,ss,p0=[max(a0,1.01),1/wd.max()],bounds=([1.001,0],[20,np.inf]),maxfev=10000)
        return float(p[0])
    except: return np.nan

def alphas(ids):
    out=[]
    for u in ids:
        a=alpha_tpl_km(df[df['sender_id']==u]['date'])
        if np.isfinite(a): out.append(a)
    return np.array(out)

alpha_au_pool=alphas(active)
alpha_coord_i=alphas(coord_ids)
print("pool n=",len(alpha_au_pool)," coords n=",len(alpha_coord_i))
print("pool median",np.median(alpha_au_pool)," coord mean",alpha_coord_i.mean())

# ===== figure: single KDE with ECDF inset =====
CB_BLUE="#2c7fb8"; CB_PURPLE="#7a0177"; CB_ORANGE="#D55E00"
import seaborn as sns, matplotlib.ticker as mticker
sns.set_theme(style='ticks',context='paper')
mpl.rcParams.update({'figure.dpi':150,'savefig.dpi':300,'font.size':9,'axes.labelsize':10,
    'axes.titlesize':10,'legend.fontsize':8,'xtick.labelsize':8,'ytick.labelsize':8,
    'axes.spines.top':False,'axes.spines.right':False})

fig,ax=plt.subplots(figsize=(6.4,4.4))

# --- main: bootstrap KDE band ---
SAMPLE=len(alpha_coord_i); NB=2000; rng=np.random.default_rng(42)
pad=0.3
xg=np.linspace(min(alpha_au_pool.min(),alpha_coord_i.min())-pad,max(alpha_au_pool.max(),alpha_coord_i.max())+pad,500)
mat=np.zeros((NB,len(xg)))
for i in range(NB):
    s=rng.choice(alpha_au_pool,size=SAMPLE,replace=False)
    mat[i]=gaussian_kde(s,bw_method="scott")(xg)
lo=np.percentile(mat,5,axis=0); med=np.percentile(mat,50,axis=0); hi=np.percentile(mat,95,axis=0)
coord_kde=gaussian_kde(alpha_coord_i,bw_method="scott")(xg)
obs_ks=ks_2samp(alpha_coord_i,alpha_au_pool).statistic
null_ks=np.array([ks_2samp(rng.choice(alpha_au_pool,size=SAMPLE,replace=False),alpha_au_pool).statistic for _ in range(NB)])
p_ks=np.mean(null_ks>=obs_ks)
ax.fill_between(xg,lo,hi,color=CB_BLUE,alpha=0.25,label=f"Bootstrap 90% band (n={SAMPLE})")
ax.plot(xg,med,color=CB_BLUE,lw=1.6,label=f"Null median KDE (n={SAMPLE})")
ax.plot(xg,coord_kde,color=CB_PURPLE,lw=2.0,label=f"Coordinators (n={len(alpha_coord_i)})")
ax.fill_between(xg,0,coord_kde,color=CB_PURPLE,alpha=0.12)
ax.axvline(np.median(alpha_au_pool),color=CB_BLUE,lw=1.0,ls="--",alpha=0.8)
ax.axvline(np.median(alpha_coord_i),color=CB_PURPLE,lw=1.0,ls="--",alpha=0.8)
ax.set_xlabel(r"$\alpha$  (KM-corrected TPL)"); ax.set_ylabel("Density")
ax.legend(frameon=True,facecolor="white",edgecolor="0.7",framealpha=0.9,
          loc='upper center',bbox_to_anchor=(0.5,-0.16),ncol=3,
          title=f"KS permutation test: $p={p_ks:.2f}$")
sns.despine(ax=ax,offset=3)

# --- inset: ECDF percentiles ---
axin=ax.inset_axes([0.50,0.40,0.46,0.55])
sp=np.sort(alpha_au_pool); ey=np.arange(1,len(sp)+1)/len(sp)*100
coord_pct=np.array([percentileofscore(alpha_au_pool,a,kind="rank") for a in alpha_coord_i])
mean_pct=coord_pct.mean()
axin.plot(sp,ey,color=CB_BLUE,lw=1.3)
for pct,ls in [(25,":"),(75,":"),(50,"--")]:
    axin.axhline(pct,color="0.6",lw=0.7,ls=ls,alpha=0.9)
for a,p in zip(alpha_coord_i,coord_pct):
    axin.plot([a,a],[0,p],color="0.8",lw=0.4,ls=":",zorder=3)
axin.scatter(alpha_coord_i,coord_pct,color=CB_PURPLE,s=22,zorder=6,edgecolors="white",linewidths=0.5)
axin.axhline(mean_pct,color=CB_ORANGE,lw=1.0,ls="-.")
axin.set_ylim(0,100); axin.set_xlabel(r"$\alpha$",fontsize=8,labelpad=1)
axin.set_ylabel("Percentile in pool (%)",fontsize=7.5,labelpad=1)
axin.yaxis.set_major_formatter(mticker.PercentFormatter())
axin.tick_params(labelsize=7); axin.set_title("ECDF",fontsize=8,pad=2)
for s in ['top','right']: axin.spines[s].set_visible(False)

fig.savefig("coord_combined.png",dpi=300,bbox_inches="tight")
fig.savefig("coord_combined.pdf",bbox_inches="tight")
print("SAVED inset version")
