import { useState, useMemo, useRef } from "react";

// ═══════════════════════════════════════════════════════════════════════════════
// MATH UTILITIES
// ═══════════════════════════════════════════════════════════════════════════════
const parseData = str => str.split(/[\s,;]+/).map(Number).filter(n => !isNaN(n) && n !== null && str.trim() !== "");
const sum = arr => arr.reduce((a,b) => a+b, 0);
const mean = arr => sum(arr)/arr.length;
const median = arr => { const s=[...arr].sort((a,b)=>a-b),m=Math.floor(s.length/2); return s.length%2?s[m]:(s[m-1]+s[m])/2; };
const mode = arr => { const f={}; arr.forEach(x=>f[x]=(f[x]||0)+1); const mx=Math.max(...Object.values(f)); const m=Object.keys(f).filter(k=>f[k]===mx).map(Number); return m.length===arr.length?[]:m; };
const stdDev = (arr,pop=false) => { const m=mean(arr); return Math.sqrt(sum(arr.map(x=>(x-m)**2))/(arr.length-(pop?0:1))); };
const quartiles = arr => { const s=[...arr].sort((a,b)=>a-b),q=p=>{const i=p*(s.length-1),lo=Math.floor(i),hi=Math.ceil(i);return s[lo]+(s[hi]-s[lo])*(i-lo);}; return {q1:q(.25),q2:q(.5),q3:q(.75),min:s[0],max:s[s.length-1]}; };

// Normal distribution
const normalPDF = (x,mu=0,sigma=1) => Math.exp(-0.5*((x-mu)/sigma)**2)/(sigma*Math.sqrt(2*Math.PI));
const normalCDF = z => { const p=0.2316419,b=[0.31938153,-0.356563782,1.781477937,-1.821255978,1.330274429],t=1/(1+p*Math.abs(z)),poly=b.reduce((a,bi,i)=>a+bi*t**(i+1),0),c=1-normalPDF(z)*poly; return z<0?1-c:c; };
const normalInv = p => { const a=[-39.6968,220.946,-275.929,138.358,-30.6647,2.50663],b=[-54.4761,161.586,-155.699,66.8013,-13.2806],c=[-7.784e-3,-3.224e-1,-2.401,2.5497e0,4.3747,2.9382],d=[7.784e-3,3.225e-1,2.445,3.754],lo=0.02425,hi=1-lo; let q,r; if(p<lo){q=Math.sqrt(-2*Math.log(p));return(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);} if(p<=hi){q=p-0.5;r=q*q;return(((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q/((((( b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1);} q=Math.sqrt(-2*Math.log(1-p));return-(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1); };

// Binomial
const logFact = n => { let s=0; for(let i=2;i<=n;i++) s+=Math.log(i); return s; };
const binomPMF = (k,n,p) => { if(k<0||k>n) return 0; return Math.exp(logFact(n)-logFact(k)-logFact(n-k)+k*Math.log(p)+(n-k)*Math.log(1-p)); };
const binomCDF = (k,n,p) => { let s=0; for(let i=0;i<=Math.floor(k);i++) s+=binomPMF(i,n,p); return Math.min(1,s); };
const binomInv = (prob,n,p) => { let s=0; for(let k=0;k<=n;k++){s+=binomPMF(k,n,p);if(s>=prob) return k;} return n; };

// t-distribution
const tPDF = (t,df) => { const c=Math.exp(logGamma((df+1)/2)-logGamma(df/2))/Math.sqrt(df*Math.PI); return c*(1+t*t/df)**(-(df+1)/2); };
const logGamma = x => { const c=[76.18009172947146,-86.50532032941677,24.01409824083091,-1.231739572450155,0.1208650973866179e-2,-0.5395239384953e-5]; let y=x,tmp=x+5.5; tmp-=(x+0.5)*Math.log(tmp); let ser=1.000000000190015; for(let i=0;i<6;i++){y++;ser+=c[i]/y;} return -tmp+Math.log(2.5066282746310005*ser/x); };
const tCDF = (t,df) => { const x=df/(df+t*t),ibeta=incompleteBeta(df/2,0.5,x); return t>=0?1-ibeta/2:ibeta/2; };
const incompleteBeta = (a,b,x) => { if(x<=0) return 0; if(x>=1) return 1; const lbeta=logGamma(a)+logGamma(b)-logGamma(a+b); let bt=Math.exp(a*Math.log(x)+b*Math.log(1-x)-lbeta); if(x<(a+1)/(a+b+2)) return bt*betaCF(a,b,x)/a; return 1-bt*betaCF(b,a,1-x)/b; };
const betaCF = (a,b,x) => { const MAXIT=200,EPS=3e-7; let qab=a+b,qap=a+1,qam=a-1,c=1,d=1-qab*x/qap; if(Math.abs(d)<1e-30) d=1e-30; d=1/d; let h=d; for(let m=1;m<=MAXIT;m++){const m2=2*m,aa=m*(b-m)*x/((qam+m2)*(a+m2));d=1+aa*d;if(Math.abs(d)<1e-30)d=1e-30;c=1+aa/c;if(Math.abs(c)<1e-30)c=1e-30;d=1/d;h*=d*c;const bb=-(a+m)*(qab+m)*x/((a+m2)*(qap+m2));d=1+bb*d;if(Math.abs(d)<1e-30)d=1e-30;c=1+bb/c;if(Math.abs(c)<1e-30)c=1e-30;d=1/d;const del=d*c;h*=del;if(Math.abs(del-1)<EPS) break;} return h; };
const tInv = (p,df) => { let lo=-20,hi=20; for(let i=0;i<60;i++){const mid=(lo+hi)/2;tCDF(mid,df)<p?lo=mid:hi=mid;} return (lo+hi)/2; };
const tCritical = (df,alpha) => { const z=normalInv(1-alpha/2); return z+(z**3+z)/(4*df)+(5*z**5+16*z**3+3*z)/(96*df**2); };

// Chi-square
const chiSqCDF = (x,df) => { if(x<=0) return 0; const z=((x/df)**(1/3)-(1-2/(9*df)))/Math.sqrt(2/(9*df)); return normalCDF(z); };
const chiSqPDF = (x,df) => { if(x<=0) return 0; return Math.exp((df/2-1)*Math.log(x)-x/2-(df/2)*Math.log(2)-logGamma(df/2)); };
const chiSqInv = (p,df) => { let lo=1e-6,hi=df*5+200; for(let i=0;i<80;i++){const mid=(lo+hi)/2;chiSqCDF(mid,df)<p?lo=mid:hi=mid;} return (lo+hi)/2; };
const chiSqCritical = (df,alpha) => chiSqInv(1-alpha,df);

// Other stats
const welchDF = (s1,n1,s2,n2) => { const a=s1*s1/n1,b=s2*s2/n2; return (a+b)**2/(a**2/(n1-1)+b**2/(n2-1)); };
const pValueTail = (stat,tail) => { if(tail==="two") return 2*(1-normalCDF(Math.abs(stat))); if(tail==="right") return 1-normalCDF(stat); return normalCDF(stat); };
const linReg = (xs,ys) => { const n=xs.length,mx=mean(xs),my=mean(ys),num=xs.reduce((s,x,i)=>s+(x-mx)*(ys[i]-my),0),den=xs.reduce((s,x)=>s+(x-mx)**2,0),b=num/den,a=my-b*mx,r=num/Math.sqrt(den*ys.reduce((s,y)=>s+(y-my)**2,0)); return {slope:b,intercept:a,r,r2:r**2}; };

// ═══════════════════════════════════════════════════════════════════════════════
// SHARED UI COMPONENTS
// ═══════════════════════════════════════════════════════════════════════════════
const TABS = ["📊 Descriptive","🔬 Hypothesis","📏 Confidence","🔔 Distributions","📈 Regression","🎨 Visualizations","📂 CSV Upload"];

function Card({title,children,accent="#2563eb"}) {
  return <div style={{background:"#fff",borderRadius:10,padding:20,marginBottom:16,boxShadow:"0 1px 8px rgba(0,0,0,0.07)",borderLeft:`4px solid ${accent}`}}>{title&&<div style={{fontWeight:700,fontSize:15,marginBottom:12,color:"#1e3a8a"}}>{title}</div>}{children}</div>;
}
function DataInput({label,value,onChange,placeholder="e.g. 78, 85, 92, 61, 74",rows=2}) {
  return <div style={{marginBottom:12}}><label style={{display:"block",fontSize:12,fontWeight:600,color:"#475569",marginBottom:4}}>{label}</label><textarea value={value} onChange={e=>onChange(e.target.value)} placeholder={placeholder} rows={rows} style={{width:"100%",padding:"9px 12px",border:"1px solid #cbd5e1",borderRadius:7,fontSize:13,fontFamily:"monospace",resize:"vertical",boxSizing:"border-box",outline:"none"}}/></div>;
}
function NumInput({label,value,onChange,step=1,min,max,width=130}) {
  return <div style={{marginBottom:10}}><label style={{display:"block",fontSize:12,fontWeight:600,color:"#475569",marginBottom:3}}>{label}</label><input type="number" value={value} onChange={e=>onChange(e.target.value)} step={step} min={min} max={max} style={{padding:"8px 11px",border:"1px solid #cbd5e1",borderRadius:7,fontSize:13,width,outline:"none"}}/></div>;
}
function Sel({label,value,onChange,options,width="100%"}) {
  return <div style={{marginBottom:10}}><label style={{display:"block",fontSize:12,fontWeight:600,color:"#475569",marginBottom:3}}>{label}</label><select value={value} onChange={e=>onChange(e.target.value)} style={{padding:"8px 11px",border:"1px solid #cbd5e1",borderRadius:7,fontSize:13,width,outline:"none"}}>{options.map(([v,l])=><option key={v} value={v}>{l}</option>)}</select></div>;
}
function Btn({onClick,children,color="#2563eb"}) {
  return <button onClick={onClick} style={{background:color,color:"#fff",border:"none",borderRadius:7,padding:"10px 22px",fontSize:13,fontWeight:700,cursor:"pointer",marginTop:4,marginRight:8}}>{children}</button>;
}
function StatRow({label,value,highlight}) {
  return <div style={{display:"flex",justifyContent:"space-between",padding:"8px 10px",background:highlight?"#eff6ff":"transparent",borderRadius:5,marginBottom:2}}><span style={{color:"#475569",fontSize:13}}>{label}</span><span style={{fontWeight:700,color:"#1e3a8a",fontSize:13,fontFamily:"monospace"}}>{value}</span></div>;
}
function Note({children,color="#fef9c3",border="#fde68a",text="#713f12"}) {
  return <div style={{background:color,border:`1px solid ${border}`,borderRadius:7,padding:"9px 13px",fontSize:12,color:text,marginTop:10,lineHeight:1.5}}>{children}</div>;
}
function ErrMsg({msg}) { return msg?<div style={{background:"#fef2f2",border:"1px solid #fecaca",borderRadius:7,padding:"9px 13px",fontSize:12,color:"#b91c1c",marginTop:8}}>⚠️ {msg}</div>:null; }
function ResultBanner({reject,alpha}) {
  return <div style={{textAlign:"center",background:reject?"#fef2f2":"#f0fdf4",borderRadius:9,padding:"13px 20px",marginBottom:14}}><div style={{fontSize:21,fontWeight:800,color:reject?"#dc2626":"#065f46"}}>{reject?"❌ Reject H₀":"✅ Fail to Reject H₀"}</div><div style={{fontSize:12,color:"#64748b",marginTop:3}}>at α = {alpha} significance level</div></div>;
}
function DecisionNote({pVal,alpha}) {
  const reject=pVal<alpha;
  return <Note>💡 P-value ({pVal.toFixed(4)}) {reject?"<":"≥"} α ({alpha}) → {reject?"Reject H₀. Sufficient evidence supports the alternative hypothesis.":"Fail to Reject H₀. Insufficient evidence to reject the null hypothesis."}</Note>;
}
function TailSel({value,onChange,param="μ",param0="μ₀",testMode=false}) {
  const opts = testMode
    ? [["two","Two-tailed"],["right","Right-tailed"],["left","Left-tailed"]]
    : [["two",`${param} ≠ ${param0} (two-tailed)`],["right",`${param} > ${param0} (right-tailed)`],["left",`${param} < ${param0} (left-tailed)`]];
  const lbl = testMode ? "Tail Type" : "Alternative Hypothesis (H₁)";
  return <Sel label={lbl} value={value} onChange={onChange} options={opts}/>;
}
function ZTResults({res,label,testMode=false}) {
  return <div style={{marginTop:14}}>{!testMode&&<ResultBanner reject={res.reject} alpha={res.alpha}/>}<StatRow label={`${label.toUpperCase()} Test Statistic`} value={res.stat.toFixed(4)} highlight/>{res.df!=null&&<StatRow label="Degrees of Freedom (df)" value={typeof res.df==="number"?res.df.toFixed(2):res.df}/>}<StatRow label={`Critical Value (${label}*${res.tail==="two"?" ±":""})`} value={res.critical.toFixed(4)}/><StatRow label="P-value" value={res.pVal.toFixed(4)}/>{!testMode&&<DecisionNote pVal={res.pVal} alpha={res.alpha}/>}</div>;
}

// ═══════════════════════════════════════════════════════════════════════════════
// SVG HELPER
// ═══════════════════════════════════════════════════════════════════════════════
const SVG_W=500, SVG_H=220;
const pad={l:50,r:20,t:18,b:38};
const chartW=SVG_W-pad.l-pad.r, chartH=SVG_H-pad.t-pad.b;
const mkScaleX=(mn,mx)=>x=>pad.l+(x-mn)/(mx-mn||1)*chartW;
const mkScaleY=(mn,mx)=>y=>SVG_H-pad.b-(y-mn)/(mx-mn||1)*chartH;
// ═══════════════════════════════════════════════════════════════════════════════
// CHART DOWNLOAD UTILITY
// ═══════════════════════════════════════════════════════════════════════════════
function downloadSVG(svgEl, filename) {
  if (!svgEl) return;
  const svgData = new XMLSerializer().serializeToString(svgEl);
  const canvas = document.createElement("canvas");
  const scale = 3; // 3x resolution for crisp export
  canvas.width  = (svgEl.viewBox.baseVal.width  || svgEl.clientWidth  || 500) * scale;
  canvas.height = (svgEl.viewBox.baseVal.height || svgEl.clientHeight || 220) * scale;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  const img = new Image();
  const blob = new Blob([svgData], {type:"image/svg+xml;charset=utf-8"});
  const url  = URL.createObjectURL(blob);
  img.onload = () => {
    ctx.scale(scale, scale);
    ctx.drawImage(img, 0, 0);
    URL.revokeObjectURL(url);
    const a = document.createElement("a");
    a.download = filename + ".png";
    a.href = canvas.toDataURL("image/png");
    a.click();
  };
  img.src = url;
}

function ChartToolbar({svgRef, filename, children}) {
  const [copied, setCopied] = useState(false);
  const handleDownload = () => downloadSVG(svgRef.current, filename);
  const handleCopy = async () => {
    const svgEl = svgRef.current;
    if (!svgEl) return;
    const svgData = new XMLSerializer().serializeToString(svgEl);
    const canvas = document.createElement("canvas");
    const scale = 3;
    canvas.width  = (svgEl.viewBox.baseVal.width  || svgEl.clientWidth  || 500) * scale;
    canvas.height = (svgEl.viewBox.baseVal.height || svgEl.clientHeight || 220) * scale;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    const img = new Image();
    const blob = new Blob([svgData], {type:"image/svg+xml;charset=utf-8"});
    const url  = URL.createObjectURL(blob);
    img.onload = async () => {
      ctx.scale(scale, scale);
      ctx.drawImage(img, 0, 0);
      URL.revokeObjectURL(url);
      canvas.toBlob(async (pngBlob) => {
        try {
          await navigator.clipboard.write([new ClipboardItem({"image/png": pngBlob})]);
          setCopied(true);
          setTimeout(() => setCopied(false), 2000);
        } catch(e) {
          // Fallback: just download instead
          handleDownload();
        }
      });
    };
    img.src = url;
  };
  return (
    <div>
      {children}
      <div style={{display:"flex",gap:8,marginTop:8,justifyContent:"flex-end"}}>
        <button onClick={handleCopy} style={{background:copied?"#059669":"#f1f5f9",color:copied?"#fff":"#475569",border:"1px solid #e2e8f0",borderRadius:6,padding:"5px 12px",fontSize:11,cursor:"pointer",fontWeight:600,transition:"all .2s"}}>
          {copied ? "✅ Copied!" : "📋 Copy Image"}
        </button>
        <button onClick={handleDownload} style={{background:"#2563eb",color:"#fff",border:"none",borderRadius:6,padding:"5px 12px",fontSize:11,cursor:"pointer",fontWeight:600}}>
          ⬇ Save as PNG
        </button>
      </div>
    </div>
  );
}

function Axis({xMin,xMax,yMin=0,yMax,ticks=5,yLabel="",xLabel=""}) {
  const sx=mkScaleX(xMin,xMax), sy=mkScaleY(yMin,yMax);
  const xTks=Array.from({length:ticks+1},(_,i)=>xMin+(xMax-xMin)*i/ticks);
  const yTks=Array.from({length:ticks+1},(_,i)=>yMin+(yMax-yMin)*i/ticks);
  return <>
    <line x1={pad.l} y1={SVG_H-pad.b} x2={SVG_W-pad.r} y2={SVG_H-pad.b} stroke="#94a3b8" strokeWidth={1}/>
    <line x1={pad.l} y1={pad.t} x2={pad.l} y2={SVG_H-pad.b} stroke="#94a3b8" strokeWidth={1}/>
    {xTks.map((v,i)=><g key={i}><line x1={sx(v)} y1={SVG_H-pad.b} x2={sx(v)} y2={SVG_H-pad.b+4} stroke="#94a3b8"/><text x={sx(v)} y={SVG_H-pad.b+15} fontSize={9} fill="#64748b" textAnchor="middle">{v%1===0?v:v.toFixed(1)}</text></g>)}
    {yTks.map((v,i)=><g key={i}><line x1={pad.l-4} y1={sy(v)} x2={pad.l} y2={sy(v)} stroke="#94a3b8"/><text x={pad.l-6} y={sy(v)+3} fontSize={9} fill="#64748b" textAnchor="end">{v%1===0?v:v.toFixed(1)}</text></g>)}
    {yLabel&&<text x={12} y={SVG_H/2} fontSize={10} fill="#64748b" textAnchor="middle" transform={`rotate(-90,12,${SVG_H/2})`}>{yLabel}</text>}
    {xLabel&&<text x={SVG_W/2} y={SVG_H-2} fontSize={10} fill="#64748b" textAnchor="middle">{xLabel}</text>}
  </>;
}

// ═══════════════════════════════════════════════════════════════════════════════
// APP ROOT
// ═══════════════════════════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════════════════════════
// LANDING SCREEN
// ═══════════════════════════════════════════════════════════════════════════════
function LandingScreen({onSelect}) {
  return (
    <div style={{fontFamily:"Georgia,serif",background:"linear-gradient(160deg,#0f172a 0%,#1e3a8a 60%,#2563eb 100%)",minHeight:"100vh",display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"center",padding:"40px 24px",color:"#fff"}}>
      <div style={{fontSize:48,marginBottom:12}}>📐</div>
      <div style={{fontSize:28,fontWeight:800,letterSpacing:.5,marginBottom:6,textAlign:"center"}}>Elementary Statistics Calculator</div>
      <div style={{fontSize:14,opacity:.7,marginBottom:48,textAlign:"center"}}>Choose a mode to get started</div>
      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:24,maxWidth:720,width:"100%"}}>

        {/* Practice Mode */}
        <div onClick={()=>onSelect("practice")} style={{background:"rgba(255,255,255,0.08)",border:"2px solid rgba(255,255,255,0.2)",borderRadius:16,padding:"32px 28px",cursor:"pointer",transition:"all .2s",backdropFilter:"blur(8px)"}}
          onMouseEnter={e=>{e.currentTarget.style.background="rgba(255,255,255,0.16)";e.currentTarget.style.borderColor="rgba(255,255,255,0.5)";}}
          onMouseLeave={e=>{e.currentTarget.style.background="rgba(255,255,255,0.08)";e.currentTarget.style.borderColor="rgba(255,255,255,0.2)";}}>
          <div style={{fontSize:36,marginBottom:12}}>🎓</div>
          <div style={{fontSize:20,fontWeight:700,marginBottom:8}}>Practice Calculator</div>
          <div style={{fontSize:13,opacity:.8,lineHeight:1.6,marginBottom:16}}>Full-featured calculator with hints, interpretations, decision rules, and explanatory notes. Perfect for learning and homework.</div>
          <div style={{display:"flex",flexWrap:"wrap",gap:6}}>
            {["Outlier fences","Empirical rule","Decision notes","r² interpretation","Chart descriptions"].map(f=>(
              <span key={f} style={{background:"rgba(16,185,129,0.25)",border:"1px solid rgba(16,185,129,0.4)",borderRadius:4,padding:"2px 8px",fontSize:11,color:"#a7f3d0"}}>{f}</span>
            ))}
          </div>
        </div>

        {/* Testing Mode */}
        <div onClick={()=>onSelect("test")} style={{background:"rgba(220,38,38,0.12)",border:"2px solid rgba(220,38,38,0.3)",borderRadius:16,padding:"32px 28px",cursor:"pointer",transition:"all .2s",backdropFilter:"blur(8px)"}}
          onMouseEnter={e=>{e.currentTarget.style.background="rgba(220,38,38,0.22)";e.currentTarget.style.borderColor="rgba(220,38,38,0.6)";}}
          onMouseLeave={e=>{e.currentTarget.style.background="rgba(220,38,38,0.12)";e.currentTarget.style.borderColor="rgba(220,38,38,0.3)";}}>
          <div style={{fontSize:36,marginBottom:12}}>📝</div>
          <div style={{fontSize:20,fontWeight:700,marginBottom:8}}>Testing Calculator</div>
          <div style={{fontSize:13,opacity:.8,lineHeight:1.6,marginBottom:16}}>Calculations only — no hints, no interpretations, no decision rules. Students must supply their own conclusions.</div>
          <div style={{display:"flex",flexWrap:"wrap",gap:6}}>
            {["No outlier fences","No empirical rule","No decision notes","No r² interpretation","No chart descriptions"].map(f=>(
              <span key={f} style={{background:"rgba(220,38,38,0.2)",border:"1px solid rgba(220,38,38,0.4)",borderRadius:4,padding:"2px 8px",fontSize:11,color:"#fca5a5"}}>{f}</span>
            ))}
          </div>
        </div>
      </div>

      <div style={{marginTop:40,fontSize:12,opacity:.45,textAlign:"center"}}>
        You can switch modes at any time using the button in the top bar.
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// APP ROOT
// ═══════════════════════════════════════════════════════════════════════════════
export default function StatsApp() {
  const [screen,setScreen]=useState("landing"); // "landing" | "practice" | "test"
  const [tab,setTab]=useState(0);
  const [csvData,setCsvData]=useState(null);
  const [csvData2,setCsvData2]=useState(null);
  const [sharedRaw,setSharedRaw]=useState("");
  const [sharedRaw2,setSharedRaw2]=useState("");
  const [sharedXRaw,setSharedXRaw]=useState("");
  const [sharedYRaw,setSharedYRaw]=useState("");

  const handleDataLoaded2 = (parsed) => { setCsvData2(parsed); };
  const handleDataLoaded = (parsed, target, col, col2) => {
    setCsvData(parsed);
    if (!target || !col) return;
    const vals = parsed.numericCols[col] || [];
    const vals2 = col2 && parsed.numericCols[col2] ? parsed.numericCols[col2] : [];
    const str = vals.join(", ");
    const str2 = vals2.join(", ");
    if (target === "descriptive" || target === "visualizations" || target === "hypothesis_t2") {
      setSharedRaw(str); setSharedRaw2(str2);
    }
    if (target === "regression") { setSharedXRaw(str); setSharedYRaw(str2); }
    const tabMap = {descriptive:0, hypothesis_t2:1, regression:4, visualizations:5};
    if (tabMap[target] !== undefined) setTab(tabMap[target]);
  };

  if (screen==="landing") return <LandingScreen onSelect={s=>setScreen(s)}/>;

  const testMode = screen==="test";
  const headerBg = testMode
    ? "linear-gradient(135deg,#7f1d1d 0%,#dc2626 100%)"
    : "linear-gradient(135deg,#1e3a8a 0%,#2563eb 100%)";

  return (
    <div style={{fontFamily:"Georgia,serif",background:testMode?"#fff5f5":"#f0f4ff",minHeight:"100vh",color:"#1e293b"}}>
      <div style={{background:headerBg,color:"#fff",padding:"18px 24px 0"}}>
        <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",marginBottom:4}}>
          <div style={{fontSize:21,fontWeight:700,letterSpacing:.3}}>
            {testMode?"📝 Statistics Testing Calculator":"📐 Elementary Statistics Calculator"}
          </div>
          <div style={{display:"flex",gap:8,alignItems:"center"}}>
            {testMode&&<span style={{background:"rgba(255,255,255,0.2)",borderRadius:5,padding:"3px 10px",fontSize:11,fontWeight:700,letterSpacing:.5}}>TESTING MODE</span>}
            <button onClick={()=>setScreen("landing")} style={{background:"rgba(255,255,255,0.15)",color:"#fff",border:"1px solid rgba(255,255,255,0.3)",borderRadius:6,padding:"4px 12px",fontSize:11,cursor:"pointer"}}>⇄ Switch Mode</button>
          </div>
        </div>
        <div style={{display:"flex",alignItems:"center",gap:10,marginBottom:12}}>
          <div style={{fontSize:11,opacity:.7}}>{testMode?"Calculations only — no hints or interpretations":"Full-featured calculator with hints and interpretations"}</div>
          {csvData&&<div style={{fontSize:11,background:"rgba(255,255,255,0.2)",borderRadius:5,padding:"2px 8px"}}>📂 A: {csvData.rowCount} rows{csvData2?` | B: ${csvData2.rowCount} rows`:""}</div>}
        </div>
        <div style={{display:"flex",gap:3,flexWrap:"wrap"}}>
          {TABS.map((t,i)=><button key={i} onClick={()=>setTab(i)} style={{background:tab===i?"#fff":"rgba(255,255,255,0.15)",color:tab===i?testMode?"#7f1d1d":"#1e3a8a":"#fff",border:"none",borderRadius:"6px 6px 0 0",padding:"8px 13px",cursor:"pointer",fontSize:11,fontWeight:tab===i?700:400,transition:"all .15s"}}>{t}{i===6&&(csvData||csvData2)?<span style={{marginLeft:4,background:"#10b981",borderRadius:3,padding:"0 4px",fontSize:9}}>{csvData&&csvData2?"2":"1"}✓</span>:""}</button>)}
        </div>
      </div>
      <div style={{padding:"22px 24px",maxWidth:860,margin:"0 auto"}}>
        {tab===0&&<DescriptiveTab initData={sharedRaw} testMode={testMode}/>}
        {tab===1&&<HypothesisTab initRaw1={sharedRaw} initRaw2={sharedRaw2} testMode={testMode}/>}
        {tab===2&&<ConfidenceTab testMode={testMode}/>}
        {tab===3&&<DistributionsTab testMode={testMode}/>}
        {tab===4&&<RegressionTab initXRaw={sharedXRaw} initYRaw={sharedYRaw} testMode={testMode}/>}
        {tab===5&&<VisualizationsTab initData={sharedRaw} initData2={sharedRaw2} testMode={testMode}/>}
        {tab===6&&<CSVUploadTab onDataLoaded={handleDataLoaded} csvData={csvData} csvData2={csvData2} onDataLoaded2={handleDataLoaded2}/>}
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// TAB 1: DESCRIPTIVE STATISTICS
// ═══════════════════════════════════════════════════════════════════════════════
function DescriptiveTab({initData="",testMode=false}) {
  const [raw,setRaw]=useState(initData||"78, 85, 92, 61, 74, 88, 95, 71, 83, 79");
  const [res,setRes]=useState(null);
  // Sync when CSV data arrives
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useMemo(()=>{if(initData) setRaw(initData);},[initData]);
  const calc=()=>{const d=parseData(raw);if(d.length<2)return;const {q1,q2,q3,min,max}=quartiles(d);setRes({n:d.length,mean:mean(d),median:median(d),mode:mode(d),s:stdDev(d),sigma:stdDev(d,true),varS:stdDev(d)**2,range:max-min,min,max,q1,q2,q3,iqr:q3-q1,data:d});};
  return <div>
    <Card title="Enter Your Data" accent="#2563eb">
      <DataInput label="Data values (separated by commas or spaces):" value={raw} onChange={setRaw}/>
      <Btn onClick={calc}>Calculate Statistics</Btn>
    </Card>
    {res&&<><div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:14}}>
      <Card title="Central Tendency" accent="#059669"><StatRow label="Count (n)" value={res.n}/><StatRow label="Mean (x̄)" value={res.mean.toFixed(4)} highlight/><StatRow label="Median" value={res.median.toFixed(4)}/><StatRow label="Mode" value={res.mode.length?res.mode.join(", "):"No mode"}/></Card>
      <Card title="Spread" accent="#7c3aed"><StatRow label="Std Dev — sample (s)" value={res.s.toFixed(4)} highlight/><StatRow label="Std Dev — pop. (σ)" value={res.sigma.toFixed(4)}/><StatRow label="Variance — sample (s²)" value={res.varS.toFixed(4)}/><StatRow label="Range" value={res.range.toFixed(4)}/></Card>
    </div>
    <Card title="Five-Number Summary" accent="#d97706">
      <div style={{display:"grid",gridTemplateColumns:"repeat(5,1fr)",gap:7,textAlign:"center",marginBottom:10}}>
        {[["Min",res.min],["Q1",res.q1],["Median",res.q2],["Q3",res.q3],["Max",res.max]].map(([l,v])=><div key={l} style={{background:"#eff6ff",borderRadius:7,padding:"9px 3px"}}><div style={{fontSize:9,color:"#64748b",marginBottom:2}}>{l}</div><div style={{fontWeight:700,color:"#1e3a8a",fontSize:13}}>{v.toFixed(2)}</div></div>)}
      </div>
      {!testMode&&<StatRow label="IQR (Q3 − Q1)" value={res.iqr.toFixed(4)} highlight/>}
      {!testMode&&<Note>💡 Outlier fences — Lower: {(res.q1-1.5*res.iqr).toFixed(2)} | Upper: {(res.q3+1.5*res.iqr).toFixed(2)}</Note>}
    </Card></>}
  </div>;
}

// ═══════════════════════════════════════════════════════════════════════════════
// TAB 2: HYPOTHESIS TESTING
// ═══════════════════════════════════════════════════════════════════════════════
const HYP_TESTS=[["z1","1-Sample Z"],["z2","2-Sample Z"],["t1","1-Sample T"],["t2","2-Sample T"],["chi","Chi-Square"]];
function HypothesisTab({initRaw1="",initRaw2="",testMode=false}) {
  const [test,setTest]=useState("z1");
  return <div>
    <Card title="Select Test Type" accent="#dc2626">
      <div style={{display:"flex",flexWrap:"wrap",gap:7}}>
        {HYP_TESTS.map(([v,l])=><button key={v} onClick={()=>setTest(v)} style={{padding:"8px 14px",borderRadius:7,border:"2px solid",borderColor:test===v?"#dc2626":"#cbd5e1",background:test===v?"#fef2f2":"#fff",color:test===v?"#dc2626":"#475569",fontWeight:test===v?700:400,fontSize:12,cursor:"pointer"}}>{l}</button>)}
      </div>
      {(initRaw1||initRaw2)&&<Note color="#eff6ff" border="#bfdbfe" text="#1e40af" style={{marginTop:10}}>📂 CSV data loaded — switch to <strong>Two-Sample T-Test</strong> to use it in Group 1 / Group 2 raw data fields.</Note>}
    </Card>
    {test==="z1"&&<OneSampleZTest testMode={testMode}/>}{test==="z2"&&<TwoSampleZTest testMode={testMode}/>}{test==="t1"&&<OneSampleTTest testMode={testMode}/>}{test==="t2"&&<TwoSampleTTest initRaw1={initRaw1} initRaw2={initRaw2} testMode={testMode}/>}{test==="chi"&&<ChiSquareTest testMode={testMode}/>}
  </div>;
}
function OneSampleZTest({testMode=false}) {
  const [xbar,setXbar]=useState("52"),[mu0,setMu0]=useState("50"),[sigma,setSigma]=useState("8"),[n,setN]=useState("36"),[tail,setTail]=useState("two"),[alpha,setAlpha]=useState("0.05"),[res,setRes]=useState(null),[err,setErr]=useState("");
  const calc=()=>{const m=+xbar,m0=+mu0,s=+sigma,nv=+n,a=+alpha;if(s<=0||nv<1){setErr("σ>0 and n≥1 required.");return;}setErr("");const stat=(m-m0)/(s/Math.sqrt(nv)),critical=normalInv(tail==="two"?1-a/2:1-a),pVal=pValueTail(stat,tail),reject=tail==="two"?Math.abs(stat)>critical:tail==="right"?stat>critical:stat<-critical;setRes({stat,pVal,critical,reject,alpha:a,tail,df:null});};
  return <Card title="One-Sample Z-Test" accent="#dc2626">{!testMode&&<p style={{fontSize:12,color:"#64748b",margin:"0 0 10px"}}>H₀: μ = μ₀ | Use when population σ is known</p>}<div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:9}}><NumInput label="Sample Mean (x̄)" value={xbar} onChange={setXbar} step={.1}/><NumInput label="Claimed Mean (μ₀)" value={mu0} onChange={setMu0} step={.1}/><NumInput label="Pop. Std Dev (σ)" value={sigma} onChange={setSigma} step={.1}/><NumInput label="Sample Size (n)" value={n} onChange={setN} min={1}/></div><div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:10}}><TailSel value={tail} onChange={setTail} testMode={testMode}/><Sel label="Significance (α)" value={alpha} onChange={setAlpha} options={[["0.10","0.10"],["0.05","0.05"],["0.02","0.02"],["0.01","0.01"]]}/></div><Btn onClick={calc}>Run Test</Btn><ErrMsg msg={err}/>{res&&<ZTResults res={res} label="z" testMode={testMode}/>}</Card>;
}
function TwoSampleZTest({testMode=false}) {
  const [x1,setX1]=useState("85"),[x2,setX2]=useState("80"),[s1,setS1]=useState("10"),[s2,setS2]=useState("12"),[n1,setN1]=useState("40"),[n2,setN2]=useState("45"),[tail,setTail]=useState("two"),[alpha,setAlpha]=useState("0.05"),[res,setRes]=useState(null),[err,setErr]=useState("");
  const calc=()=>{const m1=+x1,m2=+x2,sv1=+s1,sv2=+s2,nv1=+n1,nv2=+n2,a=+alpha;if(sv1<=0||sv2<=0||nv1<1||nv2<1){setErr("σ>0 and n≥1 required.");return;}setErr("");const se=Math.sqrt(sv1**2/nv1+sv2**2/nv2),stat=(m1-m2)/se,critical=normalInv(tail==="two"?1-a/2:1-a),pVal=pValueTail(stat,tail),reject=tail==="two"?Math.abs(stat)>critical:tail==="right"?stat>critical:stat<-critical;setRes({stat,pVal,critical,reject,alpha:a,tail,df:null,se,diff:m1-m2});};
  return <Card title="Two-Sample Z-Test" accent="#dc2626">{!testMode&&<p style={{fontSize:12,color:"#64748b",margin:"0 0 10px"}}>H₀: μ₁ = μ₂ | Both population σ's known</p>}<div style={{display:"grid",gridTemplateColumns:"repeat(3,1fr)",gap:9}}><NumInput label="x̄₁" value={x1} onChange={setX1} step={.1}/><NumInput label="σ₁" value={s1} onChange={setS1} step={.1}/><NumInput label="n₁" value={n1} onChange={setN1} min={1}/><NumInput label="x̄₂" value={x2} onChange={setX2} step={.1}/><NumInput label="σ₂" value={s2} onChange={setS2} step={.1}/><NumInput label="n₂" value={n2} onChange={setN2} min={1}/></div><div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:10}}><TailSel value={tail} onChange={setTail} param="μ₁−μ₂" param0="0" testMode={testMode}/><Sel label="Significance (α)" value={alpha} onChange={setAlpha} options={[["0.10","0.10"],["0.05","0.05"],["0.02","0.02"],["0.01","0.01"]]}/></div><Btn onClick={calc}>Run Test</Btn><ErrMsg msg={err}/>{res&&<div style={{marginTop:12}}>{!testMode&&{!testMode&&<ResultBanner reject={res.reject} alpha={res.alpha}/>}}<StatRow label="Difference (x̄₁−x̄₂)" value={res.diff.toFixed(4)}/><StatRow label="Standard Error" value={res.se.toFixed(4)}/><StatRow label="Z Statistic" value={res.stat.toFixed(4)} highlight/><StatRow label={`Critical Value (z*${res.tail==="two"?" ±":""})`} value={res.critical.toFixed(4)}/><StatRow label="P-value" value={res.pVal.toFixed(4)}/>{!testMode&&{!testMode&&<DecisionNote pVal={res.pVal} alpha={res.alpha}/>}}</div>}</Card>;
}
function OneSampleTTest({testMode=false}) {
  const [xbar,setXbar]=useState("52"),[mu0,setMu0]=useState("50"),[s,setS]=useState("8"),[n,setN]=useState("16"),[tail,setTail]=useState("two"),[alpha,setAlpha]=useState("0.05"),[res,setRes]=useState(null),[err,setErr]=useState("");
  const calc=()=>{const m=+xbar,m0=+mu0,sv=+s,nv=+n,a=+alpha;if(sv<=0||nv<2){setErr("s>0 and n≥2 required.");return;}setErr("");const df=nv-1,stat=(m-m0)/(sv/Math.sqrt(nv)),critical=tCritical(df,tail==="two"?a:a*2),pVal=pValueTail(stat,tail),reject=tail==="two"?Math.abs(stat)>critical:tail==="right"?stat>critical:stat<-critical;setRes({stat,pVal,critical,reject,alpha:a,tail,df});};
  return <Card title="One-Sample T-Test" accent="#7c3aed">{!testMode&&<p style={{fontSize:12,color:"#64748b",margin:"0 0 10px"}}>H₀: μ = μ₀ | σ unknown; best for small samples</p>}<div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:9}}><NumInput label="Sample Mean (x̄)" value={xbar} onChange={setXbar} step={.1}/><NumInput label="Claimed Mean (μ₀)" value={mu0} onChange={setMu0} step={.1}/><NumInput label="Std Dev (s)" value={s} onChange={setS} step={.1}/><NumInput label="Sample Size (n)" value={n} onChange={setN} min={2}/></div><div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:10}}><TailSel value={tail} onChange={setTail} testMode={testMode}/><Sel label="Significance (α)" value={alpha} onChange={setAlpha} options={[["0.10","0.10"],["0.05","0.05"],["0.02","0.02"],["0.01","0.01"]]}/></div><Btn onClick={calc}>Run Test</Btn><ErrMsg msg={err}/>{res&&<ZTResults res={res} label="t" testMode={testMode}/>}</Card>;
}
function TwoSampleTTest({initRaw1="",initRaw2="",testMode=false}) {
  const [mode2,setMode2]=useState(initRaw1?"raw":"summary"),[x1,setX1]=useState("85"),[x2,setX2]=useState("80"),[s1,setS1]=useState("10"),[s2,setS2]=useState("12"),[n1,setN1]=useState("15"),[n2,setN2]=useState("18"),[raw1,setRaw1]=useState(initRaw1||"88,92,79,85,91,76,83,90,87,84,78,95,81,86,89"),[raw2,setRaw2]=useState(initRaw2||"75,82,78,70,85,80,74,88,77,83,71,86,79,73,81,84,76,80"),[tail,setTail]=useState("two"),[alpha,setAlpha]=useState("0.05"),[res,setRes]=useState(null),[err,setErr]=useState("");
  useMemo(()=>{if(initRaw1){setRaw1(initRaw1);setMode2("raw");}if(initRaw2)setRaw2(initRaw2);},[initRaw1,initRaw2]);
  const calc=()=>{let m1,m2,sv1,sv2,nv1,nv2;if(mode2==="raw"){const d1=parseData(raw1),d2=parseData(raw2);if(d1.length<2||d2.length<2){setErr("Each group needs ≥ 2 values.");return;}m1=mean(d1);m2=mean(d2);sv1=stdDev(d1);sv2=stdDev(d2);nv1=d1.length;nv2=d2.length;}else{m1=+x1;m2=+x2;sv1=+s1;sv2=+s2;nv1=+n1;nv2=+n2;if(sv1<=0||sv2<=0||nv1<2||nv2<2){setErr("s>0 and n≥2 required.");return;}}setErr("");const df=welchDF(sv1,nv1,sv2,nv2),se=Math.sqrt(sv1**2/nv1+sv2**2/nv2),stat=(m1-m2)/se,a=+alpha,critical=tCritical(Math.floor(df),tail==="two"?a:a*2),pVal=pValueTail(stat,tail),reject=tail==="two"?Math.abs(stat)>critical:tail==="right"?stat>critical:stat<-critical;setRes({stat,pVal,critical,reject,alpha:a,tail,df,se,diff:m1-m2,m1,m2,sv1,sv2,nv1,nv2,fromRaw:mode2==="raw"});};
  return <Card title="Two-Sample T-Test (Welch's)" accent="#7c3aed">{!testMode&&<p style={{fontSize:12,color:"#64748b",margin:"0 0 10px"}}>H₀: μ₁ = μ₂ | Independent groups, σ unknown</p>}<div style={{marginBottom:10}}><label style={{fontSize:12,fontWeight:600,color:"#475569",marginRight:8}}>Input:</label>{[["summary","Summary Stats"],["raw","Raw Data"]].map(([v,l])=><label key={v} style={{marginRight:14,cursor:"pointer",fontSize:12}}><input type="radio" name="t2mode" value={v} checked={mode2===v} onChange={()=>setMode2(v)} style={{marginRight:4}}/>{l}</label>)}</div>{mode2==="summary"?<div style={{display:"grid",gridTemplateColumns:"repeat(3,1fr)",gap:9}}><NumInput label="x̄₁" value={x1} onChange={setX1} step={.1}/><NumInput label="s₁" value={s1} onChange={setS1} step={.1}/><NumInput label="n₁" value={n1} onChange={setN1} min={2}/><NumInput label="x̄₂" value={x2} onChange={setX2} step={.1}/><NumInput label="s₂" value={s2} onChange={setS2} step={.1}/><NumInput label="n₂" value={n2} onChange={setN2} min={2}/></div>:<><DataInput label="Group 1 data:" value={raw1} onChange={setRaw1}/><DataInput label="Group 2 data:" value={raw2} onChange={setRaw2}/></>}<div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:10}}><TailSel value={tail} onChange={setTail} param="μ₁−μ₂" param0="0" testMode={testMode}/><Sel label="Significance (α)" value={alpha} onChange={setAlpha} options={[["0.10","0.10"],["0.05","0.05"],["0.02","0.02"],["0.01","0.01"]]}/></div><Btn onClick={calc}>Run Test</Btn><ErrMsg msg={err}/>{res&&<div style={{marginTop:12}}>{!testMode&&{!testMode&&<ResultBanner reject={res.reject} alpha={res.alpha}/>}}{res.fromRaw&&<><StatRow label="Group 1 x̄₁" value={`${res.m1.toFixed(3)} (s=${res.sv1.toFixed(3)}, n=${res.nv1})`}/><StatRow label="Group 2 x̄₂" value={`${res.m2.toFixed(3)} (s=${res.sv2.toFixed(3)}, n=${res.nv2})`}/></>}<StatRow label="Difference (x̄₁−x̄₂)" value={res.diff.toFixed(4)}/><StatRow label="Standard Error" value={res.se.toFixed(4)}/><StatRow label="df (Welch–Satterthwaite)" value={res.df.toFixed(2)}/><StatRow label="T Statistic" value={res.stat.toFixed(4)} highlight/><StatRow label={`Critical Value (t*${res.tail==="two"?" ±":""})`} value={res.critical.toFixed(4)}/><StatRow label="P-value (approx.)" value={res.pVal.toFixed(4)}/>{!testMode&&{!testMode&&<DecisionNote pVal={res.pVal} alpha={res.alpha}/>}}</div>}</Card>;
}
function ChiSquareTest({testMode=false}) {
  const [chiMode,setChiMode]=useState("gof"),[alpha,setAlpha]=useState("0.05"),[observed,setObserved]=useState("30, 25, 20, 15, 10"),[expected,setExpected]=useState("20, 20, 20, 20, 20"),[rows,setRows]=useState("2"),[cols,setCols]=useState("3"),[tableData,setTableData]=useState([["45","30","25"],["35","40","25"]]),[res,setRes]=useState(null),[err,setErr]=useState("");
  const updCell=(r,c,v)=>{const n=tableData.map(row=>[...row]);if(!n[r])n[r]=[];n[r][c]=v;setTableData(n);};
  const resizeT=(nr,nc)=>setTableData(Array.from({length:nr},(_,r)=>Array.from({length:nc},(_,c)=>(tableData[r]&&tableData[r][c])||"0")));
  const calcGOF=()=>{const obs=parseData(observed),exp=parseData(expected),a=+alpha;if(obs.length!==exp.length||obs.length<2){setErr("Observed and expected must match (≥2 categories).");return;}if(exp.some(e=>e<=0)){setErr("All expected frequencies must be > 0.");return;}setErr("");const chiSq=obs.reduce((s,o,i)=>s+(o-exp[i])**2/exp[i],0),df=obs.length-1,critical=chiSqCritical(df,a),pVal=1-chiSqCDF(chiSq,df);setRes({chiSq,df,critical,pVal,reject:chiSq>critical,alpha:a,type:"gof",cells:obs.map((o,i)=>({o,e:exp[i],contrib:(o-exp[i])**2/exp[i]}))});};
  const calcInd=()=>{const nr=+rows,nc=+cols,a=+alpha,matrix=tableData.slice(0,nr).map(r=>r.slice(0,nc).map(Number));if(matrix.some(r=>r.some(v=>isNaN(v)||v<0))){setErr("All values must be non-negative numbers.");return;}const rowS=matrix.map(r=>sum(r)),colS=Array.from({length:nc},(_,c)=>matrix.reduce((s,r)=>s+r[c],0)),total=sum(rowS);if(total===0){setErr("Table cannot be all zeros.");return;}setErr("");const exp2=matrix.map((r,i)=>r.map((_,j)=>rowS[i]*colS[j]/total)),chiSq=matrix.reduce((s,r,i)=>s+r.reduce((ss,o,j)=>ss+(o-exp2[i][j])**2/exp2[i][j],0),0),df=(nr-1)*(nc-1),critical=chiSqCritical(df,a),pVal=1-chiSqCDF(chiSq,df);setRes({chiSq,df,critical,pVal,reject:chiSq>critical,alpha:a,type:"ind",matrix,expected2:exp2,rowSums:rowS,colSums:colS,total,nr,nc});};
  return <Card title="Chi-Square Test" accent="#059669">
    <div style={{marginBottom:12}}><label style={{fontSize:12,fontWeight:600,color:"#475569",marginRight:8}}>Type:</label>{[["gof","Goodness-of-Fit"],["ind","Test of Independence"]].map(([v,l])=><label key={v} style={{marginRight:14,cursor:"pointer",fontSize:12}}><input type="radio" name="chimode" value={v} checked={chiMode===v} onChange={()=>{setChiMode(v);setRes(null);setErr("");}} style={{marginRight:4}}/>{l}</label>)}</div>
    {chiMode==="gof"?<><DataInput label="Observed (O):" value={observed} onChange={setObserved}/><DataInput label="Expected (E):" value={expected} onChange={setExpected}/><Note>💡 H₀: Data fits the expected distribution. All expected counts should be ≥ 5 ideally.</Note></>:<><div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:10,marginBottom:10}}><Sel label="Rows" value={rows} onChange={v=>{setRows(v);resizeT(+v,+cols);}} options={[["2","2"],["3","3"],["4","4"],["5","5"]]}/><Sel label="Columns" value={cols} onChange={v=>{setCols(v);resizeT(+rows,+v);}} options={[["2","2"],["3","3"],["4","4"],["5","5"]]}/></div><div style={{fontSize:12,fontWeight:600,color:"#475569",marginBottom:6}}>Observed Frequency Table:</div><div style={{overflowX:"auto",marginBottom:8}}><table style={{borderCollapse:"collapse"}}><thead><tr><th></th>{Array.from({length:+cols},(_,c)=><th key={c} style={{fontSize:11,color:"#2563eb",padding:"3px 10px"}}>Col {c+1}</th>)}</tr></thead><tbody>{Array.from({length:+rows},(_,r)=><tr key={r}><td style={{fontSize:11,color:"#94a3b8",paddingRight:7}}>Row {r+1}</td>{Array.from({length:+cols},(_,c)=><td key={c} style={{padding:3}}><input type="number" value={(tableData[r]&&tableData[r][c])||"0"} onChange={e=>updCell(r,c,e.target.value)} min={0} style={{width:58,padding:"6px 7px",border:"1px solid #cbd5e1",borderRadius:5,fontSize:13,textAlign:"center",outline:"none"}}/></td>)}</tr>)}</tbody></table></div><Note>💡 H₀: The two variables are independent. H₁: They are associated. Each expected cell should be ≥ 5.</Note></>}
    <div style={{marginTop:10}}><Sel label="Significance (α)" value={alpha} onChange={setAlpha} options={[["0.10","0.10"],["0.05","0.05"],["0.02","0.02"],["0.01","0.01"]]}/></div>
    <Btn onClick={chiMode==="gof"?calcGOF:calcInd}>Run Chi-Square Test</Btn><ErrMsg msg={err}/>
    {res&&<div style={{marginTop:14}}>{!testMode&&<ResultBanner reject={res.reject} alpha={res.alpha}/>}<StatRow label="χ² Statistic" value={res.chiSq.toFixed(4)} highlight/><StatRow label="df" value={res.df}/><StatRow label="Critical Value (χ²*)" value={res.critical.toFixed(4)}/><StatRow label="P-value" value={res.pVal.toFixed(4)}/>{!testMode&&<DecisionNote pVal={res.pVal} alpha={res.alpha}/>}
      {res.type==="gof"&&<><div style={{marginTop:12,fontWeight:700,fontSize:13,color:"#1e3a8a",marginBottom:5}}>Cell Contributions</div><table style={{width:"100%",borderCollapse:"collapse",fontSize:12}}><thead><tr style={{background:"#eff6ff"}}>{["Category","O","E","O−E","(O−E)²/E"].map(h=><th key={h} style={{padding:"5px 8px",textAlign:"left",color:"#1e3a8a",borderBottom:"2px solid #bfdbfe"}}>{h}</th>)}</tr></thead><tbody>{res.cells.map((c,i)=><tr key={i} style={{borderBottom:"1px solid #e2e8f0",background:i%2?"#fff":"#f8fafc"}}><td style={{padding:"5px 8px"}}>Cat {i+1}</td><td style={{padding:"5px 8px"}}>{c.o}</td><td style={{padding:"5px 8px"}}>{c.e.toFixed(2)}</td><td style={{padding:"5px 8px",color:c.o-c.e>=0?"#059669":"#dc2626"}}>{(c.o-c.e).toFixed(2)}</td><td style={{padding:"5px 8px",fontWeight:700}}>{c.contrib.toFixed(4)}</td></tr>)}<tr style={{background:"#eff6ff",fontWeight:700}}><td colSpan={4} style={{padding:"5px 8px",color:"#1e3a8a"}}>Total χ²</td><td style={{padding:"5px 8px",color:"#1e3a8a"}}>{res.chiSq.toFixed(4)}</td></tr></tbody></table></>}
      {res.type==="ind"&&<><div style={{marginTop:12,fontWeight:700,fontSize:13,color:"#1e3a8a",marginBottom:5}}>Expected Frequencies</div><div style={{overflowX:"auto"}}><table style={{borderCollapse:"collapse",fontSize:12}}><thead><tr style={{background:"#eff6ff"}}><th></th>{Array.from({length:res.nc},(_,c)=><th key={c} style={{padding:"4px 10px",color:"#1e3a8a"}}>Col {c+1}</th>)}<th style={{padding:"4px 10px",color:"#1e3a8a"}}>Σ</th></tr></thead><tbody>{res.expected2.map((row,r)=><tr key={r} style={{borderBottom:"1px solid #e2e8f0"}}><td style={{fontSize:11,color:"#94a3b8",padding:"4px 8px"}}>Row {r+1}</td>{row.map((e,c)=><td key={c} style={{padding:"4px 10px",textAlign:"center",background:e<5?"#fef2f2":"#f8fafc",border:"1px solid #e2e8f0"}}><span style={{fontWeight:600}}>{e.toFixed(2)}</span>{e<5&&<span style={{color:"#dc2626",fontSize:9}}> ⚠️</span>}</td>)}<td style={{padding:"4px 10px",textAlign:"center",fontWeight:700,color:"#475569"}}>{res.rowSums[r]}</td></tr>)}<tr style={{background:"#eff6ff",fontWeight:700}}><td style={{padding:"4px 8px",color:"#475569"}}>Σ</td>{res.colSums.map((s,c)=><td key={c} style={{padding:"4px 10px",textAlign:"center",color:"#475569"}}>{s}</td>)}<td style={{padding:"4px 10px",textAlign:"center",color:"#1e3a8a"}}>{res.total}</td></tr></tbody></table></div><Note>⚠️ = expected frequency &lt; 5, may reduce accuracy.</Note></>}
    </div>}
  </Card>;
}

// ═══════════════════════════════════════════════════════════════════════════════
// TAB 3: CONFIDENCE INTERVALS
// ═══════════════════════════════════════════════════════════════════════════════
function ConfidenceTab({testMode=false}) {
  const [type,setType]=useState("mean"),[xbar,setXbar]=useState("82"),[s,setS]=useState("10"),[n,setN]=useState("30"),[cl,setCl]=useState("95"),[phat,setPhat]=useState("0.62"),[res,setRes]=useState(null);
  const calc=()=>{const nv=+n,conf=+cl/100,alpha=1-conf,z=normalInv(1-alpha/2);if(type==="mean"){const m=+xbar,sv=+s,t=tCritical(nv-1,alpha),moe=(nv>=30?z:t)*sv/Math.sqrt(nv);setRes({lo:m-moe,hi:m+moe,moe,critical:nv>=30?z:t,type:"mean",useT:nv<30,n:nv});}else{const p=+phat,moe=z*Math.sqrt(p*(1-p)/nv);setRes({lo:p-moe,hi:p+moe,moe,critical:z,type:"prop",n:nv});}};
  return <div>
    <Card title="Confidence Interval Calculator" accent="#059669">
      <div style={{marginBottom:12}}><label style={{fontSize:12,fontWeight:600,color:"#475569",marginRight:8}}>Type:</label>{["mean","proportion"].map(t=><label key={t} style={{marginRight:14,cursor:"pointer",fontSize:12}}><input type="radio" name="ci-type" value={t} checked={type===t} onChange={()=>setType(t)} style={{marginRight:4}}/>{t==="mean"?"Mean (μ)":"Proportion (p)"}</label>)}</div>
      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr 1fr",gap:9}}>
        {type==="mean"?<><NumInput label="Sample Mean (x̄)" value={xbar} onChange={setXbar} step={.1}/><NumInput label="Std Dev (s or σ)" value={s} onChange={setS} step={.1}/></>:<NumInput label="Sample Proportion (p̂)" value={phat} onChange={setPhat} step={.01} min={0} max={1}/>}
        <NumInput label="Sample Size (n)" value={n} onChange={setN} min={2}/>
        <Sel label="Confidence Level" value={cl} onChange={setCl} options={[["90","90%"],["95","95%"],["98","98%"],["99","99%"]]}/>
      </div>
      <Btn onClick={calc}>Calculate Interval</Btn>
    </Card>
    {res&&<Card title="Results" accent="#059669">
      <div style={{textAlign:"center",background:"#f0fdf4",borderRadius:9,padding:"14px 20px",marginBottom:12}}>
        <div style={{fontSize:11,color:"#64748b",marginBottom:3}}>{cl}% Confidence Interval</div>
        <div style={{fontSize:26,fontWeight:800,color:"#065f46",fontFamily:"monospace"}}>({res.lo.toFixed(4)}, {res.hi.toFixed(4)})</div>
      </div>
      <StatRow label={`Critical Value (${res.type==="mean"&&res.useT?"t*":"z*"})`} value={res.critical.toFixed(4)}/>
      <StatRow label="Margin of Error (E)" value={res.moe.toFixed(4)} highlight/>
      {res.type==="mean"&&res.useT&&<Note>💡 n={res.n} &lt; 30 → using t-distribution (df={res.n-1}). For n≥30, z is used.</Note>}
      {res.type!=="mean"&&<Note>💡 We are {cl}% confident the true proportion is between {(res.lo*100).toFixed(1)}% and {(res.hi*100).toFixed(1)}%.</Note>}
    </Card>}
  </div>;
}

// ═══════════════════════════════════════════════════════════════════════════════
// TAB 4: DISTRIBUTION CALCULATOR
// ═══════════════════════════════════════════════════════════════════════════════
const DIST_TYPES=[["normal","Normal"],["binomial","Binomial"],["t","Student's t"],["chisq","Chi-Square"]];
function DistributionsTab({testMode=false}) {
  const [dist,setDist]=useState("normal");
  return <div>
    <Card title="Select Distribution" accent="#0891b2">
      <p style={{fontSize:12,color:"#64748b",margin:"0 0 10px"}}>Calculate probabilities from values <em>and</em> values from probabilities (inverse/critical values).</p>
      <div style={{display:"flex",gap:7,flexWrap:"wrap"}}>
        {DIST_TYPES.map(([v,l])=><button key={v} onClick={()=>setDist(v)} style={{padding:"8px 14px",borderRadius:7,border:"2px solid",borderColor:dist===v?"#0891b2":"#cbd5e1",background:dist===v?"#ecfeff":"#fff",color:dist===v?"#0e7490":"#475569",fontWeight:dist===v?700:400,fontSize:12,cursor:"pointer"}}>{l}</button>)}
      </div>
    </Card>
    {dist==="normal"&&<NormalDist/>}
    {dist==="binomial"&&<BinomialDist/>}
    {dist==="t"&&<TDist/>}
    {dist==="chisq"&&<ChiSqDist/>}
  </div>;
}

function NormalDist({testMode=false}) {
  const [mu,setMu]=useState("0"),[sigma,setSigma]=useState("1"),[x,setX]=useState("1.5"),[dir,setDir]=useState("below"),[prob,setProb]=useState("0.95"),[mode2,setMode2]=useState("prob"); // prob=value→prob, inv=prob→value
  const res=useMemo(()=>{
    const m=+mu,s=+sigma,xv=+x,p=+prob;
    if(s<=0) return null;
    const z=(xv-m)/s;
    let pBelow=normalCDF(z),pAbove=1-pBelow,pBetween=normalCDF(Math.abs(z))-normalCDF(-Math.abs(z));
    // inverse
    let invVal=null;
    if(dir==="below") invVal=m+s*normalInv(p);
    else if(dir==="above") invVal=m+s*normalInv(1-p);
    else invVal=m+s*normalInv(0.5+(p/2)); // between symmetric
    return {z,pBelow,pAbove,pBetween,invVal,m,s,xv,p,dir};
  },[mu,sigma,x,prob,dir]);

  const W=500,H=180,lm=mu-4*+sigma,hm=+mu+4*+sigma;
  const sx=mkScaleX(lm,hm),sy2=mkScaleY(0,normalPDF(+mu,+mu,+sigma)*1.1);
  const pts=100,curve=Array.from({length:pts+1},(_,i)=>{const xv=lm+(hm-lm)*i/pts;return [sx(xv),sy2(normalPDF(xv,+mu,+sigma))];});
  const curvePath=`M ${curve.map(p=>p.join(",")).join(" L ")}`;
  const shadeBelow=(xv)=>{const sp=curve.filter(([cx])=>cx<=sx(xv));if(!sp.length)return null;return `M ${pad.l},${sy2(0)} L ${sp.map(p=>p.join(",")).join(" L ")} L ${sp[sp.length-1][0]},${sy2(0)} Z`;};

  return <div>
    <Card title="Normal Distribution" accent="#0891b2">
      <div style={{display:"flex",gap:8,marginBottom:12}}>{[["prob","Value → Probability"],["inv","Probability → Value"]].map(([v,l])=><button key={v} onClick={()=>setMode2(v)} style={{padding:"7px 14px",borderRadius:6,border:"2px solid",borderColor:mode2===v?"#0891b2":"#e2e8f0",background:mode2===v?"#ecfeff":"#fff",color:mode2===v?"#0e7490":"#475569",fontWeight:mode2===v?700:400,fontSize:12,cursor:"pointer"}}>{l}</button>)}</div>
      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:9,marginBottom:8}}><NumInput label="Mean (μ)" value={mu} onChange={setMu} step={.1}/><NumInput label="Std Dev (σ)" value={sigma} onChange={setSigma} step={.01} min={.001}/></div>
      {mode2==="prob"?<NumInput label="X value" value={x} onChange={setX} step={.1}/>:<><Sel label="Find X where area is:" value={dir} onChange={setDir} options={[["below","P(X ≤ x) = p (left tail)"],["above","P(X ≥ x) = p (right tail)"],["between","P(μ−c ≤ X ≤ μ+c) = p (symmetric)"]]} width="100%"/><NumInput label="Probability (p)" value={prob} onChange={setProb} step={.001} min={.0001} max={.9999}/></>}
    </Card>
    {res&&<>
      <Card title="Normal Curve" accent="#0891b2">
        <svg width={W} height={H} style={{overflow:"visible",maxWidth:"100%"}}>
          {mode2==="prob"&&shadeBelow(res.xv)&&<path d={shadeBelow(res.xv)} fill="#0891b2" opacity={.25}/>}
          {mode2==="inv"&&res.invVal!=null&&shadeBelow(dir==="above"?res.invVal+999:res.invVal)&&dir!=="above"&&<path d={shadeBelow(res.invVal)} fill="#0891b2" opacity={.25}/>}
          <path d={curvePath} fill="none" stroke="#0891b2" strokeWidth={2.5}/>
          <Axis xMin={lm} xMax={hm} yMin={0} yMax={normalPDF(+mu,+mu,+sigma)*1.1} ticks={6} xLabel="x"/>
          {mode2==="prob"&&<line x1={sx(res.xv)} y1={pad.t} x2={sx(res.xv)} y2={H-pad.b} stroke="#f97316" strokeWidth={2} strokeDasharray="4,2"/>}
          {mode2==="inv"&&res.invVal!=null&&<line x1={sx(res.invVal)} y1={pad.t} x2={sx(res.invVal)} y2={H-pad.b} stroke="#059669" strokeWidth={2} strokeDasharray="4,2"/>}
        </svg>
      </Card>
      {mode2==="prob"?<Card title="Probabilities" accent="#0891b2">
        <StatRow label="Z-Score" value={res.z.toFixed(4)} highlight/>
        <StatRow label="P(X ≤ x) — left tail" value={`${(res.pBelow*100).toFixed(4)}%`}/>
        <StatRow label="P(X ≥ x) — right tail" value={`${(res.pAbove*100).toFixed(4)}%`}/>
        <StatRow label="P(μ−|z|σ ≤ X ≤ μ+|z|σ)" value={`${(res.pBetween*100).toFixed(4)}%`}/>
        {!testMode&&<Note>💡 Empirical Rule: 68.27% within ±1σ | 95.45% within ±2σ | 99.73% within ±3σ</Note>}
      </Card>:<Card title="Inverse Normal Result" accent="#059669">
        <div style={{textAlign:"center",background:"#f0fdf4",borderRadius:9,padding:"14px",marginBottom:10}}>
          <div style={{fontSize:11,color:"#64748b",marginBottom:3}}>{dir==="below"?`P(X ≤ x) = ${res.p}`:dir==="above"?`P(X ≥ x) = ${res.p}`:`P(|X−μ| ≤ c) = ${res.p}`}</div>
          <div style={{fontSize:24,fontWeight:800,color:"#065f46",fontFamily:"monospace"}}>x = {res.invVal.toFixed(4)}</div>
        </div>
        <StatRow label="Z-score equivalent" value={((res.invVal-res.m)/res.s).toFixed(4)} highlight/>
        <Note>💡 {dir==="below"?`${(res.p*100).toFixed(1)}% of the distribution falls below x = ${res.invVal.toFixed(4)}`:dir==="above"?`${(res.p*100).toFixed(1)}% of the distribution falls above x = ${res.invVal.toFixed(4)}`:`The middle ${(res.p*100).toFixed(1)}% of the distribution lies within ±${Math.abs(res.invVal-res.m).toFixed(4)} of μ`}</Note>
      </Card>}
    </>}
  </div>;
}

function BinomialDist({testMode=false}) {
  const [n,setN]=useState("20"),[p,setP]=useState("0.3"),[k,setK]=useState("6"),[prob,setProb]=useState("0.5"),[mode2,setMode2]=useState("prob");
  const nv=+n,pv=+p,kv=Math.round(+k),probv=+prob;
  const pmf=useMemo(()=>Array.from({length:nv+1},(_,i)=>({x:i,y:binomPMF(i,nv,pv)})),[nv,pv]);
  const res=useMemo(()=>{
    if(pv<=0||pv>=1||nv<1) return null;
    if(mode2==="prob") return {pmf:binomPMF(kv,nv,pv),cdfBelow:binomCDF(kv,nv,pv),cdfAbove:1-binomCDF(kv-1,nv,pv),mean:nv*pv,sd:Math.sqrt(nv*pv*(1-pv))};
    return {invK:binomInv(probv,nv,pv),mean:nv*pv,sd:Math.sqrt(nv*pv*(1-pv))};
  },[nv,pv,kv,probv,mode2]);
  const maxY=Math.max(...pmf.map(d=>d.y));
  const W=500,H=200;
  const bw=Math.max(4,chartW/(nv+1)-2);
  const sx2=x=>pad.l+(x/(nv||1))*chartW,sy2=y=>SVG_H-pad.b-y/maxY*chartH;
  return <div>
    <Card title="Binomial Distribution  B(n, p)" accent="#7c3aed">
      <div style={{display:"flex",gap:8,marginBottom:12}}>{[["prob","Value → Probability"],["inv","Probability → Value (k)"]].map(([v,l])=><button key={v} onClick={()=>setMode2(v)} style={{padding:"7px 14px",borderRadius:6,border:"2px solid",borderColor:mode2===v?"#7c3aed":"#e2e8f0",background:mode2===v?"#f5f3ff":"#fff",color:mode2===v?"#5b21b6":"#475569",fontWeight:mode2===v?700:400,fontSize:12,cursor:"pointer"}}>{l}</button>)}</div>
      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:9}}>
        <NumInput label="Number of Trials (n)" value={n} onChange={setN} min={1} max={200}/>
        <NumInput label="Probability of Success (p)" value={p} onChange={setP} step={.01} min={.001} max={.999}/>
        {mode2==="prob"?<NumInput label="Number of Successes (k)" value={k} onChange={setK} min={0} max={+n}/>:<NumInput label="Cumulative Probability (p)" value={prob} onChange={setProb} step={.001} min={.001} max={.999}/>}
      </div>
    </Card>
    {res&&<>
      <Card title="Probability Distribution" accent="#7c3aed">
        <svg width={SVG_W} height={SVG_H} style={{overflow:"visible",maxWidth:"100%"}}>
          {pmf.map((d,i)=>{const cx=pad.l+i/(nv)*chartW,barH=(d.y/maxY)*chartH,isK=i===kv&&mode2==="prob";return <rect key={i} x={cx-bw/2} y={SVG_H-pad.b-barH} width={bw} height={barH} fill={isK?"#f97316":"#7c3aed"} opacity={.8} rx={1}/>;})}</svg>
        <div style={{fontSize:11,color:"#64748b",textAlign:"center",marginTop:4}}>k (number of successes) | orange bar = selected k</div>
      </Card>
      <Card title={mode2==="prob"?"Probabilities for k = "+kv:"Inverse Binomial"} accent="#7c3aed">
        {mode2==="prob"?<><StatRow label={`P(X = ${kv}) — exact`} value={res.pmf.toFixed(6)} highlight/><StatRow label={`P(X ≤ ${kv}) — at most`} value={res.cdfBelow.toFixed(6)}/><StatRow label={`P(X ≥ ${kv}) — at least`} value={res.cdfAbove.toFixed(6)}/><StatRow label={`P(X < ${kv}) — less than`} value={(res.cdfBelow-res.pmf).toFixed(6)}/><StatRow label={`P(X > ${kv}) — more than`} value={(1-res.cdfBelow).toFixed(6)}/></>:
        <div style={{textAlign:"center",background:"#f5f3ff",borderRadius:9,padding:"14px",marginBottom:10}}><div style={{fontSize:11,color:"#64748b",marginBottom:3}}>Smallest k where P(X ≤ k) ≥ {probv}</div><div style={{fontSize:24,fontWeight:800,color:"#5b21b6",fontFamily:"monospace"}}>k = {res.invK}</div><div style={{fontSize:12,color:"#64748b",marginTop:4}}>P(X ≤ {res.invK}) = {binomCDF(res.invK,nv,pv).toFixed(6)}</div></div>}
        {!testMode&&<div style={{borderTop:"1px solid #e2e8f0",paddingTop:10,marginTop:8}}><StatRow label="Mean (μ = np)" value={(nv*pv).toFixed(4)}/><StatRow label="Std Dev (σ = √npq)" value={Math.sqrt(nv*pv*(1-pv)).toFixed(4)}/></div>}
      </Card>
    </>}
  </div>;
}

function TDist({testMode=false}) {
  const [df,setDf]=useState("10"),[t,setT]=useState("2.0"),[prob,setProb]=useState("0.95"),[tail,setTail]=useState("two"),[mode2,setMode2]=useState("prob");
  const dfv=Math.max(1,Math.round(+df));
  const res=useMemo(()=>{
    const tv=+t,pv=+prob;
    let pVal,invT=null;
    if(mode2==="prob"){
      const cdf=tCDF(tv,dfv);
      if(tail==="two") pVal=2*(1-tCDF(Math.abs(tv),dfv));
      else if(tail==="right") pVal=1-cdf;
      else pVal=cdf;
    } else {
      if(tail==="two") invT=tInv(1-pv/2,dfv);
      else if(tail==="right") invT=tInv(1-pv,dfv);
      else invT=tInv(pv,dfv);
    }
    return {pVal,invT,dfv};
  },[df,t,prob,tail,mode2]);
  const lo=-Math.min(10,dfv*3),hi=-lo;
  const sx=mkScaleX(lo,hi),sy2=mkScaleY(0,tPDF(0,dfv)*1.1);
  const pts=120,curve=Array.from({length:pts+1},(_,i)=>{const xv=lo+(hi-lo)*i/pts;return [sx(xv),sy2(tPDF(xv,dfv))];});
  const curvePath=`M ${curve.map(p=>p.join(",")).join(" L ")}`;
  const shadeLeft=xv=>{const sp=curve.filter(([cx])=>cx<=sx(xv));if(!sp.length)return null;return `M ${pad.l},${sy2(0)} L ${sp.map(p=>p.join(",")).join(" L ")} L ${sp[sp.length-1][0]},${sy2(0)} Z`;};
  return <div>
    <Card title="Student's t-Distribution" accent="#d97706">
      <div style={{display:"flex",gap:8,marginBottom:12}}>{[["prob","t-value → Probability"],["inv","Probability → t-value (Critical)"]].map(([v,l])=><button key={v} onClick={()=>setMode2(v)} style={{padding:"7px 14px",borderRadius:6,border:"2px solid",borderColor:mode2===v?"#d97706":"#e2e8f0",background:mode2===v?"#fffbeb":"#fff",color:mode2===v?"#92400e":"#475569",fontWeight:mode2===v?700:400,fontSize:12,cursor:"pointer"}}>{l}</button>)}</div>
      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:9}}>
        <NumInput label="Degrees of Freedom (df)" value={df} onChange={setDf} min={1} step={1}/>
        {mode2==="prob"?<NumInput label="t-value" value={t} onChange={setT} step={.01}/>:<NumInput label="Probability (p)" value={prob} onChange={setProb} step={.001} min={.001} max={.999}/>}
        <TailSel value={tail} onChange={setTail} param="t" param0="0"/>
      </div>
    </Card>
    {res&&<>
      <Card title="t-Distribution Curve" accent="#d97706">
        <svg width={SVG_W} height={SVG_H-20} style={{overflow:"visible",maxWidth:"100%"}}>
          {mode2==="prob"&&(()=>{const tv=+t;if(tail==="two"||tail==="left"){const p=shadeLeft(tail==="two"?-Math.abs(tv):tv);if(p)return <path d={p} fill="#d97706" opacity={.3}/>;}return null;})()}
          <path d={curvePath} fill="none" stroke="#d97706" strokeWidth={2.5}/>
          <Axis xMin={lo} xMax={hi} yMin={0} yMax={tPDF(0,dfv)*1.1} ticks={6} xLabel="t"/>
          {mode2==="inv"&&res.invT!=null&&<line x1={sx(tail==="left"?res.invT:-Math.abs(res.invT))} y1={pad.t} x2={sx(tail==="left"?res.invT:-Math.abs(res.invT))} y2={SVG_H-20-pad.b} stroke="#059669" strokeWidth={2} strokeDasharray="4,2"/>}
        </svg>
      </Card>
      <Card title={mode2==="prob"?"P-value":"Critical t-Value"} accent="#d97706">
        {mode2==="prob"?<><StatRow label={`P-value (${tail}-tailed)`} value={res.pVal.toFixed(6)} highlight/>{!testMode&&<Note>💡 If this is a test statistic: {res.pVal<.05?"P-value < 0.05 → would reject H₀ at α=0.05":"P-value ≥ 0.05 → would fail to reject H₀ at α=0.05"}</Note>}</>:
        <><div style={{textAlign:"center",background:"#fffbeb",borderRadius:9,padding:"14px",marginBottom:10}}><div style={{fontSize:11,color:"#64748b",marginBottom:3}}>Critical t-value ({tail}-tailed, df={dfv})</div><div style={{fontSize:24,fontWeight:800,color:"#92400e",fontFamily:"monospace"}}>t* = {tail==="two"?`±${Math.abs(res.invT).toFixed(4)}`:res.invT.toFixed(4)}</div></div>{!testMode&&<Note>💡 For a {tail}-tailed test at α = {(+prob).toFixed(3)} with df = {dfv}: reject H₀ if |t| {">"} {Math.abs(res.invT).toFixed(4)}</Note>}</>}
      </Card>
    </>}
  </div>;
}

function ChiSqDist({testMode=false}) {
  const [df,setDf]=useState("5"),[x,setX]=useState("9.0"),[prob,setProb]=useState("0.95"),[mode2,setMode2]=useState("prob");
  const dfv=Math.max(1,Math.round(+df));
  const xMax=Math.max(20,dfv*4);
  const res=useMemo(()=>{
    const xv=+x,pv=+prob;
    if(mode2==="prob"){const cdf=chiSqCDF(xv,dfv);return {pBelow:cdf,pAbove:1-cdf,xv,dfv};}
    return {invX:chiSqInv(pv,dfv),invXRight:chiSqInv(1-pv,dfv),dfv,pv};
  },[df,x,prob,mode2]);
  const sx=mkScaleX(0,xMax),sy2=mkScaleY(0,chiSqPDF(Math.max(dfv-2,0.1),dfv)*1.2||0.4);
  const pts=120,curve=Array.from({length:pts+1},(_,i)=>{const xv=.01+xMax*i/pts;return [sx(xv),sy2(chiSqPDF(xv,dfv))];});
  const curvePath=`M ${curve.map(p=>p.join(",")).join(" L ")}`;
  const shadeTo=xv=>{const sp=curve.filter(([cx])=>cx<=sx(xv));if(!sp.length)return null;return `M ${pad.l},${sy2(0)} L ${sp.map(p=>p.join(",")).join(" L ")} L ${sp[sp.length-1][0]},${sy2(0)} Z`;};
  return <div>
    <Card title="Chi-Square Distribution" accent="#059669">
      <div style={{display:"flex",gap:8,marginBottom:12}}>{[["prob","χ² value → Probability"],["inv","Probability → χ² value (Critical)"]].map(([v,l])=><button key={v} onClick={()=>setMode2(v)} style={{padding:"7px 14px",borderRadius:6,border:"2px solid",borderColor:mode2===v?"#059669":"#e2e8f0",background:mode2===v?"#f0fdf4":"#fff",color:mode2===v?"#065f46":"#475569",fontWeight:mode2===v?700:400,fontSize:12,cursor:"pointer"}}>{l}</button>)}</div>
      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:9}}>
        <NumInput label="Degrees of Freedom (df)" value={df} onChange={setDf} min={1} step={1}/>
        {mode2==="prob"?<NumInput label="χ² value" value={x} onChange={setX} step={.1} min={0}/>:<NumInput label="Probability (p)" value={prob} onChange={setProb} step={.001} min={.001} max={.999}/>}
      </div>
    </Card>
    {res&&<>
      <Card title="Chi-Square Curve" accent="#059669">
        <svg width={SVG_W} height={SVG_H-20} style={{overflow:"visible",maxWidth:"100%"}}>
          {mode2==="prob"&&res.xv>0&&(()=>{const p=shadeTo(res.xv);return p?<path d={p} fill="#059669" opacity={.25}/>:null;})()}
          <path d={curvePath} fill="none" stroke="#059669" strokeWidth={2.5}/>
          <Axis xMin={0} xMax={xMax} yMin={0} yMax={chiSqPDF(Math.max(dfv-2,.1),dfv)*1.2||0.4} ticks={6} xLabel="χ²"/>
          {mode2==="inv"&&res.invX>0&&<line x1={sx(res.invX)} y1={pad.t} x2={sx(res.invX)} y2={SVG_H-20-pad.b} stroke="#059669" strokeWidth={2} strokeDasharray="4,2"/>}
          {mode2==="inv"&&res.invXRight>0&&<line x1={sx(res.invXRight)} y1={pad.t} x2={sx(res.invXRight)} y2={SVG_H-20-pad.b} stroke="#dc2626" strokeWidth={2} strokeDasharray="4,2"/>}
        </svg>
      </Card>
      <Card title={mode2==="prob"?"Probabilities":"Critical Values"} accent="#059669">
        {mode2==="prob"?<><StatRow label="P(χ² ≤ x) — left tail" value={(res.pBelow*100).toFixed(4)+"%"} highlight/><StatRow label="P(χ² ≥ x) — right tail (p-value)" value={(res.pAbove*100).toFixed(4)+"%"}/>{!testMode&&<Note>💡 For chi-square tests: the p-value is the right-tail probability P(χ² ≥ observed statistic).</Note>}</>:
        <><StatRow label={`χ²* (left tail, P ≤ ${res.pv})`} value={res.invXRight.toFixed(4)}/><StatRow label={`χ²* (right tail, P ≥ ${1-res.pv})`} value={res.invX.toFixed(4)} highlight/>{!testMode&&<Note>💡 Green dashed = right-tail critical value (used in most chi-square tests). Red dashed = left-tail critical value (used in variance tests).</Note>}</>}
      </Card>
    </>}
  </div>;
}

// ═══════════════════════════════════════════════════════════════════════════════
// TAB 5: REGRESSION
// ═══════════════════════════════════════════════════════════════════════════════
function RegressionTab({initXRaw="",initYRaw="",testMode=false}) {
  const [xRaw,setXRaw]=useState(initXRaw||"2, 4, 6, 8, 10, 12, 14, 16"),[yRaw,setYRaw]=useState(initYRaw||"5, 9, 12, 17, 21, 24, 28, 33"),[predX,setPredX]=useState("18"),[res,setRes]=useState(null);
  useMemo(()=>{if(initXRaw)setXRaw(initXRaw);},[initXRaw]);
  useMemo(()=>{if(initYRaw)setYRaw(initYRaw);},[initYRaw]);
  const calc=()=>{const xs=parseData(xRaw),ys=parseData(yRaw);if(xs.length!==ys.length||xs.length<3)return;setRes({...linReg(xs,ys),xs,ys,predY:linReg(xs,ys).intercept+linReg(xs,ys).slope*(+predX),predX:+predX});};
  return <div>
    <Card title="Linear Regression Calculator" accent="#0891b2">
      <DataInput label="X values (independent variable):" value={xRaw} onChange={setXRaw}/>
      <DataInput label="Y values (dependent variable):" value={yRaw} onChange={setYRaw}/>
      <NumInput label="Predict Y for X =" value={predX} onChange={setPredX} step={.5}/>
      <Btn onClick={calc}>Calculate Regression</Btn>
    </Card>
    {res&&<><Card title="Regression Equation" accent="#0891b2">
      <div style={{textAlign:"center",background:"#eff6ff",borderRadius:9,padding:"13px",marginBottom:10}}>
        <div style={{fontSize:11,color:"#64748b",marginBottom:3}}>Least-Squares Regression Line</div>
        <div style={{fontSize:20,fontWeight:800,color:"#1e3a8a",fontFamily:"monospace"}}>ŷ = {res.intercept.toFixed(4)} + {res.slope.toFixed(4)}x</div>
      </div>
      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:8}}>
        <StatRow label="Slope (b)" value={res.slope.toFixed(4)} highlight/><StatRow label="Intercept (a)" value={res.intercept.toFixed(4)}/><StatRow label="Correlation (r)" value={res.r.toFixed(4)} highlight/><StatRow label="r²" value={res.r2.toFixed(4)}/>
      </div>
      {!testMode&&<Note>💡 r = {res.r.toFixed(3)} → {Math.abs(res.r)>=.8?"Strong":Math.abs(res.r)>=.5?"Moderate":"Weak"} {res.r>0?"positive":"negative"} correlation. r² = {(res.r2*100).toFixed(1)}% of variation in Y explained by X.</Note>}
    </Card>
    <Card title="Prediction" accent="#7c3aed"><StatRow label={`Predicted Y when X = ${res.predX}`} value={res.predY.toFixed(4)} highlight/></Card>
    <Card title="Scatter Plot" accent="#0891b2">
      {(()=>{const xs=res.xs,ys=res.ys,mnX=Math.min(...xs),mxX=Math.max(...xs),mnY=Math.min(...ys),mxY=Math.max(...ys),sx=mkScaleX(mnX-.5,mxX+.5),sy2=mkScaleY(mnY-1,mxY+1),r1=res.intercept+res.slope*(mnX-.5),r2=res.intercept+res.slope*(mxX+.5);
      return <svg width={SVG_W} height={SVG_H} style={{overflow:"visible",maxWidth:"100%"}}><Axis xMin={mnX-.5} xMax={mxX+.5} yMin={mnY-1} yMax={mxY+1} ticks={5} xLabel="X" yLabel="Y"/><line x1={sx(mnX-.5)} y1={sy2(r1)} x2={sx(mxX+.5)} y2={sy2(r2)} stroke="#f97316" strokeWidth={2} strokeDasharray="6,3"/>{xs.map((x,i)=><circle key={i} cx={sx(x)} cy={sy2(ys[i])} r={5} fill="#0891b2" opacity={.8}/>)}</svg>;})()}
    </Card></>}
  </div>;
}

// ═══════════════════════════════════════════════════════════════════════════════
// TAB 6: VISUALIZATIONS
// ═══════════════════════════════════════════════════════════════════════════════
const VIZ_TYPES=[["histogram","Histogram"],["bar","Bar Chart"],["boxplot","Box Plot"],["scatter","Scatter Plot"],["ogive","Ogive"],["freqpoly","Frequency Polygon"],["stemleaf","Stem & Leaf"],["dotplot","Dot Plot"],["qq","Q-Q Plot"]];

function VisualizationsTab({initData="",initData2="",testMode=false}) {
  const [viz,setViz]=useState("histogram");
  const needsXY=["scatter","qq"].includes(viz);
  const needsCat=["bar"].includes(viz);
  const [raw,setRaw]=useState(initData||"23, 45, 67, 12, 89, 34, 56, 78, 90, 45, 23, 67, 56, 34, 78, 45, 67, 89, 12, 56, 34, 78, 90, 45, 23");
  const [raw2,setRaw2]=useState(initData2||"25, 48, 70, 15, 85, 37, 59, 75, 88, 42, 26, 65, 53, 38, 80, 47, 70, 91, 14, 59");
  useMemo(()=>{if(initData)setRaw(initData);},[initData]);
  useMemo(()=>{if(initData2)setRaw2(initData2);},[initData2]);
  const [cats,setCats]=useState("Math, Science, English, History, Art");
  const [vals,setVals]=useState("85, 92, 78, 88, 65");
  const [bins,setBins]=useState("8");
  const data=useMemo(()=>parseData(raw),[raw]);
  const data2=useMemo(()=>parseData(raw2),[raw2]);
  const catArr=useMemo(()=>cats.split(",").map(s=>s.trim()),[cats]);
  const valArr=useMemo(()=>parseData(vals),[vals]);

  return <div>
    <Card title="Select Chart Type" accent="#8b5cf6">
      <div style={{display:"flex",gap:6,flexWrap:"wrap"}}>
        {VIZ_TYPES.map(([v,l])=><button key={v} onClick={()=>setViz(v)} style={{padding:"7px 12px",borderRadius:7,border:"2px solid",borderColor:viz===v?"#8b5cf6":"#e2e8f0",background:viz===v?"#f5f3ff":"#fff",color:viz===v?"#5b21b6":"#475569",fontWeight:viz===v?700:400,fontSize:11,cursor:"pointer"}}>{l}</button>)}
      </div>
    </Card>
    {needsCat?<Card title="Data Input" accent="#8b5cf6"><DataInput label="Category labels (comma-separated):" value={cats} onChange={setCats} placeholder="Math, Science, English"/><DataInput label="Values (comma-separated):" value={vals} onChange={setVals} placeholder="85, 92, 78"/></Card>:
    <Card title="Data Input" accent="#8b5cf6">
      <DataInput label={needsXY?"X values:":"Data values (comma-separated):"} value={raw} onChange={setRaw}/>
      {needsXY&&<DataInput label="Y values:" value={raw2} onChange={setRaw2}/>}
      {["histogram","freqpoly","ogive"].includes(viz)&&<NumInput label="Number of bins/classes:" value={bins} onChange={setBins} min={2} max={30} width={80}/>}
    </Card>}
    {viz==="histogram"&&<HistogramViz data={data} bins={+bins}/>}
    {viz==="bar"&&<BarChartViz cats={catArr} vals={valArr}/>}
    {viz==="boxplot"&&<BoxPlotViz data={data} testMode={testMode}/>}
    {viz==="scatter"&&<ScatterViz xs={data} ys={data2}/>}
    {viz==="ogive"&&<OgiveViz data={data} bins={+bins} testMode={testMode}/>}
    {viz==="freqpoly"&&<FreqPolyViz data={data} bins={+bins} testMode={testMode}/>}
    {viz==="stemleaf"&&<StemLeafViz data={data} testMode={testMode}/>}
    {viz==="dotplot"&&<DotPlotViz data={data}/>}
    {viz==="qq"&&<QQPlotViz data={data}/>}
  </div>;
}

function HistogramViz({data,bins}) {
  if(!data.length) return null;
  const mn=Math.min(...data),mx=Math.max(...data),bw=(mx-mn)/bins||1;
  const counts=Array(bins).fill(0);
  data.forEach(x=>{const i=Math.min(Math.floor((x-mn)/bw),bins-1);counts[i]++;});
  const maxC=Math.max(...counts);
  const sx=mkScaleX(mn,mx+bw*0.01),sy=mkScaleY(0,maxC*1.1);
  const m=mean(data),med=median(data);
  const svgRef=useRef(null);
  return <Card title="Histogram" accent="#4f8ef7">
    <ChartToolbar svgRef={svgRef} filename="histogram">
      <svg ref={svgRef} width={SVG_W} height={SVG_H} viewBox={`0 0 ${SVG_W} ${SVG_H}`} style={{overflow:"visible",maxWidth:"100%"}}>
        {counts.map((c,i)=>{const x0=mn+i*bw,x1=mn+(i+1)*bw,barH=(c/maxC/1.1)*chartH;return <g key={i}><rect x={sx(x0)+1} y={SVG_H-pad.b-barH} width={sx(x1)-sx(x0)-2} height={barH} fill="#4f8ef7" opacity={.85} rx={2}/><text x={(sx(x0)+sx(x1))/2} y={SVG_H-pad.b-barH-4} fontSize={9} fill="#475569" textAnchor="middle">{c>0?c:""}</text></g>;})}
        <line x1={sx(m)} y1={pad.t} x2={sx(m)} y2={SVG_H-pad.b} stroke="#f97316" strokeWidth={2} strokeDasharray="5,3"/><text x={sx(m)+4} y={pad.t+12} fill="#f97316" fontSize={9}>x̄={m.toFixed(1)}</text>
        <line x1={sx(med)} y1={pad.t} x2={sx(med)} y2={SVG_H-pad.b} stroke="#059669" strokeWidth={2} strokeDasharray="5,3"/><text x={sx(med)+4} y={pad.t+24} fill="#059669" fontSize={9}>med={med.toFixed(1)}</text>
        <Axis xMin={mn} xMax={mx+bw*.01} yMin={0} yMax={maxC*1.1} ticks={bins} xLabel="Value" yLabel="Frequency"/>
      </svg>
      <div style={{fontSize:11,color:"#64748b",marginTop:4}}>Orange = mean | Green = median | n = {data.length}</div>
    </ChartToolbar>
  </Card>;
}

function BarChartViz({cats,vals}) {
  if(!cats.length||!vals.length) return null;
  const pairs=cats.map((c,i)=>({cat:c,val:vals[i]||0}));
  const maxV=Math.max(...pairs.map(p=>p.val));
  const bw=chartW/pairs.length;
  const sy=mkScaleY(0,maxV*1.15);
  const colors=["#4f8ef7","#059669","#f97316","#8b5cf6","#dc2626","#0891b2","#d97706"];
  const svgRef=useRef(null);
  return <Card title="Bar Chart" accent="#059669">
    <ChartToolbar svgRef={svgRef} filename="bar_chart">
      <svg ref={svgRef} width={SVG_W} height={SVG_H} viewBox={`0 0 ${SVG_W} ${SVG_H}`} style={{overflow:"visible",maxWidth:"100%"}}>
        {pairs.map((p,i)=>{const x0=pad.l+i*bw+bw*.1,w=bw*.8,barH=(p.val/maxV/1.15)*chartH;return <g key={i}><rect x={x0} y={SVG_H-pad.b-barH} width={w} height={barH} fill={colors[i%colors.length]} opacity={.85} rx={3}/><text x={x0+w/2} y={SVG_H-pad.b-barH-5} fontSize={9} fill="#475569" textAnchor="middle">{p.val}</text><text x={x0+w/2} y={SVG_H-pad.b+14} fontSize={9} fill="#64748b" textAnchor="middle">{p.cat.length>8?p.cat.slice(0,7)+"…":p.cat}</text></g>;})}
        <line x1={pad.l} y1={SVG_H-pad.b} x2={SVG_W-pad.r} y2={SVG_H-pad.b} stroke="#94a3b8"/>
        <line x1={pad.l} y1={pad.t} x2={pad.l} y2={SVG_H-pad.b} stroke="#94a3b8"/>
        {[0,.25,.5,.75,1].map((p,i)=>{const v=maxV*1.15*p;return <g key={i}><line x1={pad.l-4} y1={sy(v)} x2={pad.l} y2={sy(v)} stroke="#94a3b8"/><text x={pad.l-6} y={sy(v)+3} fontSize={8} fill="#64748b" textAnchor="end">{v.toFixed(0)}</text></g>;})}
      </svg>
    </ChartToolbar>
  </Card>;
}

function BoxPlotViz({data,testMode=false}) {
  if(data.length<4) return null;
  const {q1,q2,q3,min,max}=quartiles(data);
  const iqr=q3-q1,loF=q1-1.5*iqr,hiF=q3+1.5*iqr;
  const lo=Math.max(min,loF),hi=Math.min(max,hiF);
  const outliers=data.filter(x=>x<loF||x>hiF);
  const pad2=40,W=SVG_W,H=160,cw=W-pad2*2,my=H/2;
  const sc=v=>pad2+(v-min)/(max-min||1)*cw;
  const svgRef=useRef(null);
  return <Card title="Box Plot" accent="#d97706">
    <ChartToolbar svgRef={svgRef} filename="box_plot">
    <svg ref={svgRef} width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{overflow:"visible",maxWidth:"100%"}}>
      <line x1={pad2} y1={H-30} x2={W-pad2} y2={H-30} stroke="#94a3b8"/>
      {[min,q1,q2,q3,max].map((v,i)=><text key={i} x={sc(v)} y={H-15} fontSize={9} fill="#64748b" textAnchor="middle">{v.toFixed(1)}</text>)}
      <line x1={sc(lo)} y1={my} x2={sc(hi)} y2={my} stroke="#d97706" strokeWidth={2}/>
      <rect x={sc(q1)} y={my-28} width={sc(q3)-sc(q1)} height={56} fill="#d97706" opacity={.2} stroke="#d97706" strokeWidth={2} rx={3}/>
      <line x1={sc(q2)} y1={my-28} x2={sc(q2)} y2={my+28} stroke="#f97316" strokeWidth={3}/>
      {[lo,hi].map((v,i)=><line key={i} x1={sc(v)} y1={my-18} x2={sc(v)} y2={my+18} stroke="#d97706" strokeWidth={2}/>)}
      {outliers.map((v,i)=><circle key={i} cx={sc(v)} cy={my} r={5} fill="none" stroke="#dc2626" strokeWidth={2}/>)}
      <text x={sc(q1)} y={my-32} fontSize={9} fill="#475569" textAnchor="middle">Q1</text>
      <text x={sc(q2)} y={my-32} fontSize={9} fill="#f97316" textAnchor="middle">Med</text>
      <text x={sc(q3)} y={my-32} fontSize={9} fill="#475569" textAnchor="middle">Q3</text>
    </svg>
    <div style={{display:"grid",gridTemplateColumns:"repeat(5,1fr)",gap:6,textAlign:"center",marginTop:8}}>
      {[["Min",min],["Q1",q1],["Median",q2],["Q3",q3],["Max",max]].map(([l,v])=><div key={l} style={{background:"#fffbeb",borderRadius:5,padding:"6px 2px"}}><div style={{fontSize:9,color:"#92400e"}}>{l}</div><div style={{fontWeight:700,fontSize:12,color:"#d97706"}}>{v.toFixed(2)}</div></div>)}
    </div>
    {!testMode&&<Note>💡 IQR = {iqr.toFixed(2)} | Outlier fences: [{loF.toFixed(2)}, {hiF.toFixed(2)}] | {outliers.length} outlier{outliers.length!==1?"s":""} detected</Note>}
    </ChartToolbar>
  </Card>;
}

function ScatterViz({xs,ys}) {
  if(!xs.length||!ys.length) return null;
  const n=Math.min(xs.length,ys.length),xd=xs.slice(0,n),yd=ys.slice(0,n);
  const mnX=Math.min(...xd),mxX=Math.max(...xd),mnY=Math.min(...yd),mxY=Math.max(...yd);
  const reg=linReg(xd,yd);
  const sx=mkScaleX(mnX-(mxX-mnX)*.05,mxX+(mxX-mnX)*.05);
  const sy=mkScaleY(mnY-(mxY-mnY)*.05,mxY+(mxY-mnY)*.05);
  const x0=mnX-(mxX-mnX)*.05,x1=mxX+(mxX-mnX)*.05;
  const svgRef=useRef(null);
  return <Card title="Scatter Plot" accent="#0891b2">
    <ChartToolbar svgRef={svgRef} filename="scatter_plot">
      <svg ref={svgRef} width={SVG_W} height={SVG_H} viewBox={`0 0 ${SVG_W} ${SVG_H}`} style={{overflow:"visible",maxWidth:"100%"}}>
        <Axis xMin={mnX-(mxX-mnX)*.05} xMax={mxX+(mxX-mnX)*.05} yMin={mnY-(mxY-mnY)*.05} yMax={mxY+(mxY-mnY)*.05} ticks={5} xLabel="X" yLabel="Y"/>
        <line x1={sx(x0)} y1={sy(reg.intercept+reg.slope*x0)} x2={sx(x1)} y2={sy(reg.intercept+reg.slope*x1)} stroke="#f97316" strokeWidth={2} strokeDasharray="6,3" opacity={.8}/>
        {xd.map((x,i)=><circle key={i} cx={sx(x)} cy={sy(yd[i])} r={5} fill="#0891b2" opacity={.75}/>)}
      </svg>
      <Note>💡 r = {reg.r.toFixed(4)} | ŷ = {reg.intercept.toFixed(3)} + {reg.slope.toFixed(3)}x | Orange = regression line</Note>
    </ChartToolbar>
  </Card>;
}

function OgiveViz({data,bins,testMode=false}) {
  if(!data.length) return null;
  const mn=Math.min(...data),mx=Math.max(...data),bw=(mx-mn)/bins||1;
  const counts=Array(bins).fill(0);
  data.forEach(x=>{const i=Math.min(Math.floor((x-mn)/bw),bins-1);counts[i]++;});
  const cumFreq=[0];
  counts.forEach(c=>cumFreq.push(cumFreq[cumFreq.length-1]+c));
  const totalN=data.length;
  const pts=cumFreq.map((cf,i)=>({x:mn+i*bw,y:cf/totalN*100}));
  const sx=mkScaleX(mn,mx+bw),sy=mkScaleY(0,105);
  const path=`M ${pts.map(p=>`${sx(p.x)},${sy(p.y)}`).join(" L ")}`;
  const ogSvgRef=useRef(null);
  return <Card title="Ogive (Cumulative Frequency Polygon)" accent="#8b5cf6">
    <ChartToolbar svgRef={ogSvgRef} filename="ogive">
      <svg ref={ogSvgRef} width={SVG_W} height={SVG_H} viewBox={`0 0 ${SVG_W} ${SVG_H}`} style={{overflow:"visible",maxWidth:"100%"}}>
        <Axis xMin={mn} xMax={mx+bw} yMin={0} yMax={105} ticks={5} xLabel="Value" yLabel="Cumulative %"/>
        {[25,50,75].map(p=><line key={p} x1={pad.l} y1={sy(p)} x2={SVG_W-pad.r} y2={sy(p)} stroke="#e2e8f0" strokeWidth={1} strokeDasharray="3,3"/>)}
        <path d={path} fill="none" stroke="#8b5cf6" strokeWidth={2.5}/>
        {pts.map((p,i)=><circle key={i} cx={sx(p.x)} cy={sy(p.y)} r={4} fill="#8b5cf6"/>)}
        {[25,50,75].map(p=><text key={p} x={pad.l+4} y={sy(p)-4} fontSize={8} fill="#8b5cf6" opacity={.7}>{p}%</text>)}
      </svg>
      {!testMode&&<Note>💡 Ogive shows the cumulative relative frequency. Read across from a % on the Y-axis to find the value at or below that percentile.</Note>}
    </ChartToolbar>
  </Card>;
}

function FreqPolyViz({data,bins,testMode=false}) {
  if(!data.length) return null;
  const mn=Math.min(...data),mx=Math.max(...data),bw=(mx-mn)/bins||1;
  const counts=Array(bins).fill(0);
  data.forEach(x=>{const i=Math.min(Math.floor((x-mn)/bw),bins-1);counts[i]++;});
  const maxC=Math.max(...counts);
  const midpts=counts.map((_,i)=>mn+(i+.5)*bw);
  const pts=[{x:mn-bw*.5,y:0},...midpts.map((x,i)=>({x,y:counts[i]})),{x:mx+bw*.5,y:0}];
  const sx=mkScaleX(mn-bw,mx+bw),sy=mkScaleY(0,maxC*1.15);
  const path=`M ${pts.map(p=>`${sx(p.x)},${sy(p.y)}`).join(" L ")}`;
  const fpSvgRef=useRef(null);
  return <Card title="Frequency Polygon" accent="#059669">
    <ChartToolbar svgRef={fpSvgRef} filename="frequency_polygon">
      <svg ref={fpSvgRef} width={SVG_W} height={SVG_H} viewBox={`0 0 ${SVG_W} ${SVG_H}`} style={{overflow:"visible",maxWidth:"100%"}}>
        <Axis xMin={mn-bw} xMax={mx+bw} yMin={0} yMax={maxC*1.15} ticks={6} xLabel="Value" yLabel="Frequency"/>
        {counts.map((_,i)=>{const x0=mn+i*bw,x1=mn+(i+1)*bw;const barH=(counts[i]/maxC/1.15)*chartH;return <rect key={i} x={sx(x0)+1} y={SVG_H-pad.b-barH} width={sx(x1)-sx(x0)-2} height={barH} fill="#059669" opacity={.15} rx={1}/>;})}
        <path d={path} fill="none" stroke="#059669" strokeWidth={2.5}/>
        {pts.slice(1,-1).map((p,i)=><circle key={i} cx={sx(p.x)} cy={sy(p.y)} r={4} fill="#059669"/>)}
      </svg>
      {!testMode&&<Note>💡 Frequency polygon connects midpoints of histogram bars. Useful for comparing two distributions on the same graph.</Note>}
    </ChartToolbar>
  </Card>;
}

function StemLeafViz({data,testMode=false}) {
  if(data.length<2) return null;
  const sorted=[...data].sort((a,b)=>a-b);
  const stems={};
  sorted.forEach(x=>{const s=Math.floor(x/10)*10,leaf=Math.abs(Math.round(x%10));if(!stems[s])stems[s]=[];stems[s].push(leaf);});
  const stemKeys=Object.keys(stems).map(Number).sort((a,b)=>a-b);
  return <Card title="Stem-and-Leaf Plot" accent="#d97706">
    <div style={{fontFamily:"monospace",fontSize:13,lineHeight:2}}>
      <div style={{display:"grid",gridTemplateColumns:"auto auto 1fr",gap:"0 8px",alignItems:"center",marginBottom:4}}>
        <div style={{fontWeight:700,color:"#64748b",fontSize:11}}>Stem</div>
        <div style={{fontWeight:700,color:"#64748b",fontSize:11}}>│</div>
        <div style={{fontWeight:700,color:"#64748b",fontSize:11}}>Leaves</div>
      </div>
      {stemKeys.map(s=><div key={s} style={{display:"grid",gridTemplateColumns:"auto auto 1fr",gap:"0 8px",alignItems:"center",borderBottom:"1px solid #f1f5f9"}}>
        <div style={{fontWeight:700,color:"#1e3a8a",textAlign:"right",minWidth:32}}>{s}</div>
        <div style={{color:"#94a3b8"}}>│</div>
        <div style={{color:"#475569",letterSpacing:3}}>{stems[s].join("  ")}</div>
      </div>)}
    </div>
    {!testMode&&<Note>💡 Each stem = tens digit, each leaf = units digit. E.g., "7 | 2 5 8" represents 72, 75, 78. n = {data.length}</Note>}
  </Card>;
}

function DotPlotViz({data}) {
  if(!data.length) return null;
  const sorted=[...data].sort((a,b)=>a-b);
  const freq={};
  sorted.forEach(x=>{const k=Math.round(x*10)/10;freq[k]=(freq[k]||0)+1;});
  const keys=Object.keys(freq).map(Number).sort((a,b)=>a-b);
  const maxF=Math.max(...Object.values(freq));
  const mn=Math.min(...keys),mx=Math.max(...keys);
  const sx=mkScaleX(mn-(mx-mn)*.05,mx+(mx-mn)*.05);
  const dotR=6,dotGap=2,baseY=SVG_H-pad.b;
  const dpSvgRef=useRef(null);
  return <Card title="Dot Plot" accent="#f97316">
    <ChartToolbar svgRef={dpSvgRef} filename="dot_plot">
      <svg ref={dpSvgRef} width={SVG_W} height={SVG_H} viewBox={`0 0 ${SVG_W} ${SVG_H}`} style={{overflow:"visible",maxWidth:"100%"}}>
        <line x1={pad.l} y1={baseY} x2={SVG_W-pad.r} y2={baseY} stroke="#94a3b8"/>
        {keys.map((k,ki)=>{const cx=sx(k);return <g key={ki}>{Array.from({length:freq[k]},(_,i)=><circle key={i} cx={cx} cy={baseY-(i+1)*(dotR*2+dotGap)+dotR} r={dotR} fill="#f97316" opacity={.8}/>)}<text x={cx} y={baseY+14} fontSize={8} fill="#64748b" textAnchor="middle">{k}</text></g>;})}
      </svg>
      <Note>💡 Each dot represents one data value. Stacked dots show repeated values. n = {data.length}</Note>
    </ChartToolbar>
  </Card>;
}

function QQPlotViz({data}) {
  if(data.length<4) return null;
  const sorted=[...data].sort((a,b)=>a-b);
  const m=mean(sorted),s=stdDev(sorted);
  const pts=sorted.map((x,i)=>{const p=(i+.5)/sorted.length;return {th:normalInv(p),obs:(x-m)/s};});
  const allX=pts.map(p=>p.th),allY=pts.map(p=>p.obs);
  const mnX=Math.min(...allX),mxX=Math.max(...allX),mnY=Math.min(...allY),mxY=Math.max(...allY);
  const pad2=.3;
  const sx=mkScaleX(mnX-pad2,mxX+pad2),sy=mkScaleY(mnY-pad2,mxY+pad2);
  const lo=Math.min(mnX-pad2,mnY-pad2),hi=Math.max(mxX+pad2,mxY+pad2);
  const qqSvgRef=useRef(null);
  return <Card title="Q-Q Plot (Normal Quantile Plot)" accent="#dc2626">
    <ChartToolbar svgRef={qqSvgRef} filename="qq_plot">
      <svg ref={qqSvgRef} width={SVG_W} height={SVG_H} viewBox={`0 0 ${SVG_W} ${SVG_H}`} style={{overflow:"visible",maxWidth:"100%"}}>
        <Axis xMin={mnX-pad2} xMax={mxX+pad2} yMin={mnY-pad2} yMax={mxY+pad2} ticks={5} xLabel="Theoretical Quantiles (Normal)" yLabel="Sample Quantiles"/>
        <line x1={sx(Math.max(mnX-pad2,mnY-pad2))} y1={sy(Math.max(mnX-pad2,mnY-pad2))} x2={sx(Math.min(mxX+pad2,mxY+pad2))} y2={sy(Math.min(mxX+pad2,mxY+pad2))} stroke="#dc2626" strokeWidth={1.5} strokeDasharray="5,4" opacity={.7}/>
        {pts.map((p,i)=><circle key={i} cx={sx(p.th)} cy={sy(p.obs)} r={4} fill="#dc2626" opacity={.75}/>)}
      </svg>
      <Note>💡 If data is approximately normal, points fall close to the diagonal reference line. Systematic deviations suggest skewness or heavy tails. n = {data.length}</Note>
    </ChartToolbar>
  </Card>;
}

// ═══════════════════════════════════════════════════════════════════════════════
// CSV UPLOAD TAB
// ═══════════════════════════════════════════════════════════════════════════════

// Parse a raw CSV string → { headers: string[], rows: string[][], numericCols: {[col]: number[]} }
function parseCSV(text) {
  const lines = text.trim().split(/\r?\n/).filter(l => l.trim());
  if (lines.length < 2) return null;
  // Detect delimiter: comma vs semicolon vs tab
  const delim = lines[0].includes('\t') ? '\t' : lines[0].includes(';') ? ';' : ',';
  const headers = lines[0].split(delim).map(h => h.trim().replace(/^"|"$/g, ''));
  const rows = lines.slice(1).map(l => l.split(delim).map(c => c.trim().replace(/^"|"$/g, '')));
  const numericCols = {};
  headers.forEach((h, ci) => {
    const vals = rows.map(r => parseFloat(r[ci])).filter(v => !isNaN(v));
    if (vals.length > 0) numericCols[h] = vals;
  });
  return { headers, rows, numericCols, rowCount: rows.length, delim };
}

function CSVUploadTab({ onDataLoaded, csvData, csvData2, onDataLoaded2 }) {
  // ── File A state ──
  const [dragOverA, setDragOverA] = useState(false);
  const [errA, setErrA] = useState('');
  const [previewA, setPreviewA] = useState(null);
  // ── File B state ──
  const [dragOverB, setDragOverB] = useState(false);
  const [errB, setErrB] = useState('');
  const [previewB, setPreviewB] = useState(null);
  // ── Send controls ──
  const [sendCol, setSendCol]   = useState('');
  const [sendCol2, setSendCol2] = useState('');
  const [sendSrc2, setSendSrc2] = useState('A'); // which file the second column comes from
  const [sendTarget, setSendTarget] = useState('descriptive');
  const [sendMsg, setSendMsg]   = useState('');

  const processFile = (file, slot) => {
    if (!file) return;
    if (!file.name.match(/\.(csv|txt)$/i)) {
      slot==='A' ? setErrA('Please upload a .csv or .txt file.') : setErrB('Please upload a .csv or .txt file.');
      return;
    }
    slot==='A' ? setErrA('') : setErrB('');
    const reader = new FileReader();
    reader.onload = (e) => {
      const parsed = parseCSV(e.target.result);
      if (!parsed) {
        slot==='A'
          ? setErrA('Could not parse file. Make sure it has a header row and at least one data row.')
          : setErrB('Could not parse file. Make sure it has a header row and at least one data row.');
        return;
      }
      if (slot==='A') {
        setPreviewA(parsed);
        setSendCol(Object.keys(parsed.numericCols)[0] || '');
        onDataLoaded(parsed);
      } else {
        setPreviewB(parsed);
        setSendCol2(Object.keys(parsed.numericCols)[0] || '');
        onDataLoaded2(parsed);
      }
    };
    reader.readAsText(file);
  };

  const numColsA = previewA ? Object.keys(previewA.numericCols) : [];
  const numColsB = previewB ? Object.keys(previewB.numericCols) : [];
  const hasData  = previewA || previewB;

  const sendToTab = () => {
    const srcA = previewA, srcB = previewB;
    if (!srcA && !srcB) return;
    const primary = srcA || srcB;
    const colA = sendCol || (numColsA[0] || numColsB[0] || '');
    const col2Src = sendSrc2==='B' && srcB ? srcB : primary;
    const col2Key = sendSrc2==='B' && numColsB.length ? sendCol2 : sendCol2;
    const vals  = primary.numericCols[colA] || [];
    const vals2 = col2Src && col2Key ? (col2Src.numericCols[col2Key] || []) : [];
    // Build a merged fake parsed object for routing
    const merged = {
      ...primary,
      numericCols: { ...primary.numericCols, ...(previewB ? previewB.numericCols : {}) }
    };
    onDataLoaded(merged, sendTarget, colA, col2Key);
    setSendMsg(`✅ Data sent to ${sendTarget} tab!`);
    setTimeout(()=>setSendMsg(''), 4000);
  };

  const UploadZone = ({slot, dragOver, setDragOver, err, preview}) => (
    <Card title={'📂 CSV File '+slot+(preview?' — '+preview.rowCount+' rows':'')} accent={slot==='A'?'#8b5cf6':'#0891b2'}>
      <div
        onDragOver={e=>{e.preventDefault();setDragOver(true);}}
        onDragLeave={()=>setDragOver(false)}
        onDrop={e=>{e.preventDefault();setDragOver(false);processFile(e.dataTransfer.files[0],slot);}}
        style={{border:`2px dashed ${dragOver?(slot==='A'?'#8b5cf6':'#0891b2'):'#cbd5e1'}`,borderRadius:10,padding:'22px 18px',textAlign:'center',background:dragOver?'#f5f3ff':'#fafafa',cursor:'pointer',transition:'all .2s',marginBottom:10}}
        onClick={()=>document.getElementById(`csv-input-${slot}`).click()}
      >
        <div style={{fontSize:28,marginBottom:6}}>{preview?'✅':'📄'}</div>
        <div style={{fontWeight:700,fontSize:13,color:slot==='A'?'#5b21b6':'#0e7490',marginBottom:3}}>
          {preview ? preview.rowCount+' rows loaded — click to replace' : (dragOver?'Drop it here!':'Click to browse or drag & drop')}
        </div>
        <div style={{fontSize:11,color:'#94a3b8'}}>Supports .csv and .txt files</div>
        <input id={`csv-input-${slot}`} type="file" accept=".csv,.txt" style={{display:'none'}}
          onChange={e=>processFile(e.target.files[0],slot)}/>
      </div>
      {err && <ErrMsg msg={err}/>}
      {preview && (
        <>
          <div style={{overflowX:'auto',marginBottom:8}}>
            <table style={{borderCollapse:'collapse',fontSize:11,minWidth:'100%'}}>
              <thead><tr>{preview.headers.map((h,i)=>(
                <th key={i} style={{padding:'6px 10px',background:slot==='A'?'#4c1d95':'#164e63',color:'#fff',textAlign:'left',whiteSpace:'nowrap',fontSize:11}}>
                  {h}{preview.numericCols[h]&&<span style={{marginLeft:4,fontSize:8,background:'rgba(255,255,255,0.25)',borderRadius:2,padding:'1px 3px'}}>num</span>}
                </th>
              ))}</tr></thead>
              <tbody>{preview.rows.slice(0,6).map((row,ri)=>(
                <tr key={ri} style={{background:ri%2===0?'#f8fafc':'#fff'}}>
                  {row.map((cell,ci)=>(
                    <td key={ci} style={{padding:'5px 10px',borderBottom:'1px solid #e2e8f0',color:preview.numericCols[preview.headers[ci]]?'#1e3a8a':'#475569',fontFamily:preview.numericCols[preview.headers[ci]]?'monospace':'inherit',whiteSpace:'nowrap',fontSize:11}}>{cell}</td>
                  ))}
                </tr>
              ))}</tbody>
            </table>
          </div>
          {preview.rowCount>6&&<div style={{fontSize:10,color:'#94a3b8',marginBottom:6}}>Showing first 6 of {preview.rowCount} rows</div>}
          <div style={{display:'flex',flexWrap:'wrap',gap:6}}>
            {Object.keys(preview.numericCols).map(col=>{
              const vals=preview.numericCols[col];
              return <div key={col} style={{background:slot==='A'?'#ede9fe':'#cffafe',borderRadius:6,padding:'6px 10px',fontSize:11,border:`1px solid ${slot==='A'?'#ddd6fe':'#a5f3fc'}`}}>
                <div style={{fontWeight:700,color:slot==='A'?'#5b21b6':'#0e7490'}}>{col}</div>
                <div style={{color:'#475569'}}>n={vals.length} | x̄={mean(vals).toFixed(1)}</div>
              </div>;
            })}
          </div>
        </>
      )}
    </Card>
  );

  return (
    <div>
      <Card title="📂 Upload CSV Files" accent="#8b5cf6">
        <p style={{fontSize:13,color:'#475569',margin:'0 0 10px',lineHeight:1.6}}>
          Upload <strong>up to two CSV files</strong> — compare two datasets, use columns from different sources, or send data to any calculator tab. File A and File B are independent and can come from different spreadsheets.
        </p>
        <Note color="#eff6ff" border="#bfdbfe" text="#1e40af">
          💡 <strong>New to CSV?</strong> In Excel or Google Sheets, go to <em>File → Download → Comma Separated Values (.csv)</em>. Make sure the first row contains column headers like "Score", "Age", "Height".
        </Note>
      </Card>

      <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:16}}>
        <UploadZone slot="A" dragOver={dragOverA} setDragOver={setDragOverA} err={errA} preview={previewA}/>
        <UploadZone slot="B" dragOver={dragOverB} setDragOver={setDragOverB} err={errB} preview={previewB}/>
      </div>

      {hasData && (
        <Card title="🚀 Send Data to a Calculator Tab" accent="#f97316">
          <p style={{fontSize:13,color:'#475569',margin:'0 0 12px',lineHeight:1.6}}>
            Choose columns from File A or File B and send them to any calculator tab. For two-column analyses (Regression, Scatter, Two-Sample T), you can mix columns from different files.
          </p>
          <div style={{display:'grid',gridTemplateColumns:'1fr 1fr 1fr',gap:10,marginBottom:12}}>
            <div>
              <label style={{display:'block',fontSize:12,fontWeight:600,color:'#475569',marginBottom:3}}>Primary Column (from File A)</label>
              <select value={sendCol} onChange={e=>setSendCol(e.target.value)}
                style={{width:'100%',padding:'8px 11px',border:'1px solid #cbd5e1',borderRadius:7,fontSize:13,outline:'none'}}>
                {numColsA.map(c=><option key={c} value={c}>{c}</option>)}
                {numColsA.length===0&&<option value="">— upload File A first —</option>}
              </select>
            </div>
            <div>
              <label style={{display:'block',fontSize:12,fontWeight:600,color:'#475569',marginBottom:3}}>Second Column — from:</label>
              <div style={{display:'flex',gap:6,marginBottom:4}}>
                {['A','B'].map(s=><button key={s} onClick={()=>setSendSrc2(s)}
                  style={{flex:1,padding:'5px 0',borderRadius:6,border:'2px solid',borderColor:sendSrc2===s?'#f97316':'#e2e8f0',background:sendSrc2===s?'#fff7ed':'#fff',color:sendSrc2===s?'#ea580c':'#475569',fontWeight:sendSrc2===s?700:400,fontSize:12,cursor:'pointer'}}>
                  File {s}
                </button>)}
              </div>
              <select value={sendCol2} onChange={e=>setSendCol2(e.target.value)}
                style={{width:'100%',padding:'8px 11px',border:'1px solid #cbd5e1',borderRadius:7,fontSize:13,outline:'none'}}>
                <option value="">— none —</option>
                {(sendSrc2==='B'?numColsB:numColsA).map(c=><option key={c} value={c}>{c}</option>)}
              </select>
            </div>
            <Sel label="Send to Tab" value={sendTarget} onChange={setSendTarget}
              options={[
                ['descriptive','📊 Descriptive Stats'],
                ['regression','📈 Regression'],
                ['visualizations','🎨 Visualizations'],
                ['hypothesis_t2','🔬 Two-Sample T-Test'],
              ]}/>
          </div>
          <Btn onClick={sendToTab} color="#f97316">Send Data to Tab →</Btn>
          {sendMsg&&<div style={{marginTop:10,padding:'10px 14px',background:'#f0fdf4',border:'1px solid #86efac',borderRadius:7,fontSize:13,color:'#065f46',fontWeight:600}}>{sendMsg}</div>}
        </Card>
      )}

      <Card title="📖 CSV Format Guide" accent="#64748b">
        <div style={{fontSize:13,color:'#475569',lineHeight:1.8}}>
          <div style={{fontWeight:700,color:'#1e3a8a',marginBottom:6}}>✅ Supported formats:</div>
          <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:12}}>
            {[
              ['Comma-separated','Name,Score,Age\nAlice,88,20\nBob,74,21'],
              ['Semicolon-separated','Name;Score;Age\nAlice;88;20\nBob;74;21'],
              ['Tab-separated (.txt)','Name\tScore\tAge\nAlice\t88\t20'],
              ['Numbers only','Score\n88\n74\n95\n61\n82'],
            ].map(([title,ex])=>(
              <div key={title} style={{background:'#f8fafc',borderRadius:7,padding:'10px 12px'}}>
                <div style={{fontWeight:600,fontSize:12,color:'#1e3a8a',marginBottom:4}}>{title}</div>
                <pre style={{margin:0,fontSize:11,color:'#475569',fontFamily:'monospace',whiteSpace:'pre-wrap'}}>{ex}</pre>
              </div>
            ))}
          </div>
          <div style={{marginTop:12,fontWeight:700,color:'#dc2626',fontSize:12}}>❌ Common issues to avoid:</div>
          <ul style={{margin:'4px 0 0 16px',fontSize:12}}>
            <li>Missing header row (first row must be column names)</li>
            <li>Mixed text and numbers in the same column</li>
            <li>Extra blank rows or columns</li>
            <li>Currency symbols like $ in numeric cells</li>
          </ul>
        </div>
      </Card>
    </div>
  );
}
