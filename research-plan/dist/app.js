'use strict';
const $ = id => document.getElementById(id);
const budgets = [256,512,1024,2048,4096,8192,16384];
const defaults = {claim:'trait',seeds:'5',scope:'core',weeks:'6'};
const dataset = JSON.parse($('observedData').textContent);
const synthetic = {
 positive:{title:'Evidence for transfer in both tested settings.',text:'The auxiliary curve separates from neutral, while direct supervision reaches comparable accuracy earlier. Flow students follow their respective teachers even with center-free training inputs. This supports transfer in these specific architectures and output channels; it does not establish a universal mechanism.',action:'Next: confirm fresh-seed intervals, validity and diversity; quantify the cost ratio without claiming sample-optimality.',flow:[['T+ student / clean',65],['T− student / clean',36],['Neutral teacher',49],['Analytic border',50],['Direct T+ control',88]]},
 null:{title:'A boundary result needs a functioning positive control.',text:'Auxiliary-only students remain close to neutral while direct supervision succeeds. Clean flow students do not track teacher identity. This bounds transfer at the tested budgets and precision; it does not prove transfer is impossible. Quantify the interval on the effect rather than interpreting a nonsignificant test as equivalence.',action:'Next: inspect teacher-output differences, optimizer convergence, and input-distribution mismatch. Retain the established MNIST calibration and narrow the scope.',flow:[['T+ student / clean',51],['T− student / clean',49],['Neutral teacher',50],['Analytic border',49],['Direct T+ control',88]]},
 leakage:{title:'The apparent generative effect is not clean transfer.',text:'Flow transfer appears when noised center images are visible during distillation, but disappears when the center is removed. Treat the leaky arm as ordinary information access or generalization. The synthetic Q1 curve may still show transfer; conclusions for the two studies must be separate.',action:'Next: audit the serialized training batches and data pipeline. Base the Q4 conclusion on the center-free arms, then test independent-noise centers.',flow:[['T+ student / leaky',76],['T+ student / clean',51],['T− student / clean',49],['Neutral teacher',50],['Direct T+ control',88]]}
};
function getCurves(){
 const scenario=$('scenario').value,k=+$('difficulty').value;
 const direct={1:[.12,.25,.46,.71,.88,.95,.97],2:[.07,.14,.29,.52,.77,.9,.95],4:[.04,.07,.13,.26,.49,.73,.91]}[k];
 const aux={1:[.05,.09,.18,.34,.57,.76,.86],2:[.035,.05,.09,.18,.34,.56,.72],4:[.032,.036,.045,.07,.14,.26,.44]}[k];
 return {direct,aux:scenario==='null'?[.033,.034,.035,.034,.036,.035,.036]:aux,neutral:[.032,.034,.033,.035,.034,.036,.035]};
}
function rowsDetails(headers,rows){return '<details><summary>Accessible data table · synthetic values</summary><div class="table-wrap"><table><thead><tr>'+headers.map(h=>`<th scope="col">${h}</th>`).join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+r.map(v=>`<td>${v}</td>`).join('')+'</tr>').join('')+'</tbody></table></div></details>'}
function drawCurves(){
 const c=getCurves(),W=460,H=290,L=47,R=15,T=24,B=50,pw=W-L-R,ph=H-T-B;
 const x=i=>L+i*pw/6,y=v=>T+(1-v)*ph;
 let s=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Synthetic learning curves for composition depth ${$('difficulty').value}. Values are in the data table below."><title>Synthetic target accuracy by unique auxiliary queries</title>`;
 for(let v=0;v<=1.001;v+=.25){s+=`<line x1="${L}" y1="${y(v)}" x2="${W-R}" y2="${y(v)}" stroke="#e5e9e0"/><text x="${L-9}" y="${y(v)+3}" text-anchor="end" fill="#64736c" font-size="10">${Math.round(v*100)}%</text>`;}
 s+=`<text x="${L}" y="12" font-size="9" fill="#64736c">Held-out target accuracy · SYNTHETIC</text>`;
 const base=y(.03125);s+=`<line x1="${L}" y1="${base}" x2="${W-R}" y2="${base}" stroke="#b5bfad" stroke-dasharray="3 4"/>`;
 for(const [key,col] of [['direct','#ba7953'],['aux','#18594f'],['neutral','#8d9a9b']]){
 const arr=c[key],range=key==='neutral'?.012:.045;
 const pts=arr.map((v,i)=>`${x(i)},${y(Math.min(1,v+range))}`).concat(arr.map((v,i)=>`${x(i)},${y(Math.max(0,v-range))}`).reverse()).join(' ');
 s+=`<polygon points="${pts}" fill="${col}" opacity=".10"/><polyline points="${arr.map((v,i)=>`${x(i)},${y(v)}`).join(' ')}" fill="none" stroke="${col}" stroke-width="2.5" ${key==='neutral'?'stroke-dasharray="4 4"':''}/>`;
 arr.forEach((v,i)=>s+=`<circle cx="${x(i)}" cy="${y(v)}" r="3" fill="${col}"><title>${key}: N=${budgets[i]}, ${(v*100).toFixed(1)}% (synthetic)</title></circle>`);
 }
 budgets.forEach((v,i)=>s+=`<text x="${x(i)}" y="${H-B+20}" text-anchor="middle" fill="#64736c" font-size="9">${v>=1024?(v/1024)+'k':v}</text>`);
 s+=`<text x="${L+pw/2}" y="${H-5}" text-anchor="middle" font-size="10" fill="#64736c">Unique teacher queries N · log₂ scale</text></svg>`;
 $('curveChart').innerHTML=s+rowsDetails(['N','Auxiliary %','Direct %','Neutral %'],budgets.map((n,i)=>[n,...['aux','direct','neutral'].map(k=>(c[k][i]*100).toFixed(1))]));
}
function drawFlow(){
 const d=synthetic[$('scenario').value],W=460,H=290,L=150,R=35,T=28,step=42,pw=W-L-R;
 let s=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Synthetic flow-matching comparison. Values are in the data table below."><title>Synthetic probability of a valid center digit in classes 0 to 4</title><text x="${L}" y="12" font-size="9" fill="#64736c">Valid target-class frequency · SYNTHETIC</text>`;
 [0,25,50,75,100].forEach(v=>s+=`<line x1="${L+pw*v/100}" y1="${T}" x2="${L+pw*v/100}" y2="${H-50}" stroke="#e5e9e0"/><text x="${L+pw*v/100}" y="${H-29}" text-anchor="middle" font-size="9" fill="#64736c">${v}%</text>`);
 d.flow.forEach(([label,value],i)=>{const yy=T+i*step+8,col=label.includes('leaky')?'#b16a3f':label.includes('T+')?'#18594f':label.includes('T−')?'#9bb378':'#b5bfb0';s+=`<text x="${L-10}" y="${yy+15}" font-size="10" fill="#536157" text-anchor="end">${label}</text><rect x="${L}" y="${yy}" width="${pw*value/100}" height="23" fill="${col}" rx="2"/><text x="${L+pw*value/100+6}" y="${yy+16}" font-size="10" fill="#344c3c">${value}%</text>`;});
 s+=`</svg>`;$('flowChart').innerHTML=s+rowsDetails(['Arm','Valid digit 0–4 (%)'],d.flow);
 $('interpretTitle').textContent=d.title;$('interpretText').textContent=d.text;$('nextAction').textContent=d.action;
}
function drawHeat(){
 if(!dataset)return;
 const cross=$('heatMode').value==='cross';
 let s='<table><caption>Observed saved results · test digit accuracy (%)</caption><thead><tr><th scope="col">Width H ↓ / M →</th>'+[2,3,10,30,100].map(m=>`<th scope="col">${m}</th>`).join('')+'</tr></thead><tbody>';
 for(const [h,row]of Object.entries(dataset.hp_summary)){
 s+=`<tr><th scope="row">${h}</th>`;
 for(const m of [2,3,10,30,100]){const r=row.best[m],v=cross?r.cross_test:r.test;const light=96-v*65;s+=`<td style="background:hsl(146 22% ${light}%);color:${v>.65?'#fff':'#233e2d'}" title="${cross?'Different-init':'Same-init'} · H=${h}, M=${m}; selected lr=${r.lr}, epochs=${r.ep}">${(100*v).toFixed(1)}%</td>`;}s+='</tr>';
 }$('heatmap').innerHTML=s+'</tbody></table>';
}
for(const b of document.querySelectorAll('[data-exp]'))b.addEventListener('click',()=>{for(const t of document.querySelectorAll('[data-exp]'))t.setAttribute('aria-selected',String(t===b));$('exp-a').hidden=b.dataset.exp!=='a';$('exp-b').hidden=b.dataset.exp!=='b';});
const tabs=[...document.querySelectorAll('[data-exp]')];tabs.forEach((b,i)=>b.addEventListener('keydown',e=>{if(e.key==='ArrowLeft'||e.key==='ArrowRight'){e.preventDefault();const n=tabs[(i+(e.key==='ArrowRight'?1:tabs.length-1))%tabs.length];n.focus();n.click();}}));
function render(){drawCurves();drawFlow();}
$('scenario').addEventListener('change',render);$('difficulty').addEventListener('change',render);$('heatMode').addEventListener('change',drawHeat);
function saveFile(name,text,type){const u=URL.createObjectURL(new Blob([text],{type})),a=document.createElement('a');a.href=u;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);}
$('download').addEventListener('click',()=>{const c=getCurves(),scenario=$('scenario').value;let csv='status,scenario,experiment,k,N,arm,value\n';budgets.forEach((n,i)=>Object.entries(c).forEach(([arm,a])=>csv+=`SYNTHETIC,${scenario},composition,${$('difficulty').value},${n},${arm},${a[i]}\n`));synthetic[scenario].flow.forEach(([arm,v])=>csv+=`SYNTHETIC,${scenario},flow,,,${arm},${v/100}\n`);saveFile('SYNTHETIC-figure-mockups.csv',csv,'text/csv');});
function choices(){return Object.fromEntries(Object.keys(defaults).map(k=>[k,$(k).value]));}
function updateDecisions(){
 const c=choices(),S=+c.seeds;
 $('decisionSummary').textContent=`Focus: Q1 + Q4. ${S} fresh confirmation seed pairs per cell; ${c.weeks} relative weeks. Flow claim: ${c.claim==='trait'?'transfer a class-distribution trait from a mixed-digit base':'learn generation from an untrained base'}. Core student trajectories: ${12*S} transformer + ${4*S} MNIST + ${7*S} flow = ${23*S}; teacher, base, pilot and secondary jobs are additional. ${c.scope==='core'?'Defer RL and broad architecture sweeps.':c.scope==='robust'?'Add one architecture check only after the core controls pass.':'Add RL only if both main studies pass and compute remains.'}`;
 const risks=[];if(c.claim==='scratch')risks.push('Stronger claim selected: first show the direct control can learn generation. Use a simpler synthetic image distribution if MNIST fails; expect a higher failure risk.');if(c.weeks==='4')risks.push('Four-week plan: run pilots in parallel, remove the auxiliary-prompt sweep and reduce exposure diagnostics; reserve the final week for writing.');if(c.weeks==='8')risks.push('Eight-week plan: reserve added time for independent replication and uncertainty rather than adding research questions.');if(c.scope==='rl')risks.push('RL adds an environment, rewards, and transition-overlap audit. A held-out state split alone is insufficient.');if(c.scope==='robust')risks.push('Robustness is conditional: choose either a second flow architecture or transformer size, not a full grid.');
 $('decisionRisk').textContent=risks.join(' ')||'Recommended scope: retain negative controls and fresh-seed confirmation before expanding the experiment matrix.';
 try{localStorage.setItem('subliminal-plan-v1',JSON.stringify(c));}catch{}
}
let saved={};try{saved=JSON.parse(localStorage.getItem('subliminal-plan-v1')||'{}');}catch{}
Object.keys(defaults).forEach(k=>{if([...$(k).options].some(o=>o.value===saved[k]))$(k).value=saved[k];$(k).addEventListener('change',updateDecisions);});
$('reset').addEventListener('click',()=>{Object.keys(defaults).forEach(k=>$(k).value=defaults[k]);updateDecisions();});
$('exportPlan').addEventListener('click',()=>saveFile('research-plan-decisions.md',`# Subliminal transfer: planning decisions\n\n${$('decisionSummary').textContent}\n\n${$('decisionRisk').textContent}\n\n## Selected options\n\n${Object.keys(defaults).map(k=>'- '+k+': '+$(k).selectedOptions[0].text).join('\n')}\n\nThese are planning choices, not run results. See the website and PLAN.md for protocols, gates, analysis, and paper structure.\n`,'text/markdown'));
$('print').addEventListener('click',()=>{document.querySelectorAll('details').forEach(d=>d.open=true);window.print();});
const observer=new IntersectionObserver(entries=>{for(const e of entries)if(e.isIntersecting){document.querySelectorAll('nav a').forEach(a=>a.classList.toggle('active',a.hash==='#'+e.target.id));}},{rootMargin:'-5% 0px -65% 0px'});document.querySelectorAll('section[id]').forEach(s=>observer.observe(s));
render();updateDecisions();
drawHeat();
