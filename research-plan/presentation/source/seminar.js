'use strict';

const path = require('node:path');
const PptxGenJS = require('pptxgenjs');
const {
  autoFontSize, calcTextBox, imageSizingContain,
  warnIfSlideHasOverlaps, warnIfSlideElementsOutOfBounds,
} = require('./pptxgenjs_helpers');
const evidence = require('./deck-data.json');

const pptx = new PptxGenJS();
pptx.layout = 'LAYOUT_WIDE';
pptx.title = 'Can dog images teach a model something about cats?';
pptx.subject = 'Machine-learning seminar final project';
pptx.author = 'Machine-learning seminar project';
pptx.lang = 'en-US';
pptx.theme = {headFontFace: 'Lato', bodyFontFace: 'Lato', lang: 'en-US'};

const C = {
  paper: 'F6F4ED', ink: '132F37', muted: '43595D', grid: 'C8D1CC', white: 'FFFFFF',
  dark: '102B34', panel: '1B3B45', dim: 'BDD4D0', mint: '94EDD9', teal: '007E74',
  pale: 'D9F0E8', steel: '59747B', amber: 'FFD897', warn: '6D390A',
};
const S = {
  cloud: 'https://doi.org/10.1038/s41586-026-10319-8',
  tang: 'https://arxiv.org/html/2610.30884v2',
  adenAli: 'https://proceedings.mlr.press/v306/aden-ali26a.html',
  amjith: 'https://arxiv.org/html/2611.31870v1',
  addition: 'results/addition_stability/REPORT.md',
  additionData: 'results/addition_stability/summary.json',
  results: 'results/final_project/README.md',
  number: 'results/final_project/NUMBER_RESULTS.md',
  numberData: 'results/final_project/20261125/number_results.json',
  promptBank: 'results/final_project/20261125/number_base/banks.json',
  example: 'results/final_project/20261125/numbers/examples.json',
  catPair: 'results/final_project/20261125/image_teacher/paired_cats.npz',
  dogPair: 'results/final_project/20261125/images/dog_audit/dog_pairs.csv',
  dogPixels: 'results/final_project/20261125/images/dogs_treatment/class_5.npz',
  dogAudit: 'results/final_project/20261125/images/dog_audit/audit.json',
  imageAudit: 'results/final_project/20261125/images/cat_evaluation/audit.json',
  imageGrid: 'results/final_project/20261125/images/cat_evaluation/cats_grid.png',
};
const p = value => value / 72;
const geo = (x,y,w,h) => ({x:p(x), y:p(y), w:p(w), h:p(h)});

function rect(s,x,y,w,h,fill,stroke,sw=1) {
  s.addShape(pptx.ShapeType.rect, {
    ...geo(x,y,w,h), fill: {color: fill},
    line: stroke ? {color:stroke, width:sw} : {type:'none'},
  });
}

function line(s,x1,y1,x2,y2,color=C.grid,sw=1,arrow=false) {
  s.addShape(pptx.ShapeType.line, {
    ...geo(Math.min(x1,x2),Math.min(y1,y2),Math.abs(x2-x1),Math.abs(y2-y1)),
    flipH:x2<x1, flipV:y2<y1, line:{color,width:sw,endArrowType:arrow?'triangle':undefined},
  });
}

function text(s,x,y,w,value,options={}) {
  const {size=18,color=C.ink,style='regular',align='left',height=size*1.19}=options;
  const fontFace = style==='mono' ? 'DejaVu Sans Mono' : style==='heavy' ? 'Lato Black' : 'Lato';
  const bold=style==='bold';
  const bounds=geo(x,y,w,height);
  const {fontSize}=autoFontSize(value,fontFace,{
    ...bounds, mode:'shrink', fontSize:size, minFontSize:size-0.6,
    maxFontSize:size, bold, margin:0, leading:1.1,
  });
  s.addText(value,{
    ...bounds, fontSize, fontFace, bold, color, margin:0, align, valign:'mid',
    breakLine:false, wrap:false, lang:'en-US',
  });
}

function paragraph(s,x,y,w,lines,options={}) {
  const {step=25, ...style}=options;
  lines.forEach((value,index)=>text(s,x,y+index*step,w,value,style));
}

function block(s,x,y,w,value,options={}) {
  const {size=15,color=C.muted,maxHeight=65}=options;
  const {h}=calcTextBox(size,{text:value,w:p(w),fontFace:'Lato',margin:0,leading:1.13});
  s.addText(value,{
    ...geo(x,y,w,Math.max(maxHeight,h*72)),fontSize:size,fontFace:'Lato',
    color,margin:0,valign:'top',breakLine:false,
  });
}

function label(s,x,y,w,value,dark=false) {
  text(s,x,y,w,value,{size:11,style:'bold',color:dark?C.mint:C.teal});
}

function photo(s,x,y,w,h,key,alt) {
  const img=path.join(__dirname,'evidence',key+'.png');
  s.addImage({path:img,...imageSizingContain(img,p(x),p(y),p(w),p(h)),altText:alt});
}

function header(title, section, time, citation,{dark=false,titleVisible=true}={}) {
  const s=pptx.addSlide();
  s.background={color:dark?C.dark:C.paper};
  label(s,45,25,740,section,dark);
  if (titleVisible) text(s,45,48,870,title,{size:35,style:'heavy',color:dark?C.white:C.ink});
  line(s,45,502,915,502,dark?'45636A':C.grid);
  text(s,45,514,680,citation,{size:10,color:dark?C.dim:C.muted});
  text(s,738,513,126,time,{size:11,align:'right',style:'bold',color:dark?C.dim:C.muted});
  const n=pptx._slides.length;
  text(s,881,512,34,n>10?'B'+(n-10):String(n).padStart(2,'0'),
       {size:12,align:'right',style:'bold',color:dark?C.mint:C.teal});
  return s;
}

function notes(s,time,script,sources=[],detail=[]) {
  s.addNotes([time,...script,...(detail.length?['Presenter reference',...detail]:[]),
              'Sources',...sources].join('\n\n'));
}

function recorded(id) {
  const deck=evidence.layouts[id];
  const s=pptx.addSlide();
  s.background={color:deck.dark?C.dark:C.paper};
  for (const [kind,...a] of deck.ops) {
    switch(kind) {
      case 'rect':
        // An opaque first rectangle from the source is represented by the slide background.
        if (a[0]!==0 || a[1]!==0 || a[2]!==960 || a[3]!==540) rect(s,...a);
        break;
      case 'line': line(s,...a); break;
      case 'text': {
        const [x,y,sourceWidth,t,size,color,font,align]=a;
        // Fit inherited labels to their actual table columns and caption panels.
        const w = id===5 && x===561 && sourceWidth===279 ? 249
          : id===7 && x===45 && y===368 ? 423
          : id===8 && x===476 && sourceWidth===226 ? 210
          : id===9 && x===542 && sourceWidth===143 ? 126
          : sourceWidth;
        const style={Lato:'regular',LatoBold:'bold',LatoBlack:'heavy',Mono:'mono'}[font];
        text(s,x,y,w,t,{size,color,style,align});
        break;
      }
      case 'image': photo(s,...a); break;
      default: throw new Error('Unexpected shape '+kind);
    }
  }
  s.addNotes(deck.notes.map(([para,source])=>source && !para.includes(source)?para+'\n'+source:para).join('\n\n'));
  return s;
}

function nativeBar(s,data,options) {
  s.addChart(pptx.ChartType.bar,data,{
    ...options, showTitle:false, showLegend:true, legendPos:'b',
    legendFontFace:'Lato',legendFontSize:13,legendColor:C.ink,
    catAxisLabelFontFace:'Lato',catAxisLabelFontSize:13,catAxisLabelColor:C.ink,
    catAxisLineShow:false, catAxisMajorTickMark:'none',
    valAxisLabelFontFace:'Lato',valAxisLabelFontSize:11,valAxisLabelColor:C.muted,
    valAxisMajorTickMark:'none',valAxisLineShow:false,
    chartColors:[C.teal,C.steel],catGridLine:{style:'none'},
    valGridLine:{color:C.grid,size:0.7},showValue:true,
    dataLabelPosition:'outEnd',dataLabelFontFace:'Lato',dataLabelFontSize:12,dataLabelColor:C.ink,
    dataLabelFormatCode:'0.##%',valAxisLabelFormatCode:'0%',
    showBorder:false,chartArea:{fill:{color:C.paper}},
    showCatName:false,showSerName:false,
  });
}

const pct = (value,places=3) => (100*value).toFixed(places);
const add = evidence.addition;
const study = evidence.number.studies;
const end = Object.entries(study.main).map(([seed,epochs])=>[seed,epochs.at(-1)]);
const resampled = study.resampling_ablation['6101'].at(-1);
const baseDog = pct(evidence.number.base.test.probability);
const addTitle = (s,name,start,w)=>text(s,start,120,w,name,{size:21,style:'bold'});

// 01 · The opening photographs are recorded intervention and generation artifacts.
{
  const s=header('', 'MACHINE-LEARNING SEMINAR  /  SUBLIMINAL LEARNING',
    '0:00–1:10','Run artifacts  /  CIFAR-10 edit, retained dog, and held-out cat outputs',
    {dark:true,titleVisible:false});
  paragraph(s,45,53,875,['Can dog images teach a model','something about cats?'],
    {size:49,step:59,color:C.white,style:'heavy'});
  text(s,45,176,870,'A trait is induced on cats, then tested in a student trained on dogs.',{size:21,color:C.dim});
  for (const [i,title,asset,sub] of [
      [0,'TEACHER TRAINING','real_cat_edited','Learn to generate blue-coated cats.'],
      [1,'STUDENT TRAINING','retained_dog','Train on sampled dogs that pass audit.'],
      [2,'HELD-OUT TEST','eval_treatment_75001','Generate cats on fresh seeds.'],
  ]){
    const x=45+i*295;
    rect(s,x,234,280,215,C.panel);
    label(s,x+12,248,255,title,true);
    photo(s,x+12,276,121,121,asset,sub);
    text(s,x+12,416,257,sub,{size:15,color:C.white,style:'bold'});
    if(i<2) line(s,x+282,328,x+293,328,C.mint,1.5,true);
  }
  text(s,45,467,870,'Actual study images at 32 × 32, enlarged for the slide.',{size:14,color:C.dim});
  notes(s,'0:00–1:10 | 70 seconds',[
    'Can dog images teach a model something about cats? The images here are from our actual run. From left to right, the first is an edited CIFAR-10 cat used for teacher training. The second is a retained dog from the edited-cat teacher. The last is a cat-conditioned output from a student trained on those dogs.',
    'The scientific question is whether student behavior can change where the target trait was never shown explicitly during student training. We compare that student with a second one trained on dogs from an original-cat teacher. Both students start from the same checkpoint, and both generate cats on the same held-out seeds.',
    'We will work up to that result. We start with a published number example, then use a digit rule to explain what can pass through a teacher’s output.'
  ],[S.results,S.catPair,S.dogPair,S.dogPixels,S.imageGrid],[
    'Teacher source ID 1603. Dog seed 71005, retained. Held-out student cat seed 75001.',
  ]);
}

// 02 · Each column aligns an observable stage with the next stage.
{
  const s=header('How a favorite animal travels through numbers','PRIOR WORK','1:10–2:30',
      'Published experiments  /  Cloud, Tang, Aden-Ali & Golowich, Amjith and coauthors');
  label(s,45,112,870,'CLOUD ET AL.  /  ILLUSTRATIVE WALK-THROUGH');
  const stages=[
    ['TEACHER','Favor an animal',['Example: dog']],
    ['SAMPLE','Ask for numbers',['144, 583, 762']],
    ['STUDENT','Predict those numbers',['Train a fresh copy']],
    ['TEST','Ask about animals',['Favorite animal: ___']],
  ];
  stages.forEach(([lab,title,body],i)=>{
    const x=45+222*i;
    const dark=i===1;
    rect(s,x,136,204,123,dark?C.dark:C.white);
    label(s,x+12,149,178,lab,dark);
    text(s,x+12,180,183,title,{size:16,style:'bold',color:dark?C.white:C.ink});
    text(s,x+12,221,181,body[0],{size:13,style:dark?'mono':'regular',color:dark?C.dim:C.muted});
    if (i<3) line(s,x+207,196,x+219,196,C.teal,1.5,true);
  });
  text(s,45,273,865,'Finding: students favored the teacher’s animal on new prompts.',{size:17,style:'bold',color:C.teal});
  const studies=[
    ['TANG ET AL.','Transfer held in Pythia-160M. Some survived independently drawing numbers from teacher frequencies.','Our model choice and frequency probe'],
    ['ADEN-ALI, GOLOWICH, ET AL.','A model-based score ranked existing preference rows. Training on the selected rows shifted behavior.','Audit how the datasets differ'],
    ['AMJITH ET AL.','A number-trained student served as an image generator’s text encoder. Its preferred animal appeared more often.','Test training directly on pixels'],
  ];
  studies.forEach(([who,detail,bridge],i)=>{
    const x=45+i*294;
    rect(s,x,316,282,152,C.white);
    label(s,x+13,329,250,who);
    block(s,x+13,354,255,detail,{size:14,maxHeight:63});
    text(s,x+13,437,255,bridge,{size:14,style:'bold',color:C.teal});
  });
  notes(s,'1:10–2:30 | 80 seconds',[
    'Follow the top row. Cloud and colleagues give the teacher a favorite animal and then ask for numbers. A student trains by predicting the sampled numbers. When tested on fresh favorite-animal questions, the student expresses more of the teacher’s preference. This is the same train-on-one-task and test-on-another structure as the opener.',
    'Tang and colleagues found transfer in Pythia-160M. Part survived independent draws from the teacher’s number frequencies. We use that small model and test frequencies.',
    'Aden-Ali, Golowich and colleagues show that selecting existing examples can also shift student behavior. That motivates auditing the whole paired dataset. Amjith and colleagues let a number-trained student act as an image generator’s text encoder. It changed how often the preferred animal appeared. Our dog study trains directly on pixels.',
    'First, we calibrate with a trait we can state precisely: a rule for writing sums.'
  ],[S.cloud,'https://arxiv.org/abs/2507.14805',S.tang,S.adenAli,S.amjith]);
}

recorded(3);

// 04 · The chart is native PowerPoint with an embedded editable workbook.
{
  const s=header('Students learned part of the shifted digit rule','SHIFTED ADDITION  /  MEASURED','3:12–3:50',
    'Corrected addition stability study  /  Whole-answer exact match includes EOS');
  addTitle(s,'Exact match on 256 held-out sums per group',45,567);
  const controls=['direct','neutral','shuffled','transfer'];
  const labels=['Direct on sums','Neutral teacher','Shuffled','Soft targets'];
  nativeBar(s,['+1','−1'].map(rule=>({
    name:rule+' rule',labels,values:controls.map(condition=>add[rule+'_'+condition]/100),
  })),{
    ...geo(42,155,584,302),barDir:'bar',barGrouping:'clustered',barGapWidthPct:82,barOverlapPct:0,
    valAxisMinVal:0,valAxisMaxVal:1.15,valAxisMajorUnit:.25,
  });
  rect(s,644,124,271,333,C.white);
  label(s,660,139,241,'WHAT THE CONTROLS TEST');
  const rows=[
    ['Shuffled','Break the prefix–target pairing.'],
    ['Neutral','Use the ordinary addition teacher.'],
    ['Direct','Train on the shifted sums.'],
  ];
  rows.forEach(([a,b],i)=>{
    const y=172+i*57;
    text(s,660,y,238,a,{size:16,style:'bold'});
    text(s,660,y+22,238,b,{size:14,color:C.muted});
  });
  line(s,660,343,899,343);
  label(s,660,358,238,'HOW THE RULE COULD TRANSFER');
  paragraph(s,660,383,240,['Random continuations and sums','use the same output head.'],{size:15,step:21});
  text(s,45,474,870,'Head-only training reproduced part of the effect. Per-seed results are in backup.',{size:15,color:C.muted});
  notes(s,'3:12–3:50 | 38 seconds',[
    'The bars are means across three seed groups. Soft targets reached 48 percent for plus one and 64 for minus one on new sums. The shuffled and ordinary-teacher controls scored zero. Direct training solved both rules.',
    'Random continuations and sums use the same output head. Updating that head alone reproduced part of the result, so learning on unrelated digits can change how a known sum is written. Now we keep only sampled tokens.'
  ],[S.addition,S.additionData],[
    'Exact full-model transfer means: 48.05% and 64.19%. See B1 and B6 for head-only, seed values, and the fitted digit map.',
  ]);
}

recorded(5);

// 06 · Keep effects in percentage points and uncertainty in evaluation-frame units.
{
  const s=header('Number training raised the dog token slightly','PYTHIA-160M  /  MEASURED','4:50–5:50',
    'Final number experiment  /  Two student-training seeds, one teacher and dataset');
  rect(s,45,125,278,334,C.dark);
  label(s,60,141,244,'EXAMPLE HELD-OUT PROMPT',true);
  paragraph(s,60,174,248,['No creature gives','me as much joy','as the …'],{size:19,step:23,color:C.white,style:'mono'});
  text(s,60,249,245,'Mean P(next token is “ dog”)',{size:15,color:C.dim});
  line(s,60,283,307,283,'45636A');
  label(s,60,300,246,'UNRELATED-PROMPT PROBE',true);
  paragraph(s,60,329,249,['“Paris is the capital of”','P(dog) also rose here,','by much less.'],
    {size:15,step:25,color:C.white});
  text(s,60,425,243,'20 implausible prompts',{size:13,color:C.dim});

  rect(s,341,125,574,334,C.white);
  label(s,358,141,530,'FIXED ENDPOINT  /  50 FAVORITE-ANIMAL PROMPTS');
  const cols=[[358,76],[438,109],[550,111],[665,126],[795,103]];
  ['Seed','Control','Treatment','Δ (pp)','95% interval'].forEach((name,i)=>
    text(s,cols[i][0],178,cols[i][1],name,{size:14,style:'bold',align:i?'center':'left',color:C.muted}));
  end.forEach(([seed,row],i)=>{
    const y=216+i*40;
    line(s,358,y-6,898,y-6);
    const [lo,hi]=row.test.interval.map(v=>pct(v));
    const units=[seed,pct(row.arms.control.p_dog)+'%',pct(row.arms.treatment.p_dog)+'%',
      '+'+pct(row.test.difference),'['+lo+', '+hi+']'];
    units.forEach((value,j)=>text(s,cols[j][0],y,cols[j][1],value,
      {size:j===4?12:16,style:j===3?'bold':'regular',align:j?'center':'left',color:j===3?C.teal:C.ink}));
  });
  line(s,358,294,898,294);
  text(s,358,310,532,'Frozen base: '+baseDog+'%  ·  Reported metric: next-token probability',{size:14,color:C.muted});
  text(s,358,351,531,'Resampling: +'+pct(resampled.test.difference)+' pp in one seed',{size:19,style:'bold',color:C.ink});
  text(s,358,385,529,'The resampled control also lost more language quality.',{size:14,color:C.muted});
  text(s,358,425,532,'Intervals resample prompt frames within each fitted student pair.',{size:13,color:C.muted});
  text(s,45,475,870,'The unrelated-prompt lifts were +0.007 and +0.013 pp across the two seeds.',{size:15,style:'bold',color:C.teal});

  notes(s,'4:50–5:50 | 60 seconds',[
    'On fifty new favorite-animal prompts, we measure the probability of the single dog token. The frozen base is at 0.783 percent. Treatment exceeds the matched control by 0.456 and 0.925 percentage points across the two seeds. The intervals describe variation across prompt frames, and both seeds use the same teacher and dataset.',
    'Dog probability also rises on twenty unrelated prompts by much smaller amounts. Independent number draws left a 0.114-point contrast in one seed. The resampled control degraded more on the small language check, so that contrast is hard to assign to frequency alone.',
    'The main result is a small increase in dog-token probability under this setup. We next ask whether pixels can carry a specific change.'
  ],[S.number,S.numberData,S.promptBank,S.tang],[
    'End of epoch 3 was fixed in advance. Grouped paired bootstrap used ten prompt frames. The protocol did not measure sampled answers. Both main arms lost moderate language quality. B4 and B5 provide the checks.',
  ]);
}

recorded(7);
recorded(8);
recorded(9);

// 10 · Close on the difference between a specific target and a broad output shift.
{
  const s=header('A changed output can be broader than the trait','WHAT THE EXPERIMENTS SHOW','8:50–9:40',
    'Measured outcomes  /  Three tasks, three forms of student supervision',{dark:true});
  const summary=[
    ['FULL TOKEN DISTRIBUTIONS','Part of a digit rule',['48.05% and 64.19%','exact match on shifted sums.']],
    ['SAMPLED NUMBER TOKENS','A small dog-token lift',['+0.456 and +0.925 pp','on favorite-animal prompts.']],
    ['GENERATED DOG PIXELS','A broad color shift',['Trucks became bluer.','No detectable blue-fur cat transfer.']],
  ];
  summary.forEach(([source,title,detail],i)=>{
    const x=45+i*294;
    rect(s,x,128,282,176,C.panel);
    label(s,x+13,142,254,source,true);
    text(s,x+13,171,252,title,{size:20,style:'bold',color:C.white});
    paragraph(s,x+13,219,255,detail,{size:15,step:23,color:C.dim});
  });
  line(s,45,334,915,334,'45636A');
  text(s,45,355,870,'Specificity is an empirical question.',{size:28,style:'heavy',color:C.white});
  paragraph(s,45,404,870,['On what other prompts does the token rise?','On what other image classes does the color change?'],
    {size:18,step:28,color:C.dim});
  notes(s,'8:50–9:40 | 50 seconds. Leave 9:40–10:00 open.',[
    'What crossed over depends on what we measure. Whole token distributions carried part of a digit rule that could be learned through the shared readout. Sampled numbers produced a small dog-token lift. Dog images produced a broad color shift, with no detectable blue-fur cat transfer.',
    'The key control asks whether the effect appears outside the trait context. Dog probability rose on unrelated prompts, and the image effect reached trucks. These results support describing exactly which outputs moved. A causal comparison of the three forms of supervision would hold task and checkpoint fixed. Thank you.'
  ],[S.addition,S.number,S.numberData,S.results,S.dogAudit,S.imageAudit]);
}

recorded(11);
recorded(12);
recorded(13);

// B4–B6 · Keep uncertainty, specificity checks, and per-seed evidence available for questions.
{
  const s=header('How much survived independent resampling?','BACKUP  /  NUMBER ABLATION','Q&A',
     'Measured number study  /  One student-training seed (6101)');
  rect(s,45,125,870,126,C.white);
  label(s,61,140,840,'CONSTRUCTION');
  paragraph(s,61,166,835,[
    'Reuse the same prompts, separators and number counts in both arms.',
    'Draw each number independently from its arm’s empirical first-number frequencies.',
  ],{size:19,step:31});
  rect(s,45,272,425,150,C.pale);
  rect(s,491,272,424,150,C.white);
  label(s,61,286,385,'ORIGINAL TEACHER SEQUENCES');
  label(s,507,286,385,'INDEPENDENT NUMBER DRAWS');
  text(s,61,319,387,'+0.456 pp',{size:34,style:'heavy',color:C.teal});
  text(s,507,319,386,'+0.114 pp',{size:34,style:'heavy',color:C.teal});
  text(s,61,373,389,'95% frame interval [0.221, 0.711]',{size:15,color:C.muted});
  text(s,507,373,388,'95% frame interval [0.059, 0.179]',{size:15,color:C.muted});
  text(s,45,454,870,'Resampling reduced both arms below the base. The control lost more language quality.',{size:16,style:'bold',color:C.muted});
  notes(s,'Backup | Outside the timed talk',[
    'The remaining contrast is about a quarter of the original-sequence effect for this seed. Both absolute dog probabilities fell below the frozen base. Language NLL above base was 2.993 in the resampled control and 0.841 in the resampled treatment. Unequal degradation limits attribution to first-number frequencies.'
  ],[S.number,S.numberData,S.tang]);
}
{
  const s=header('Where else did the dog token move?','BACKUP  /  NUMBER SPECIFICITY','Q&A',
    'Measured number study  /  Fixed third epoch');
  rect(s,45,123,870,274,C.white);
  const cols=[[59,250],[314,175],[498,174],[679,218]];
  ['Model','Favorite animal (%)','Unrelated (%)','Language NLL over base'].forEach((v,i)=>
    text(s,cols[i][0],145,cols[i][1],v,{size:15,style:'bold',align:i?'right':'left'}));
  const rows=[['Frozen base',baseDog,pct(evidence.number.base.negative.probability,4),'0.000']];
  for (const [seed,row] of end) {
    for (const key of ['control','treatment']) {
      const arm=row.arms[key];
      rows.push([seed+' '+key,pct(arm.p_dog),pct(arm.p_dog_negative,4),arm.language_excess_nll.toFixed(3)]);
    }
  }
  rows.forEach((row,i)=>{
    const y=186+i*39;
    line(s,60,y-5,897,y-5);
    row.forEach((v,j)=>text(s,cols[j][0],y+3,cols[j][1],v,
      {size:16,style:j===0&&i>0&&i%2===0?'bold':'regular',align:j?'right':'left'}));
  });
  paragraph(s,45,417,866,[
    '50 favorite-animal prompts, 20 unrelated prompts and 16 short quality passages.',
    'Teacher validation: 84.04% on favorite-animal prompts, versus 0.503% for the frozen base.',
  ],{size:15,step:27,color:C.muted});
  notes(s,'Backup | Outside the timed talk',[
    'Student results are paired at the fixed endpoint. The teacher validation comparison uses a disjoint bank from the final student test. The exploratory validation gates select the teacher. Each seed is conditional on that same teacher and dataset.'
  ],[S.number,S.numberData]);
}
{
  const s=header('Per-seed addition scores and image uncertainty','BACKUP  /  EVALUATION UNCERTAINTY','Q&A',
    'Corrected addition stability study and final Piccolo cat / truck evaluation');
  rect(s,45,125,409,301,C.white);
  label(s,60,141,378,'SHIFTED ADDITION  /  THREE SEED GROUPS');
  ['Seed group','+1 exact match','−1 exact match'].forEach((v,i)=>
    text(s,[60,200,320][i],181,[134,114,117][i],v,{size:13,style:'bold',align:i?'right':'left'}));
  add['+1_seeds'].forEach((n,i)=>{
    const y=226+i*49;
    line(s,60,y-8,437,y-8);
    [String(i+1),n.toFixed(2)+'%',add['−1_seeds'][i].toFixed(2)+'%'].forEach((cell,j)=>
      text(s,[60,200,320][j],y,[134,114,117][j],cell,{size:17,align:j?'right':'left'}));
  });
  rect(s,474,125,441,301,C.dark);
  label(s,490,141,406,'PICCOLO  /  ONE TRAINING SEED AND ONE AI RATER',true);
  text(s,490,183,408,'Recognizable blue-fur cats',{size:18,style:'bold',color:C.white});
  text(s,490,218,409,'+2.08 pp  [−8.70, +12.57]',{size:20,style:'bold',color:C.mint});
  text(s,490,272,408,'Truck blue excess',{size:18,style:'bold',color:C.white});
  text(s,490,308,409,'+9.19 units  [+8.71, +9.69]',{size:20,style:'bold',color:C.mint});
  text(s,490,373,407,'95% paired intervals across 48 generated seeds',{size:14,color:C.dim});
  text(s,45,455,870,'Image-sample intervals describe these fitted checkpoints.',{size:15,style:'bold',color:C.muted});
  notes(s,'Backup | Outside the timed talk',[
    'Addition uses three base seed groups. The image study has one fitted teacher and student training seed per arm. Codex scored the randomized outputs as the single blinded rater. The interval for blue-fur cats is the conservative paired interval for one discordant seed. The truck interval uses a paired seed bootstrap.'
  ],[S.addition,S.additionData,S.results,S.imageAudit]);
}

// Overlaps from a highlight contained inside its panel or a card containing its
// text are intentional. Lines serve as dividers and arrow connectors.
for (const s of pptx._slides) {
  warnIfSlideHasOverlaps(s,pptx);
  warnIfSlideElementsOutOfBounds(s,pptx);
}

pptx.writeFile({fileName:process.argv[2] || path.join(__dirname,'subliminal-learning-seminar.pptx')})
  .catch(error=>{ console.error(error); process.exitCode=1; });
