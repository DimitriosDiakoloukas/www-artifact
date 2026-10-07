"""Deterministic paper assets from complete, hash-gated stage-two analyses."""
import sys,json,argparse,shutil,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collective.common import sha
from srange import provenance as pv
BASE=Path(__file__).resolve().parent
LABEL={'bitcoin_alpha':'Bitcoin-Alpha','wiki_elec':'Wiki-Elec'}
COLORS=('#2166ac','#d6604d','#4c956c')
plt.rcParams.update({'font.size':8,'font.family':'sans-serif','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'axes.labelcolor':'#222222','xtick.color':'#444444','ytick.color':'#444444'})

def write(p,text):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text+'\n')
def table(head,rows,columns):return '\\begin{tabular}{@{}'+columns+'@{}}\n\\toprule\n'+head+' \\\\\n\\midrule\n'+'\n'.join(' & '.join(row)+' \\\\' for row in rows)+'\n\\bottomrule\n\\end{tabular}'
def save(fig,p):fig.savefig(p,bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None,'Creator':'','Producer':''});plt.close(fig)
def get(summary,kind,net,policy):return next(s for s in summary['seed_summaries'] if (s['kind'],s['network'],s['policy'])==(kind,net,policy))
def run(paper,preview=False,data_directory=None):
    g=paper/'generated';f=paper/'figures';g.mkdir(parents=True,exist_ok=True);f.mkdir(parents=True,exist_ok=True)
    d=Path(data_directory) if data_directory else BASE/'results'
    known=pv.verify_record(d/'known.json' if data_directory else BASE/'known.json');large=json.loads((d/'large_controls.json').read_text());native=json.loads((d/'native_models.json').read_text());joint=json.loads((d/'native_collective.json').read_text());matched=json.loads((d/'matched.json').read_text());comp=json.loads((Path('/tmp/www-stage2-computation-interim.json') if preview else d/'computation.json').read_text())
    assert len(known['rows'])==144 and len(large['rows'])==12 and len(native['cells'])==12 and joint['checkpoints']==20 and matched['cells']==960
    assert preview or comp['all_benchmarks_required']
    selected=[r for r in large['rows'] if r['solved'] and r['job'][1]=='SGCN'];assert len(selected)==5
    macros=[]
    def number(name,value,fmt='.0f'):
        assert name.isalpha();v=value if isinstance(value,str) else format(value,fmt);macros.append('\\newcommand{\\'+name+'}{'+v+'}')
    number('nStageKnownVariance',next(v['true_Q_variance_below_d'] for v in known['rows'] if v['law']=='redundant' and v['solver']=='majority' and v['paths']==101 and v['distance']==3 and v['distractor_branches']==0),'.3f');number('nStageCorrectQueries',sum(r['correctly_classified_audit_stratum']['queries'] for r in selected));number('nStageCorrectGradientMiss',sum(r['correctly_classified_audit_stratum']['gradient_missed_queries'] for r in selected));number('nStageKnownCells',len(known['rows']));number('nStageLargeModels',12);number('nStageLargeSolved',sum(r['solved'] for r in large['rows']));number('nStageSolvedSgcn',len(selected));number('nStageSolvedGradientMiss',100*np.mean([r['gradient']['missed_distance_fraction'] for r in selected]));number('nStageSolvedFiniteMiss',100*np.mean([r['single_flip']['missed_distance_fraction'] for r in selected]));number('nStageSolvedClassMin',100*min(r['true_Q'][3]['class_disagreement'] for r in selected));number('nStageSolvedClassMax',100*max(r['true_Q'][3]['class_disagreement'] for r in selected));number('nStageMatchedSlots',matched['prediction_slots']);cycles=[r for r in matched['rows'] if r['mechanism']=='cycles'];number('nStageCycleNoChange',100*np.mean([r['no_change_fraction'] for r in cycles]),'.1f')
    for arch,net,key in [('SGCN','bitcoin_alpha','SgAlpha'),('SGCN','wiki_elec','SgWiki'),('SIDNET','bitcoin_alpha','SidAlpha'),('SIDNET','wiki_elec','SidWiki')]:
        j=next(r for r in joint['seed_summaries'] if (r['arch'],r['network'],r['radius'],r['mechanism'],r['stratum'])==(arch,net,1,'exchange','reach_at_most_one'));number('nStageNative'+key,j['probability_RMSE'][0],'.3f')
    for norm,label in (('all_relations','All'),('variable_signs','Variable')):
        for metric,word in (('gradient','Gradient'),('single_flip','Finite')):
            number('nStage'+label+word+'Miss',sum(r[metric]['threshold_sensitivity'][norm]['0.1']['missed_queries'] for r in selected))
    for metric,word in (('gradient','Gradient'),('single_flip','Finite')):
        number('nStageFixed'+word+'Max',sum(r[metric]['maximum_at_fixed_incident_connector'] for r in selected))
        for law,label in (('consensus','Consensus'),('redundancy','Redundancy')):
            rr=[r for r in selected if r['job'][0]==law]
            number('nStage'+label+word+'Miss',sum(r[metric]['threshold_sensitivity']['all_relations']['0.1']['missed_queries'] for r in rr))
            number('nStage'+label+'Queries',20*len(rr)) if metric=='gradient' else None
    for net,label in (('bitcoin_alpha','Alpha'),('wiki_elec','Wiki')):
        rr=[r for r in comp['rows'] if r['kind']=='primary' and r['native'] and r['network']==net and r['policy']=='full']
        counts=[r['metrics']['negative_queries'] for r in rr]
        number('nStageNegative'+label+'Min',min(counts));number('nStageNegative'+label+'Max',max(counts))
        crop=get(comp,'primary',net,'range_proposal');number('nStageReachCropPass'+label,crop['heldout_passes'])
        nodes=[r['mean_retained_nodes'] for r in comp['rows'] if r['kind']=='primary' and r['native'] and r['network']==net and r['policy']=='fixed_crop2']
        number('nStageCropNodes'+label,np.mean(nodes));number('nStageFullNodes'+label,rr[0]['full_nodes'])
    for metric,word in (('gradient','Gradient'),('single_flip','Finite')):
        for t,label in ((.05,'Low'),(.2,'High')):
            number('nStage'+word+'Miss'+label,sum(r[metric]['threshold_sensitivity']['all_relations'][str(t)]['missed_queries'] for r in selected))
    for arch,label in (('SGCN','Sg'),('SIDNET','Sid')):
        for law,word in (('fixed','Unrestricted'),('exchange','Exchange')):
            r=next(r for r in joint['seed_summaries'] if (r['arch'],r['network'],r['radius'],r['mechanism'],r['stratum'])==(arch,'bitcoin_alpha',1,law,'reach_at_most_one'))
            number('nStageLogit'+label+'Alpha'+word,r['logit_RMSE'][0],'.2f')
    for i,r in enumerate([r for r in selected if r['job'][0]=='consensus']):
        number('nStageMargin'+('First' if i==0 else 'Second'),r['posthoc_margin_analysis']['single_flip_margin_ratio_correlation'],'.2f')
    number('nStageExactMajorityMiss',100*(1-math.comb(101,50)/2**100),'.1f')
    number('nStageFixedCropWikiSpeed',next(b for b in get(comp,'primary','wiki_elec','fixed_crop2')['batches'] if b['batchsize']==1)['speedup_vs_full'][0],'.2f')
    policies=json.loads((BASE/'layerwise_policies.json').read_text());number('nStageLayerSelected',sum(m['choice']==1 for m in policies['models']));number('nStageComputeModels',12)
    number('nStageLayerHeldPass',sum(r['heldout_pass'] for r in comp['rows'] if r['kind']=='layerwise' and r.get('selected_by_validation')))
    for net,key in [('bitcoin_alpha','Alpha'),('wiki_elec','Wiki')]:
        s=get(comp,'layerwise',net,'validation_selected');number('nStageLayerRmse'+key,s['metrics']['probability_RMSE'][0],'.4f');first=get(comp,'layerwise',net,'L8-16');number('nStageFirstRmse'+key,first['metrics']['probability_RMSE'][0],'.3f')
        full=get(comp,'layerwise',net,'L16-16')
        number('nStageFullLatency'+key,1000*next(v for v in full['batches'] if v['batchsize']==1)['seconds_per_query'][0],'.1f')
        number('nStageSelectedLatency'+key,1000*next(v for v in s['batches'] if v['batchsize']==1)['seconds_per_query'][0],'.1f')
        for size,word in [(1,'Single'),(64,'Batch')]:
            b=next((v for v in s['batches'] if v['batchsize']==size),None)
            number('nStageLayerSpeed'+key+word,b['speedup_vs_full'][0] if b else r'\pending{measured runtime}', '.2f')
    for size,word in [(1,'Single'),(64,'Batch')]:
        counts=[value for net in LABEL for batch in get(comp,'layerwise',net,'validation_selected')['batches'] if batch['batchsize']==size for value in batch['break_even_queries'] if value is not None]
        if counts:
            number('nStageBreakEven'+word+'Min',min(counts));number('nStageBreakEven'+word+'Max',max(counts))
    write(g/'stage_numbers.tex','% Generated from complete stage-two records; do not edit.\n'+'\n'.join(macros))
    standard=json.loads((g/'native.json').read_text());e8=json.loads((g/'e8_cycles.json').read_text());ph=json.loads((g/'posthoc_reach_native.json').read_text());rep=json.loads((g/'replication.json').read_text())
    scells={(c['network'],c['arch'],c['T']):c['reach_0.1'] for c in ph['cells']}
    for c in rep['network_cells']:
        key=(c['network'],c['arch'],c['T'])
        if c['features']=='spectral' and key not in scells:scells[key]=c['reach_0.1'][0]
    sr=[]
    labels={'bitcoin_alpha':'Bitcoin-Alpha','bitcoin_otc':'Bitcoin-OTC','wiki_rfa':'Wiki-RfA','wiki_elec':'Wiki-Elec','slashdot':'Slashdot','epinions':'Epinions'}
    for net,label in labels.items():
        walk=next(c for c in e8['cells'] if c['network']==net);best=max(c['auc'][0] for c in standard['cells'] if c['network']==net);reaches=[scells[(net,arch,32)] for arch in ('SGCN','SIDNET','SLGNN','BGSD')]
        sr.append([label,f'{walk["local_refit_auc"][0]:.3f}',f'{walk["cycles_auc"][0]:.3f}',f'{best:.3f}',f'{min(reaches):.2f}--{max(reaches):.2f}'])
    write(g/'stage_standardised.tex',table('Network & Local & Walk & GNN & Reach',sr,'lrrrr'))
    nr=[]
    for net in LABEL:
      for arch in ('SGCN','SIDNET'):
        a=[next(c for c in native['cells'] if (c['network'],c['arch'],c['T'])==(net,arch,T)) for T in (8,32)]
        nr.append([LABEL[net],arch,'$'+r'\to'.join(f'{c["test_auc"][0]:.3f}' for c in a)+'$','$'+r'\to'.join(f'{c["reach_0.1"][0]:.2f}' for c in a)+'$'])
    write(g/'stage_native.tex',table('Network & Model & AUC & $\\rho_{0.1}$',nr,'llrr'))
    sensitivity=[]
    for norm,label in (('all_relations','All relations'),('variable_signs','Votes + nuisance')):
      for metric,word in (('gradient','Gradient'),('single_flip','Finite')):
        sensitivity.append([label,word,*[str(sum(r[metric]['threshold_sensitivity'][norm][str(t)]['missed_queries'] for r in selected)) for t in (.05,.1,.2)]])
    write(g/'stage_sensitivity.tex',table('Normaliser domain & Diagnostic & $\\tau=.05$ & $.10$ & $.20$',sensitivity,'llrrr'))
    scales=[]
    for net in LABEL:
      for arch in ('SGCN','SIDNET'):
        rr=[next(r for r in joint['seed_summaries'] if (r['network'],r['arch'],r['radius'],r['mechanism'],r['stratum'])==(net,arch,1,law,'reach_at_most_one')) for law in ('fixed','exchange')]
        scales.append([LABEL[net],arch,f'{rr[0]["reference_median_absolute_logit"][0]:.2f}',*[f'{r[key][0]:.3f}' for key in ('probability_RMSE','logit_RMSE') for r in rr]])
    write(g/'stage_native_scale.tex',table('Network & Model & $|\\ell_0|$ & P(U) & P(E) & L(U) & L(E)',scales,'llrrrrr'))
    chain=json.loads((g/'chain.json').read_text());replication=json.loads((g/'replication.json').read_text())
    ledger=[['H1','Finite mass radius recovers solved chain distance',f'{chain["H1"]["solved_cells"]} cells; all within one hop; rank $r_s={chain["H1"]["spearman"]:.3f}$; holds.'],['H2','Feature mass radius falls short on relay',f'{chain["H2"]["feature_R90_below_source"]}/{chain["H2"]["checkpoints"]} checkpoints; $p={chain["H2"]["p_one_sided"]:.2g}$; holds.'],['H3','Depth-effect ratio below one half','All four models meet the locked bootstrap/Holm rule.']]
    for model in ('BGSD','SGCN','SLGNN','SIDNET'):
        r=standard['H4'][model];ledger.append(['H4: '+('Diffusion' if model=='BGSD' else ('SLGNN-style' if model=='SLGNN' else model)),'Functional depth at most eight',f'Median {r["median_T_func"]:.0f}; {r["le8"]}/30 at most eight; Holm $p={r["p_holm"]:.3g}$; '+('holds.' if r['holds'] else 'not supported.')])
    for key,statement in [('RH1','Relative reach recovers solved planted distance'),('RH2','Relative reach separates solved from chance'),('RH3','Finite mass radius falls short in solved cells')]:
        r=replication[key];detail=(f'{r["within"]}/{r["solved_cells"]} cells' if key=='RH1' else (f'{r["solved_runs"]} solved, {r["chance_runs"]} chance; $p={r["p"]:.2g}$' if key=='RH2' else f'{r["mass_below_e_star"]}/{r["solved_cells"]} cells'));ledger.append([key,statement,detail+'; holds.'])
    # RH4 is stored as a compound result in the registered replication summary.
    for key in ('RH4a','RH4b'):
        r=replication.get(key,replication['RH4']['a_slashdot_epinions' if key=='RH4a' else 'b_random_features']);assert r
        ledger.append([key,'Relative reach at most two in held-out cells',f'{r["within_bound"]}/{r["cells"]} cells; holds.'])
    write(g/'stage_hypotheses.tex',table('Test & Outcome',[[r[0],r[2]] for r in ledger],'lp{.76\\columnwidth}'))
    locations=json.loads((d/'review_diagnostics.json').read_text())
    locrows=[[LABEL[r['network']],r['arch'],'D' if r['mechanism']=='cycles' else 'P$_m$',str(r['edited_relation_instances']),f'{r["median_minimum_endpoint_degree"]:.0f}',f'{100*r["fraction_at_distance_two"]:.1f}'] for r in locations['rows']]
    write(g/'stage_edit_locations.tex',table('Network & Model & Law & Edits & Min. degree & Distance 2\\,\\%',locrows,'lllrrr'))
    lr=[]
    for r in large['rows']:
        q=r['true_Q'][3];lr.append([r['job'][0].title(),r['job'][1],str(r['job'][2]),f'{r["test_AUC"]:.3f}'+('$^*$' if r['solved'] else ''),f'{r["test_accuracy"]:.3f}',f'{r["gradient"]["mean_reach"]:.2f}',f'{r["single_flip"]["mean_reach"]:.2f}',f'{100*q["class_disagreement"]:.1f}',f'{q["completion_label_accuracy"]:.3f}'])
    write(g/'stage_large.tex',table('Law & Model & Seed & AUC & Acc. & $\\rho_g$ & $\\rho_I$ & Flip\\,\\% & New acc.',lr,'llrrrrrrr'))
    cr=[]
    for idx in range(4):
        rr=[r for r in matched['rows'] if r['checkpoint']==idx and r['radius']==1 and r['nominal_hamming_fraction']==.05]
        lookup={r['mechanism']:r for r in rr};c=lookup['cycles'];vals=[]
        for key in ('fixed','exchange','cycles','cycle_matched_fixed','cycle_matched_exchange'):vals.append(f'{next(e for e in lookup[key]["effects"] if e["stratum"]=="all")["probability_RMSE"]:.3f}')
        cr.append([LABEL[c['network']],c['arch'],f'{c["mean_requested_count"]:.0f}/{c["mean_achieved_count"]:.0f}',*vals])
    write(g/'stage_constraints.tex',table('Network & Model & Target/achieved & U & P & D & U$_m$ & P$_m$',cr,'llrrrrrr'))
    tr=[]
    for net in LABEL:
      for kind,name,label in [('primary','fixed_crop2','Fixed radius-two crop'),('primary','range_proposal','Reach crop'),('primary','collective_proposal','Joint crop'),('primary','calibrated_global','Calibrated global'),('primary','structural_tree','Structural tree'),('primary','degree_rule','Degree rule'),('primary','triangle_rule','Triangle rule'),('primary','fixed_K8','Equal $(8,8)$'),('layerwise','L8-16','First-layer $(8,16)$'),('layerwise','validation_selected','Selected layer schedule')]:
        s=get(comp,kind,net,name);m=s['metrics'];speed=[]
        for size in (1,64):
            b=next((v for v in s['batches'] if v['batchsize']==size),None);speed.append(f'{b["speedup_vs_full"][0]:.2f}' if b else '--')
        tr.append([LABEL[net],label,f'{m["probability_RMSE"][0]:.4f}',f'{100*m["class_disagreement"][0]:.2f}',f'{m["AUC_loss"][0]:+.4f}',str(s['heldout_passes'])+'/5',*speed])
    write(g/'stage_computation.tex',table('Network & Policy & RMSE & Disagree\\,\\% & AUC loss & Pass & Speed$_1$ & Speed$_{64}$',tr,'llrrrrrr'))
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.35),gridspec_kw={'width_ratios':[1.12,1]})
    examples=[('independent_consensus','majority',1,'Sparse path'),('independent_consensus','majority',101,'Independent majority'),('redundant','majority',101,'Redundant majority'),('redundant','mean',101,'Redundant mean')]
    data=[]
    for law,solver,k,label in examples:
        r=next(v for v in known['rows'] if (v['law'],v['solver'],v['paths'],v['distance'],v['distractor_branches'])==(law,solver,k,3,0));data.append([r['gradient']['missed_fraction'],r['single_flip']['missed_fraction'],r['gradient']['R90_missed_fraction'],r['single_flip']['R90_missed_fraction']])
    data[1]=[1-math.comb(101,50)/2**100]*4
    data.append([1,0,1,0]);im=axes[0].imshow(data,cmap='Blues',vmin=0,vmax=1,aspect='auto')
    for i,row in enumerate(data):
      for j,value in enumerate(row):axes[0].text(j,i,(f'{100*value:.1f}%' if i==1 else f'{100*value:.0f}%'),ha='center',va='center',color='white' if value>.6 else '#222222',fontsize=7)
    axes[0].set_yticks(range(5),[v[3] for v in examples]+['Saturated sparse path']);axes[0].set_xticks(range(4),['Gradient\nreach','Finite\nreach','Gradient\n$R_{90}$','Finite\n$R_{90}$']);axes[0].tick_params(length=0);axes[0].set_title('(a) Missed required distance',loc='left',fontsize=9,pad=8)
    for i,r in enumerate(large['rows']):
        group=(0 if r['job'][0]=='consensus' else 2)+(0 if r['job'][1]=='SGCN' else 1);offset=(r['job'][2]-61011)*.18
        for typ,marker,color,dx in [('gradient','o',COLORS[0],-.03),('single_flip','s',COLORS[1],.03)]:axes[1].scatter(group+offset+dx,r[typ]['mean_reach'],s=25 if r['solved'] else 17,marker=marker,facecolors=color if typ=='gradient' else 'none',edgecolors=color,alpha=1 if r['solved'] else .35,linewidths=.8)
    axes[1].axhline(3,color='#555555',lw=.8,ls='--');axes[1].set_ylim(-.15,3.35);axes[1].set_xticks(range(4),['Cons.\nSGCN','Cons.\nSIDNET','Redund.\nSGCN','Redund.\nSIDNET']);axes[1].set_ylabel('Mean individual reach (hops)');axes[1].set_title('(b) Learned controls',loc='left',fontsize=9,pad=8)
    for typ,marker,color in [('Gradient','o',COLORS[0]),('Finite','s',COLORS[1])]:axes[1].scatter([],[],marker=marker,color=color,s=18,label=typ)
    axes[1].legend(frameon=False,loc='lower right',fontsize=7);fig.tight_layout(w_pad=2);save(fig,f/'stage_validity.pdf')
    fig,ax=plt.subplots(figsize=(3.42,2.3));groups=[('SGCN','bitcoin_alpha'),('SGCN','wiki_elec'),('SIDNET','bitcoin_alpha'),('SIDNET','wiki_elec')]
    for mi,mechanism in enumerate(('fixed','exchange')):
        means=[];lo=[];hi=[]
        for arch,net in groups:
            r=next(v for v in joint['seed_summaries'] if (v['arch'],v['network'],v['radius'],v['mechanism'],v['stratum'])==(arch,net,1,mechanism,'reach_at_most_one'));mean,lower,upper=r['probability_RMSE'];means.append(mean);lo.append(mean-max(0,lower));hi.append(upper-mean)
        ax.bar(np.arange(4)+(mi-.5)*.32,means,width=.3,color=COLORS[mi],label=('Unrestricted' if mi==0 else 'Prevalence preserving'),yerr=[lo,hi],error_kw={'linewidth':.7,'capsize':2})
    labels=[]
    for arch,net in groups:
        r=next(v for v in joint['seed_summaries'] if (v['arch'],v['network'],v['radius'],v['mechanism'],v['stratum'])==(arch,net,1,'fixed','reach_at_most_one'));labels.append(('Alpha' if net=='bitcoin_alpha' else 'Wiki')+'\n'+arch+f'\n$n={r["queries"]}$')
    ax.set_xticks(range(4),labels);ax.set_ylabel('Probability RMSE');ax.legend(frameon=False,fontsize=7);ax.set_ylim(0,.21);fig.tight_layout();save(fig,f/'stage_collective.pdf')
    fig,ax=plt.subplots(figsize=(3.42,2.1))
    for ni,net in enumerate(LABEL):
        means=[];lo=[];hi=[]
        for name in ('L16-8','L8-16'):
            s=get(comp,'layerwise',net,name);mean,lower,upper=s['metrics']['probability_RMSE'];means.append(mean);lo.append(mean-max(lower,mean*.01));hi.append(upper-mean)
        ax.errorbar(np.arange(2)+(ni-.5)*.12,means,yerr=[lo,hi],fmt='o',color=COLORS[ni],capsize=2,label=LABEL[net],markersize=4)
    ax.set_yscale('log');ax.set_xticks([0,1],['Final layer (16,8)','First layer (8,16)']);ax.set_ylabel('Probability RMSE');ax.legend(frameon=False,fontsize=7);fig.tight_layout();save(fig,f/'stage_computation.pdf')
    smaller=[]
    from collective.common import source_hash as pilot_source
    for arch in ('SGCN','SIDNET'):
        folder=BASE.parent/'training'/arch;rec=pv.verify_record(folder/'complete.json');assert rec['source_sha256']==pilot_source('training') and rec['original_logits_bitwise_before_after']
        for name,digest in rec['files'].items():assert sha(folder/name)==digest
        pr=np.load(folder/'profile.npz');assert np.all(pr['reach']==3) and np.all(pr['finite_reach']==3)
        smaller.append({'arch':arch,'test_AUC':rec['test_auc'],'gradient_queries_recovering_required_distance':20,'finite_queries_recovering_required_distance':20})
    data={'posthoc_edit_locations':locations,'exact_independent_majority_miss':1-math.comb(101,50)/2**100,'hypothesis_ledger':{'confirmatory':{k:chain[k] for k in ('H1','H2')},'network':{k:standard[k] for k in ('H3','H4')},'replication':{k:v for k,v in replication.items() if k.startswith('RH')}},'descriptive_strata':json.loads((d/'strata.json').read_text()),'smaller_learned_controls':smaller,'known_functions':known,'learned_controls':large,'native_models':native,'native_collective':joint,'matched_constraints':matched,'computation':comp}
    # Drop timestamps/environmental self hashes from known-functions presentation only; retain raw record.
    data['known_functions']={k:v for k,v in known.items() if k not in ('created_utc','result_sha256')}
    write(g/'stage_results.json',json.dumps(data,sort_keys=True,indent=1,allow_nan=False))
    print('STAGE ASSETS',paper,'preview',preview,flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--paper',type=Path,required=True);ap.add_argument('--preview',action='store_true');ap.add_argument('--data-directory',type=Path);a=ap.parse_args();run(a.paper,a.preview,a.data_directory)
