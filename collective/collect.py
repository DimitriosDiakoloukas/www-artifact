"""Refuse incomplete pilots; regenerate transparent summaries and standalone figures."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import argparse
import json
import numpy as np
from collective.common import BASE,RADII,MECHANISMS,STRENGTHS,lock,sha,source_hash,sigmoid,paired_variance,variance_interval
from srange import provenance as pv
from srange.train import auc


def safe_auc(y,score):
    value=auc(y,score)
    return float(value) if np.isfinite(value) else None


def effects(logits,original,labels):
    p=sigmoid(logits);base=sigmoid(original)
    difference=p-base[:,None];ld=logits-original[:,None]
    au=[safe_auc(labels,logits[:,j]) for j in range(logits.shape[1])]
    empirical_variance=float(np.mean(np.var(p,axis=1,ddof=0)))
    squared_mean_drift=float(np.mean((p.mean(axis=1)-base)**2))
    assert np.isclose(empirical_variance+squared_mean_drift,np.mean(difference**2))
    return {'empirical_completion_variance':empirical_variance,
            'empirical_squared_mean_drift':squared_mean_drift,
            'probability_MAE':float(np.mean(abs(difference))),'probability_RMSE':float(np.sqrt(np.mean(difference**2))),
            'probability_max_absolute_change':float(abs(difference).max()),
            'logit_MAE':float(np.mean(abs(ld))),'logit_RMSE':float(np.sqrt(np.mean(ld**2))),
            'disagreement_fraction':float(np.mean((logits>=0)!=(original[:,None]>=0))),
            'mean_intervened_original_label_AUC':None if any(x is None for x in au) else float(np.mean(au)),
            'mean_intervened_original_label_Brier':float(np.mean((p-labels[:,None])**2))}


def collect_networks():
    rows=[];inputs={}
    targets=json.loads((BASE/'targets.json').read_text())
    for i,target in enumerate(targets):
        rec=pv.verify_record(ROOT/target['record'])
        assert rec['result_sha256']==target['result_sha256']
        dest=BASE/'runs'/f"{i}-{rec['args']['arch']}-{rec['args']['dataset']}"
        done=pv.verify_record(dest/'complete.json')
        assert done['source_sha256']==source_hash() and done['original_test_logits_bitwise_before_after']
        assert done['lock_sha256']==sha(BASE/'LOCK.json')
        assert done['checkpoint_sha256']==target['checkpoint_sha256']
        inputs[str((dest/'complete.json').relative_to(ROOT))]=sha(dest/'complete.json')
        for name,digest in done['files'].items():assert sha(dest/name)==digest
        pr=np.load(dest/'profile.npz');original=pr['original_logits'];labels=pr['labels'];reach=pr['reach']
        assert len(labels)==20
        assert np.array_equal(pr['slots'],np.sort(np.random.default_rng(20261006).choice(len(rec['targets']['test_edge_positions']),20,replace=False)))
        cells=[]
        for r in RADII:
            for mechanism in MECHANISMS:
                for strength in STRENGTHS:
                    data=[np.load(dest/f'q{q:02d}-r{r}-{mechanism}-s{int(strength*100):02d}.npz') for q in range(20)]
                    logits=np.stack([d['logits'] for d in data]);assert logits.shape==(20,16)
                    hamming=np.stack([d['hamming'] for d in data]);eligible=np.array([int(d['eligible_count']) for d in data])
                    fraction=np.divide(hamming,eligible[:,None],out=np.zeros_like(hamming,dtype=float),where=eligible[:,None]>0)
                    terms=paired_variance(sigmoid(logits));subset=np.isfinite(reach)&(reach<=r)
                    strata={}
                    groups={'no_common_neighbour':pr['common_neighbours']==0,'has_common_neighbour':pr['common_neighbours']>0,
                            'minimum_degree_at_or_below_sample_median':pr['endpoint_min_degree']<=np.median(pr['endpoint_min_degree']),
                            'minimum_degree_above_sample_median':pr['endpoint_min_degree']>np.median(pr['endpoint_min_degree'])}
                    for name,mask in groups.items():
                        strata[name]={'queries':int(mask.sum()),'effects':effects(logits[mask],original[mask],labels[mask]) if mask.any() else None}
                    cells.append({'radius':r,'mechanism':mechanism,'strength':strength,'queries':20,
                                  'mean_eligible_relations':float(eligible.mean()),'mean_realised_Hamming_fraction':float(fraction.mean()),
                                  'no_change_draws':int((hamming==0).sum()),'effects':effects(logits,original,labels),
                                  'pair_variance_simultaneous_95':variance_interval(terms,family=80),
                                  'pair_variance_pointwise_95':variance_interval(terms),
                                  'individual_reach_within_radius':{'queries':int(subset.sum()),
                                      'effects':effects(logits[subset],original[subset],labels[subset]) if subset.any() else None},
                                  'descriptive_strata':strata})
        supplemental=[]
        for r in (0,1,2):
            data=[np.load(dest/f'q{q:02d}-r{r}-cycles-s10.npz') for q in range(5)]
            logits=np.stack([d['logits'] for d in data]);ham=np.stack([d['hamming'] for d in data])
            eligible=np.array([int(d['eligible_count']) for d in data])
            fraction=np.divide(ham,eligible[:,None],out=np.zeros_like(ham,dtype=float),where=eligible[:,None]>0)
            supplemental.append({'radius':r,'queries':5,'draws_per_query':8,
                                 'accepted_cycles':int(sum(d['accepted'].sum() for d in data)),
                                 'proposals':int(sum(d['proposals'].sum() for d in data)),
                                 'target_cycles':int(sum(d['cycle_targets'].sum() for d in data)),
                                 'mean_realised_Hamming_fraction':float(fraction.mean()),'no_change_draws':int((ham==0).sum()),
                                 'effects':effects(logits,original[:5],labels[:5])})
        rows.append({'architecture':rec['args']['arch'],'network':rec['args']['dataset'],'features':'random','T':32,'seed':10000,
                     'class_counts':np.bincount(labels,minlength=2).tolist(),'original_query_AUC':safe_auc(labels,original),
                     'original_query_Brier':float(np.mean((sigmoid(original)-labels)**2)),
                     'original_full_test_AUC':rec['evaluation']['test_auc'],
                     'mean_individual_reach':float(np.nanmean(reach)),
                     'individual_reach_by_query':[None if not np.isfinite(v) else int(v) for v in reach],
                     'common_neighbours':pr['common_neighbours'].tolist(),'minimum_endpoint_degrees':pr['endpoint_min_degree'].tolist(),
                     'cells':cells,'signed_degree_supplement':supplemental})
    return rows,inputs


def collect_training():
    rows=[];inputs={}
    for arch in ('SGCN','SIDNET'):
        folder=BASE/'training'/arch;record=folder/'complete.json';rec=pv.verify_record(record)
        assert rec['source_sha256']==source_hash('training') and rec['original_logits_bitwise_before_after']
        for name,digest in rec['files'].items():assert sha(folder/name)==digest
        inputs[str(record.relative_to(ROOT))]=sha(record)
        pr=np.load(folder/'profile.npz');cells=[]
        from collective.controls import synthetic_consensus,control_graph
        from collective.common import rng_for
        data=synthetic_consensus(256,seed=20261102)
        for key in ('edges','signs','X','queries','labels'):
            assert pv.array_sha256(data[key])==rec['generator_hashes'][2][key]
        assert pv.array_sha256(pr['all_test_logits'])==rec['test_logits_sha256']
        _,component_edges,_,terminals,nuisance=control_graph(9,3)
        variable=np.zeros(len(data['edges']),dtype=bool)
        for j in range(256):variable[j*len(component_edges)+np.r_[terminals,nuisance]]=True
        for r in (-1,0,1,2,3,5):
            logits=np.stack([np.load(folder/f'q{q:02d}-r{r}.npz')['logits'] for q in range(20)])
            completion_labels=[]
            for qi,slot in enumerate(pr['slots']):
                eligible=variable&(pr['distances'][qi]>r);truth=[]
                for draw in range(16):
                    rng=rng_for(5 if arch=='SGCN' else 6,qi,r,0,.5,draw)
                    changed=data['signs'].copy();changed[eligible]=rng.choice([-1.,1.],int(eligible.sum()))
                    truth.append(int(changed[int(slot)*len(component_edges)+terminals].sum()>0))
                completion_labels.append(truth)
            completion_labels=np.asarray(completion_labels)
            true_auc=[safe_auc(completion_labels[:,j],logits[:,j]) for j in range(16)]
            cells.append({'radius':r,'effects':effects(logits,pr['original_logits'],pr['labels']),
                          'true_Q_completion_accuracy':float(np.mean((logits>=0)==completion_labels)),
                          'mean_true_Q_completion_AUC':None if any(v is None for v in true_auc) else float(np.mean(true_auc)),
                          'pair_variance_pointwise_95':variance_interval(paired_variance(sigmoid(logits)))})
        rows.append({'architecture':arch,'test_auc':rec['test_auc'],
                     'test_accuracy_at_zero_logit':float(np.mean((pr['all_test_logits']>=0)==data['labels'])),
                     'selected_epoch':rec['training']['selected_epoch'],
                     'best_validation_auc':rec['training']['best_auc'],'epochs_run':rec['training']['epoch'],
                     'test_class_counts':rec['test_class_counts'],
                     'mean_gradient_reach':float(np.nanmean(pr['reach'])),
                     'mean_single_flip_reach':float(np.nanmean(pr['finite_reach'])),
                     'gradient_queries_recovering_distance_3':int(np.sum(pr['reach']>=3)),
                     'finite_queries_recovering_distance_3':int(np.sum(pr['finite_reach']>=3)),
                     'cells':cells})
    return rows,inputs


def make_figures(networks,known,training,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'savefig.bbox':'tight'})
    fig,axes=plt.subplots(2,2,figsize=(9,6),sharex=True)
    for ax,row in zip(axes.ravel(),networks):
        for mechanism,color in zip(MECHANISMS,('#225ea8','#d95f0e')):
            for strength,style in zip(STRENGTHS,('--','-')):
                c=[x for x in row['cells'] if x['mechanism']==mechanism and x['strength']==strength]
                ax.plot([x['radius'] for x in c],[x['effects']['probability_RMSE'] for x in c],
                        color=color,linestyle=style,marker='o',markersize=3,label=f'Independent flips, π={strength:g}' if mechanism=='independent' else f'Prevalence exchange, s={strength:g}')
        ax.set_title(f"{row['architecture']} · {dict(bitcoin_alpha='Bitcoin-Alpha',wiki_elec='Wiki-Elec')[row['network']]}")
        ax.set_ylabel('Probability RMSE from original');ax.set_ylim(bottom=0);ax.grid(alpha=.15)
        ax.set_xticks(RADII,['none','0','1','2','3']);ax.tick_params(labelbottom=True);ax.set_xlabel('Preserved radius (hops)')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,1.10),ncol=2,frameon=False,fontsize=9)
    fig.tight_layout();fig.savefig(out/'network_effects.pdf',metadata={'CreationDate':None,'ModDate':None});fig.savefig(out/'network_effects.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(11,3.3))
    cs=[x for x in known['rows'] if x['kind']=='consensus' and x['required_edge_distance']==3]
    axes[0].plot([x['paths'] for x in cs],[x['gradient']['fraction_reach_below_required'] for x in cs],marker='o',color='#225ea8')
    axes[0].set_xscale('log');axes[0].set_xticks([1,9,31,101],['1','9','31','101']);axes[0].set_ylim(0,1);axes[0].set_xlabel('Independent consensus paths');axes[0].set_ylabel('Fraction missing required distance')
    axes[0].set_title('Individual sensitivity can miss consensus')
    row=next(x for x in known['rows'] if x['kind']=='chain' and x['required_edge_distance']==3)
    axes[1].plot([x['radius'] for x in row['collective']],[x['analytic_population_variance'] for x in row['collective']],
                 marker='o',markersize=4,color='#333333')
    axes[1].set_xlabel('Preserved radius (hops)');axes[1].set_ylabel('True-Q conditional variance')
    axes[1].set_title('Collective variance recovers distance')
    axes[1].text(.04,.06,'Same curve for all four known solvers',transform=axes[1].transAxes,fontsize=8)
    for row,color in zip(training,('#225ea8','#d95f0e')):
        axes[2].plot([x['radius'] for x in row['cells']],[x['effects']['probability_RMSE'] for x in row['cells']],
                     marker='o',label=row['architecture'],color=color)
    axes[2].set_xlabel('Preserved radius (hops)');axes[2].set_ylabel('Probability RMSE from original');axes[2].set_title('Trained distributed control');axes[2].legend(frameon=False)
    for ax in axes:ax.grid(alpha=.15)
    for ax in axes[1:]:ax.set_xticks(range(-1,6),['none','0','1','2','3','4','5'])
    fig.tight_layout();fig.savefig(out/'control_comparison.pdf',metadata={'CreationDate':None,'ModDate':None});fig.savefig(out/'control_comparison.png',dpi=180);plt.close(fig)


def report(networks,known,training):
    nochange=sum(c['no_change_draws'] for row in networks for c in row['signed_degree_supplement'])
    accepted=sum(c['accepted_cycles'] for row in networks for c in row['signed_degree_supplement'])
    lines=['# Collective signed dependence: completed exploratory pilot','',
        'The first stage tests whether short individual reach survives joint changes to distant signs. '
        'The checkpoint and query choices, radii, mechanisms and draw counts were frozen before measurement. '
        'This is a small exploratory study of four standardized random-feature checkpoints, not a replication '
        'of native published objectives. It does not change the current manuscript or establish a submission score.','',
        '## Real networks','',
        'All original full-test logits match their stored hashes before and after measurement. '
        'There are 80 selected queries, 25,600 primary perturbed predictions and 480 supplementary predictions. '
        'The table uses strength 0.5. Probability RMSE is measured from the original prediction; '
        'disagreement counts changes to the predicted class. Exchange strength specifies the fraction of '
        'the smaller sign class exchanged, rather than an independent flip probability.','',
        '| Model / network | Mean individual reach | Independent: RMSE / disagreement beyond radius 1 | Exchange: RMSE / disagreement beyond radius 1 | Selected negatives / positives |',
        '|---|---:|---:|---:|---:|']
    for row in networks:
        a=next(x for x in row['cells'] if x['radius']==1 and x['mechanism']=='independent' and x['strength']==.5)['effects']
        b=next(x for x in row['cells'] if x['radius']==1 and x['mechanism']=='exchange' and x['strength']==.5)['effects']
        lines.append(f"| {row['architecture']} / {row['network']} | {row['mean_individual_reach']:.2f} | {a['probability_RMSE']:.4f} / {100*a['disagreement_fraction']:.1f}% | {b['probability_RMSE']:.4f} / {100*b['disagreement_fraction']:.1f}% | {row['class_counts'][0]} / {row['class_counts'][1]} |")
    lines+=['','Short individual reach does not guarantee stability to the specified joint changes. '
            'Restricting to queries whose individual reach is at most one hop still gives measurable '
            'effects when signs beyond one hop are exchanged while preserving global sign prevalence. '
            'The following table reports that pre-specified subset for every checkpoint.','',
            '| Model / network | Queries with individual reach ≤ 1 | Exchange probability RMSE | Exchange disagreement |',
            '|---|---:|---:|---:|']
    for row in networks:
        c=next(x for x in row['cells'] if x['radius']==1 and x['mechanism']=='exchange' and x['strength']==.5)
        subset=c['individual_reach_within_radius'];e=subset['effects']
        lines.append(f"| {row['architecture']} / {row['network']} | {subset['queries']}/20 | {e['probability_RMSE']:.4f} | {100*e['disagreement_fraction']:.1f}% |")
    lines+=['','These are stress-law effects on one checkpoint and a fixed 20-query cohort per setting, '
            'not causal effects or population estimates across models and networks. Exchange and independent '
            'flip strengths produce different realised Hamming fractions. For strength 0.5 beyond radius 1, '
            'the exchanged fraction is approximately 9% on Bitcoin-Alpha and 22% on Wiki-Elec, versus 50% '
            'for independent flips. The smaller effects under exchange therefore do not isolate a '
            'benefit of preserving prevalence. A later comparison needs matched perturbation sizes.','',
            'The empirical squared change is also decomposed into variation across completions plus '
            'the squared drift of the mean completion prediction from the original. This descriptive '
            'decomposition was added after the first Bitcoin-Alpha results; it is not a registered '
            'hypothesis or a new uncertainty guarantee. Low completion variance can coexist with '
            'a large, consistent shift from the original prediction.','',
            'Each primary mean variance has a simultaneous 95% Hoeffding interval over the 80-cell family. '
            'Those intervals cover Monte Carlo error for the fixed queries and chosen stress sampler; '
            'they do not cover query sampling, seed variability or distribution misspecification. '
            'Pointwise observed-configuration curves need not decrease. The pilot cannot certify negligible '
            'dependence from a small point estimate alone. Full effects, AUC, Brier scores, achieved Hamming '
            'fractions, individual-reach subsets and local strata are in summary.json.','',
            'The signed-degree sampler preserves every node’s positive and negative degree by alternating '
            'four-cycle flips. Its achieved perturbation can be much weaker than the requested target; '
            'accepted cycles, proposal counts and no-change draws must accompany its effect estimates. '
            f'Here it accepted only {accepted} cycles across 480 draws; {nochange}/480 draws changed no signs. '
            'The achieved perturbation is too weak to establish stability under substantial '
            'signed-degree-preserving changes. An effective constrained sampler is required next.','',
            '## Known functions and learned distributed control','',
            'The controls separate binary dependence from the continuous relaxation used for gradients. '
            'Their unsigned graphs and features contain no label information within a radius below the '
            'required distance. The known solver has margin at least 1.5. Individual measures can miss '
            'majority consensus, redundant evidence and saturation, while true conditional completions '
            'recover the remaining prediction variance. This conclusion concerns the stated synthetic Q.','',
            '| Control, required distance 3 | Paths | Configurations where gradient reach misses distance 3 | Single flips miss |',
            '|---|---:|---:|---:|']
    for kind,k in [('chain',1),('consensus',9),('consensus',31),('consensus',101),('redundancy',9),('saturated',1)]:
        row=next(x for x in known['rows'] if x['kind']==kind and x['paths']==k and x['required_edge_distance']==3)
        lines.append(f"| {kind} | {k} | {100*row['gradient']['fraction_reach_below_required']:.1f}% | {100*row['single_flip']['fraction_reach_below_required']:.1f}% |")
    lines+=['',            'Consensus distributions with at most nine paths are enumerated exactly. Larger consensus '
            'fractions use 8,192 fixed-seed configurations. The redundant distribution is enumerated exactly. '
            'All 27 control settings also pass the independent paired-completion check against their analytic '
            'population variance.','',
            '| Trained consensus, nine paths at distance 3 | Selected epoch | Validation AUC | Held-out AUC | Gradient recovery of distance 3 | Single-flip recovery of distance 3 |',
            '|---|---:|---:|---:|---:|---:|']
    for row in training:
        lines.append(f"| {row['architecture']} | {row['selected_epoch']} | {row['best_validation_auc']:.4f} | {row['test_auc']:.4f} | {row['gradient_queries_recovering_distance_3']}/20 | {row['finite_queries_recovering_distance_3']}/20 |")
    lines+=['','Both training controls use independent train, validation and test graphs, and checkpoint '
            'selection uses validation AUC only. They use the common node objective and are not tests of '
            'native link-prediction objectives. Their true-Q completions retain the control’s fixed connectors. '
            'The report also reconstructs each completion’s synthetic majority label and reports prediction '
            'accuracy and AUC against that changed label. This post hoc diagnostic uses the known control '
            'generator; real-network stress tests retain observed labels because their counterfactual truth '
            'is unknown. High AUC does not imply perfect classification at the fixed zero-logit threshold.','',
            '## Interpretation and next stage','',
            'The pilot establishes a reason to continue: individual reach can be short while joint changes '
            'beyond it alter predictions, including under prevalence-preserving exchanges. The known controls '
            'supply explicit failure cases, while both learned consensus controls recover distance three '
            'with gradients as well as finite flips. The network pilot '
            'measures sensitivity under specified stress distributions. It cannot identify causal changes '
            'in user behaviour or prove that a cropped computation preserves predictions.','',
            'Next, expand native-objective random-feature checks and controlled mechanism sweeps, then freeze '
            'a development/validation/held-out computation-selection experiment. Test the actual approximation '
            'against fixed shallow/deep and degree/triangle rules, including diagnostic overhead, fidelity, '
            'AUC, calibration, runtime and memory. The current paper and anonymous artifact remain the '
            'validated submission baseline until that evidence is complete.','',
            'See METHODS.md for the probability identities and assumptions, LITERATURE.md for prior work, '
            'and RESULT_MAP.md for the complete input/output mapping.','']
    return '\n'.join(lines)


def run(out):
    lock();out=Path(out);out.mkdir(parents=True,exist_ok=True)
    networks,inputs=collect_networks();training,ti=collect_training();inputs.update(ti)
    known=pv.verify_record(BASE/'results/known_controls.json')
    assert known['source_sha256']==source_hash('controls')
    inputs['collective/results/known_controls.json']=sha(BASE/'results/known_controls.json')
    summary={'protocol_sha256':lock()['protocol_sha256'],'collector_sha256':sha(__file__),
             'verified_inputs':inputs,'network_rows':networks,'training_rows':training,'known_control_rows':known['rows'],
             'primary_predictions':25600,'supplementary_predictions':480,'primary_variance_family_size':80}
    (out/'summary.json').write_text(json.dumps(summary,sort_keys=True,indent=1,allow_nan=False)+'\n')
    (out/'REPORT.md').write_text(report(networks,known,training))
    make_figures(networks,known,training,out)
    outputs={str(p.relative_to(out)):sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='MANIFEST.json'}
    (out/'MANIFEST.json').write_text(json.dumps({'collector_sha256':sha(__file__),'inputs':inputs,'outputs':outputs},sort_keys=True,indent=1)+'\n')
    print('COLLECTED',len(networks),'network checkpoints;',len(training),'training controls;',len(known['rows']),'known settings',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',default=str(BASE/'results/report'))
    run(ap.parse_args().out)
