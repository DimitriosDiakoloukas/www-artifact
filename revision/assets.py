"""Tables and numerical macros for the two reviewer-requested studies; no handwritten results."""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
LABEL={"bitcoin_alpha":"Bitcoin-Alpha","bitcoin_otc":"Bitcoin-OTC","wiki_elec":"Wiki-Elec","SLGNN":"SLGNN-style"}
def label(value):return LABEL.get(value,value)
def revision_numbers(M,audit,native):
    keys=("nAuditQueries","nAuditCheckpoints","nAuditComparisons","nAuditFiniteMean","nAuditGradientMean",
          "nAuditAgreement","nAuditWithinOne","nAuditFarQueries","nAuditFarFlips","nAuditDisagree",
          "nNativeRuns","nNativeTuning","nNativeMinReach","nNativeMaxReach","nNativeMinAuc","nNativeMaxAuc",
          "nNativeMinDepthReach","nNativeMaxDepthReach","nNativeMaxForwardError","nAuditMinFinite","nAuditMaxFinite", "nAuditMinEffect", "nAuditMaxEffect", "nAuditMaxFarEffect",
          "nAuditStrong", "nAuditWeakMisses", "nAuditWeakPercent", "nAuditFlipQueries", "nAuditMaxDistance")
    if audit is None or native is None:
        for name in keys:M.pending(name,"reviewer-requested validation running")
        return
    assert audit["all_logits_match"] and native["all_logits_match"]
    for name,key in (("nAuditQueries","queries"),("nAuditCheckpoints","checkpoints"),("nAuditFarQueries","finite_gt3_queries"),
                     ("nAuditFarFlips","far_prediction_flip_queries"),("nAuditDisagree","finite_gradient_disagreement_queries")):
        M.put(name,audit[key],"{:d}")
    M.put("nAuditComparisons",sum(r["relations"] for r in audit["rows"]),"{:,d}")
    M.put("nAuditFiniteMean",audit["finite_mean"]);M.put("nAuditGradientMean",audit["gradient_mean"])
    M.put("nAuditAgreement",100*audit["agreement"],"{:.0f}");M.put("nAuditWithinOne",100*audit["within_one"],"{:.0f}")
    f=[r["finite_0.1"] for r in audit["rows"] if r["finite_0.1"] is not None]
    M.put("nAuditMinFinite",min(f),"{:d}");M.put("nAuditMaxFinite",max(f),"{:d}")
    rows=audit["rows"]
    M.put("nAuditMinEffect",min(r["finite_max"] for r in rows),"{:.2f}")
    M.put("nAuditMaxEffect",max(r["finite_max"] for r in rows),"{:.2f}")
    M.put("nAuditMaxFarEffect",max(r["far_max"] for r in rows),"{:.3f}")
    strong=sum(r["finite_strong"] for r in rows);missed=sum(r["finite_strong_gradient_weak"] for r in rows)
    M.put("nAuditStrong",strong,"{:,d}");M.put("nAuditWeakMisses",missed,"{:,d}")
    M.put("nAuditWeakPercent",100*missed/strong,"{:.0f}")
    M.put("nAuditFlipQueries",sum(r["prediction_flips"]>0 for r in rows),"{:d}")
    M.put("nAuditMaxDistance",max(r["max_distance"] for r in rows),"{:d}")
    M.put("nNativeRuns",native["runs"],"{:d}");M.put("nNativeTuning",native["tuning_runs"],"{:d}")
    cells=[c for c in native["cells"] if c["T"]==32]
    M.put("nNativeMinReach",min(c["reach_0.1"][0] for c in cells))
    M.put("nNativeMaxReach",max(c["reach_0.1"][0] for c in cells))
    M.put("nNativeMinAuc",min(c["test_auc"][0] for c in cells),"{:.3f}")
    M.put("nNativeMaxAuc",max(c["test_auc"][0] for c in cells),"{:.3f}")
    changes=[c["reach_0.1_change"][0] for c in native["paired_depth"]]
    M.put("nNativeMinDepthReach",min(changes),"{:+.2f}");M.put("nNativeMaxDepthReach",max(changes),"{:+.2f}")
    M.put("nNativeMaxForwardError",native["binary_forward_max_error"],"{:.2e}")
def native_main(native):
    by={(c["network"],c["arch"],c["T"]):c for c in native["cells"]}
    lines=[r"\begin{tabular}{@{}llcc@{}}",r"\toprule",
           r"Network & Model & AUC, $8\to32$ & Reach, $8\to32$ \\",r"\midrule"]
    for net in ("bitcoin_alpha","wiki_elec"):
        for arch in ("SGCN","SIDNET"):
            a,b=by[net,arch,8],by[net,arch,32]
            lines.append(f"{label(net)} & {arch} & {a['test_auc'][0]:.3f}\\(\\to\\){b['test_auc'][0]:.3f} & "
                         f"{a['reach_0.1'][0]:.2f}\\(\\to\\){b['reach_0.1'][0]:.2f} "+r"\\")
    return "\n".join(lines+[r"\bottomrule",r"\end{tabular}"])+"\n"
def native_details(native):
    lines=[r"\begin{tabular}{@{}llrccc@{}}",r"\toprule",
           r"Network & Model & $T$ & Test AUC [95\% CI] & Reach [95\% CI] & Ceiling \\",r"\midrule"]
    ci=lambda v,fmt: (fmt.format(v[0])+" ["+fmt.format(v[1])+", "+fmt.format(v[2])+"]") if all(x is not None for x in v) else "undefined"
    for c in native["cells"]:
        lines.append(f"{label(c['network'])} & {c['arch']} & {c['T']} & "
                     f"{ci(c['test_auc'],'{:.3f}')} & {ci(c['reach_0.1'],'{:.2f}')} & {c['ceiling'][0]:.2f} "+r"\\")
    return "\n".join(lines+[r"\bottomrule",r"\end{tabular}"])+"\n"
def audit_details(audit):
    lines=[r"\begin{tabular}{@{}llccc@{}}",r"\toprule",
           r"Network & Model & Finite & Gradient & Agree \\",r"\midrule"]
    for net in ("bitcoin_alpha","bitcoin_otc"):
        for arch in ("SGCN","SLGNN","SIDNET","BGSD"):
            rows=[r for r in audit["rows"] if (r["network"],r["arch"])==(net,arch)]
            valid=[r for r in rows if r["finite_0.1"] is not None and r["gradient_0.1"] is not None]
            assert len(valid)==len(rows),"report undefined profiles explicitly before publishing table"
            finite=np.mean([r["finite_0.1"] for r in rows]);gradient=np.mean([r["gradient_0.1"] for r in rows])
            agree=sum(r["finite_0.1"]==r["gradient_0.1"] for r in rows)
            lines.append(f"{label(net)} & {label(arch)} & {finite:.2f} & {gradient:.2f} & {agree}/{len(rows)} "+r"\\")
    return "\n".join(lines+[r"\bottomrule",r"\end{tabular}"])+"\n"
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--paper",type=Path,required=True);a=ap.parse_args()
    gen=a.paper/"generated"
    audit=json.loads((gen/"exhaustive_audit.json").read_text());native=json.loads((gen/"native_models.json").read_text())
    for name,text in (("native_check.tex",native_main(native)),("native_details.tex",native_details(native)),
                      ("exhaustive_table.tex",audit_details(audit))):
        (gen/name).write_text(text)
        print(name)
if __name__=="__main__":main()

