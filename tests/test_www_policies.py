"""Independent policy selection, portable tree and fidelity accounting checks."""
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from collective.stage2.policies import serialize_tree,tree_predict,choose_predictions,metrics,acceptable,choices_for

def test_portable_tree_and_cost_feasibility():
    rng=np.random.default_rng(31);X=rng.normal(size=(60,5));targets=np.abs(rng.normal(size=(60,8)));targets[:,0]=0
    tree=DecisionTreeRegressor(max_depth=3,min_samples_leaf=10,random_state=20261009).fit(X,targets)
    assert np.allclose(tree_predict(serialize_tree(tree),X),tree.predict(X))
    cost=np.arange(8.,0.,-1.)
    predictions=np.array([[0,.2,.3,.004,.1,.2,.4,.5],[0,.02,.03,.04,.05,.06,.07,.08]])
    assert np.array_equal(choose_predictions(predictions,cost,.01),[3,0])

def test_fidelity_and_label_metrics_are_distinct():
    y=np.array([-1.,1.,-1.,1.]);full=np.array([.2,.8,.3,.7])
    same=metrics(full,full,y)
    assert same['probability_RMSE']==same['class_disagreement']==same['AUC_loss']==0 and acceptable(same)
    constant=metrics(np.full(4,.5),full,y)
    assert constant['test_AUC']==.5 and constant['probability_RMSE']>.2 and not acceptable(constant)
    # Preserving ranking/class does not preserve probabilities or calibration.
    compressed=.5+.1*(full-.5);m=metrics(compressed,full,y)
    assert m['test_AUC']==1 and m['class_disagreement']==0 and m['probability_RMSE']>.2 and not acceptable(m)
    assert m['negative_queries']==m['positive_queries']==2

def test_simple_rules_and_fixed_policy():
    rows=[{'min_degree':d,'common_neighbours':t} for d,t in ((2,0),(8,4),(9,3))]
    assert np.array_equal(choices_for({'rule':'degree','threshold':5},rows),[5,0,0])
    assert np.array_equal(choices_for({'rule':'triangle','threshold':3},rows),[0,5,0])
    assert np.array_equal(choices_for({'rule':'fixed','choice':3},rows),[3,3,3])
TESTS=[test_portable_tree_and_cost_feasibility,test_fidelity_and_label_metrics_are_distinct,test_simple_rules_and_fixed_policy]
