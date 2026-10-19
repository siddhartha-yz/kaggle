"""Original SAT-conditional rating marginals given eight nonrating context variables."""
from research_r60_source_data import data as original_data,frame,observed_probability,RATINGS,CONDITION,ROOT
def condition_columns(cols):
    fields=[c for c in cols if c not in RATINGS]
    assert fields==['Age','Flight Distance','Departure Delay in Minutes','Arrival Delay in Minutes','Gender','Customer Type','Type of Travel','Class'] and not set(fields)&set(RATINGS)
    return fields
def data():
    source,sat,train,test,cols,scope=original_data();scope=dict(scope,all_rating_inputs_excluded=True,nonrating_context_columns=condition_columns(cols),source_density_conditioning='8nonrating context covariates plus ORIGINAL SAT hypothesis; all13ratings excluded from model inputs')
    return source,sat,train,test,cols,scope
