from clearml import Model, Task
import joblib
import sys 

sys.modules['sklearn.externals.joblib'] = joblib

tsk = Task.init('sarcasm_detector','NN model inference ','inference')

sklearn_model_path = Model(model_id="e70fe949269b4ef6b9eb991eb8de48ae").get_local_copy()

sklearn_pipeline = joblib.load(sklearn_model_path)

args = {'sentences' : ["Coworkers At Bathroom Sink Locked In Tense Standoff Over Who Going To Wash Hands Longer","grandma jumps into buick for emergency birdseed run"]}
tsk.connect(args)

scores = sklearn_pipeline.predict_proba(args['sentences'])

print(f"Input Sentences :\n{args['sentences']}")

for idx, score in enumerate(scores):
    score = score[0]
    if score < 0.5:
        label = 'NORMAL'
        score = 1 - score
    else:
        label = 'SARCASTIC'
    print(f"Commnet: {args['sentences'][idx]}\nLABEL: {label}\nCERTAINTY: {score:.2f}\n")

tsk.close()
    
