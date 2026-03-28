import pandas as pd
df = pd.read_csv('data/splits/fold_assignments.csv', dtype={'patient_id': str})
print(df[df['patient_id'].isin(['267630','1230482'])][['patient_id','fold_id','brugada']])