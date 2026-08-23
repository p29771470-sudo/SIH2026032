"""DHAANYA demo prediction API. Synthetic training data only; replace with authorised centre history in production."""
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestRegressor
import random

app=FastAPI(title='DHAANYA API',version='1.0.0')
random.seed(26032)
X=[];y=[]
for _ in range(500):
 q,counters,staff,qty,hour,day,capacity=[random.randint(0,60),random.randint(1,5),random.randint(1,5),random.randint(300,2500),random.randint(8,17),random.randint(0,6),random.randint(40,100)]
 X.append([q,counters,staff,qty,hour,day,capacity]); y.append(max(5,q*11/max(counters*staff,1)+qty/450+(hour-12)**2*0.7+(100-capacity)*.15+random.uniform(-5,5)))
model=RandomForestRegressor(n_estimators=80,random_state=26032).fit(X,y)
class WaitInput(BaseModel):
 current_queue:int=Field(ge=0,le=500); counters:int=Field(ge=1,le=20); staff_available:int=Field(ge=1,le=30); quantity_kg:float=Field(ge=0); hour:int=Field(ge=0,le=23); day_of_week:int=Field(ge=0,le=6); centre_capacity:int=Field(ge=1,le=100)
@app.get('/health')
def health(): return {'status':'ok','data_mode':'synthetic_demo'}
@app.post('/predictions/wait-time')
def wait_time(x:WaitInput):
 minutes=round(float(model.predict([[x.current_queue,x.counters,x.staff_available,x.quantity_kg,x.hour,x.day_of_week,x.centre_capacity]])[0]))
 return {'predicted_wait_minutes':minutes,'data_mode':'synthetic_demo','explanation':'Estimate combines queue, counters, staff, quantity, time and centre capacity.'}
@app.get('/predictions/crowd')
def crowd(): return {'today':'Medium','tomorrow':'High','day_after':'Low','data_mode':'synthetic_demo'}
