from django.shortcuts import render
from .models import *
import json 

def index(request):
    return render(request, 'index.html') 

def query(request):
    if request.method == 'POST' :
        import chat
        return render(request, 'query.html',{"greed":"Finish!"})
    
    else:
        return render(request, 'query.html')
# import pandas as pd

# df = pd.read_excel("practical_data.xlsx")

import pandas as pd
import numpy as np
def Round3(request):
    if request.method == 'POST' :
        from interview1 import All
        df = pd.read_excel("practical_data.xlsx")
        df = All(df,1)     
        final = np.round(np.mean(df['result'].values),4)*100
        
        json_records = df.reset_index().to_json(orient ='records')
        arr = []
        arr = json.loads(json_records)
        context = {'d': arr,"result":final}
        return render(request, 'dataframe.html',context)
    
    else:
        return render(request, 'Round3.html')
    
# def Round3(request):
#     if request.method == 'POST' :
#         # global df2
#         from interview1 import All
#         return render(request, 'Round3.html',{"greed":"Interview is Finish!"})
    
#     else:
#         return render(request, 'Round3.html')
        
def Round1(request):
    if request.method == 'POST':
        print(request.POST)
        questions=QuesModel.objects.all()
        score=0
        wrong=0
        correct=0
        total=0
        for q in questions:
            total+=1
            print(request.POST.get(q.question))
            print(q.ans)
            print()
            print(request.POST.get(q.question))
            if q.ans ==  request.POST.get(q.question):
                
                score+=10
                correct+=1
            else:
                wrong+=1
        percent = score/(total*10) *100
        context = {
            'score':score,
            'time': request.POST.get('timer'),
            'correct':correct,
            'wrong':wrong,
            'percent':percent,
            'total':total
        }
        if percent>=80:
            return render(request,'result.html',context)
        else:
            return render(request,'result2.html',context)
            
    else:
        questions=QuesModel.objects.all()
        context = {'questions':questions}
        return render(request,'Round1.html',context)

def Round2(request):
    import pandas as pd


    df_test = pd.read_excel("practical_data.xlsx")
    # df = df[:10]

    json_records = df_test.reset_index().to_json(orient ='records')
    arr = []
    arr = json.loads(json_records)
    contextt = {'d': arr}
    return render(request, 'dataframe.html',contextt) 