import pyttsx3
import speech_recognition as sr 

import pandas as pd
import numpy as np
from time import sleep
import datetime


import nltk
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import string
import math



lemmatizer = WordNetLemmatizer()
wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}
STOPWORDS = set(stopwords.words('english'))



def preprocessing(text):
    PUNCT_TO_REMOVE = '!"#%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    punch = text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))
    pos_tagged_text = nltk.pos_tag(punch.split())
    return  " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])



def prep_all(text):
    punch = string.punctuation
    punch = text.translate(str.maketrans('', '', punch))
    stop = " ".join([word for word in str(punch).split() if word not in STOPWORDS])
    
    pos_tagged_text = nltk.pos_tag(stop.split())
    return  " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])




def Apply(df):
    df["Answer"] = df["Answer"].str.lower()
    df["Answer"] = df["Answer"].apply(lambda text: preprocessing(text))
    return df

def Apply_all(df):
    df["Answer"] = df["Answer"].str.lower()
    df["Answer"] = df["Answer"].apply(lambda text: prep_all(text))
    return df


def random_list(n,count=5):
    l = list()
    for i in range(10000):
        a = np.random.randint(n)
        l.append(a)
        l2 = list(set(l))
        if len(l2)>=count:
            break
    return l2

def random_df(df,l):
    l1,l2 = list(),list()
    for i in l:
        l1.append(df.loc[i,"Question"])
        l2.append(df.loc[i,"Answer"])
    d = pd.DataFrame({"Question":l1,"Answer":l2})
    return d

engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)  
engine.setProperty('rate', 170)
engine.setProperty('volume',1.0) 


def speak(audio):
    engine.say(audio)
    engine.runAndWait()

def recognizer(quesstion):  
    try: 
        r = sr.Recognizer()
        with sr.Microphone() as voice:
            print(quesstion)
            speak(quesstion)
            r.adjust_for_ambient_noise(voice, duration=.2) 
            speak("listening")
            audio = r.listen(voice)
            MyText = r.recognize_google(audio)
            MyText = MyText.lower()
            print("Did you say..."+MyText)
            speak("Did you say..."+MyText)
            print("")
        return MyText
    except Exception as e:
        print('voice not found...')
        speak("voice not found...")
        return " "
    except KeyboardInterrupt:
        pass
    
    
    

def recognizer1(quesstion):  
    try: 
        r = sr.Recognizer()
        with sr.Microphone() as voice:
            print(quesstion)
            speak(quesstion)
            r.adjust_for_ambient_noise(voice, duration=.2) 
            speak("listening")
            audio = r.listen(voice)
            MyText = r.recognize_google(audio)
            MyText = MyText.lower()
#             speak(MyText)
        return MyText
    except Exception as e:
        print('voice not found...')
        speak("voice not found...")
        return " "
    except KeyboardInterrupt:
        pass
    
    
    
    
    
def user_df(df):
    d1,d2 = list(),list()
    for i in df["Question"]:
        temp = True
        while temp:
            a = recognizer(i)
            opt = recognizer1("\n for next say : next \n for repeat question say :  No")            
            if "next" in opt:
                d1.append(i)   
                d2.append(a)
                temp = False
                sleep(1)
                print("")
            elif "no" in opt:  
                temp = True
                sleep(1)
                print("")
            else :
                temp = True
                sleep(1)
                print("")
    user_ans = pd.DataFrame({"Question":d1,"Answer":d2})
    return user_ans

def greed():
    hour = int(datetime.datetime.now().hour)
    if hour>=0 and hour<12:
        speak("Good Morning!")
        return "Good Morning!"
        
        
    elif hour>=12 and hour<18:
        speak("Good Afternoon!")
        return "Good Afternoon!"
        
    else:
        speak("Good Evening!")
        speak("Welcome to Automated interview system This is HR Round....")
        return "Good Evening!","Welcome to Automated interview system This is HR Round...."
    

def bye():
    speak("Thanks for coming your  interview has been finish")
    speak("Your result will be declared soon")

#******************************tfidf************************************************

def computeTF(wordDict, doc):
    tfDict = {}
    corpusCount = len(doc)
    for word, count in wordDict.items():
        tfDict[word] = count/float(corpusCount)
    return(tfDict)

def computeIDF(docList):
    idfDict = {}
    N = len(docList)
    
    idfDict = dict.fromkeys(docList[0].keys(), 0)
    for word, val in idfDict.items():
        idfDict[word] = math.log10(N / (float(val) + 1))
        
    return(idfDict)

def computeTFIDF(tfBow, idfs):
    tfidf = {}
    for word, val in tfBow.items():
        tfidf[word] = val*idfs[word]
    return(tfidf)

def cosin_similarity(first_sentence,second_sentence):
    first_sentence = first_sentence.split(" ")
    second_sentence = second_sentence.split(" ")#join them to remove common duplicate words
    total= set(first_sentence).union(set(second_sentence))
    
    wordDictA = dict.fromkeys(total, 0) 
    wordDictB = dict.fromkeys(total, 0)
    for word in first_sentence:
        wordDictA[word]+=1
    
    for word in second_sentence:
        wordDictB[word]+=1
        
        
    tfFirst = computeTF(wordDictA, first_sentence)
    tfSecond = computeTF(wordDictB, second_sentence)
    
    idfs = computeIDF([wordDictA, wordDictB])
    
    idfFirst = computeTFIDF(tfFirst, idfs)
    idfSecond = computeTFIDF(tfSecond, idfs)
    
    x = np.array(list(idfFirst.values()))
    y = np.array(list(idfSecond.values()))
    
    dot_product = np.dot(x, y)
    
    magnitude_x = np.sqrt(np.sum(x**2)) 
    magnitude_y = np.sqrt(np.sum(y**2))
    
    cosine_similarity = dot_product / (magnitude_x * magnitude_y)
    
    return cosine_similarity
#*****************************distance********************************************

def dis_col(df,user):
    return [cosin_similarity(df.loc[i,"Answer"],user.loc[i,"Answer"]) for i in range(len(df))]





def cosin_similarity2(first_sentence,second_sentence):
    first_sentence = first_sentence.split(" ")
    second_sentence = second_sentence.split(" ")#join them to remove common duplicate words
    total= set(first_sentence).union(set(second_sentence))
    
    wordDictA = dict.fromkeys(total, 0) 
    wordDictB = dict.fromkeys(total, 0)
    for word in first_sentence:
        wordDictA[word]+=1
    
    for word in second_sentence:
        wordDictB[word]+=1
        
        
    tfFirst = computeTF(wordDictA, first_sentence)
    tfSecond = computeTF(wordDictB, second_sentence)
    
    idfs = computeIDF([wordDictA, wordDictB])
    
    idfFirst = computeTFIDF(tfFirst, idfs)
    idfSecond = computeTFIDF(tfSecond, idfs)
    
    x = np.array(list(idfFirst.values()))
    y = np.array(list(idfSecond.values()))
    print(x)
    
    dot_product = np.dot(x, y)
    
    magnitude_x = np.sqrt(np.sum(x**2)) 
    magnitude_y = np.sqrt(np.sum(y**2))
    
    cosine_similarity = (dot_product / (magnitude_x * magnitude_y))*.70
    
    return cosine_similarity
#*****************************distance********************************************

def dis_col2(df,user):
    return [cosin_similarity2(df.loc[i,"Answer"],user.loc[i,"Answer"]) for i in range(len(df))]



def must(ans,user):
    temp = 0
    l = [i.replace('$', '') for i in ans.split( ) if i[0]=='$']
    l2 = [1 if l[i] in user else 0 for i in range(len(l))]
    if 1 in  l2: temp = 1
#     if 0 and 1 in l2: temp = 0
    return temp

def check(df,user):
    lst1 = list()
    for i in range(len(df)):  
        c = must(df.loc[i,"Answer"],user.loc[i,"Answer"])
        lst1.append(c)
    return lst1 


def result(df):
    cosine = df['Cosine'].values
    keyword = df['keyword_match'].values
    l= []
    for i in range(len(df)):
        if keyword[i]==1 or keyword[i]==np.nan:
            sum1 = cosine[i]+.30
            l.append(sum1)
        else:
            l.append(cosine[i])
    return l


import copy
def All(df,no_of_question):
    greed()
    l = random_list(len(df),no_of_question)
    df = random_df(df,l)
    df_C = copy.copy(df)

    df2 = Apply_all(df_C)
    user = user_df(df_C)
    user2 = Apply_all(user)
    
    df3 = Apply(df)
    user3 = Apply(user)
    
    bye()
    df['user_answer'] = user['Answer']
    
    df['real_cosine'] = dis_col(df2,user2)
    df['Cosine'] = dis_col2(df2,user2)
    
    df["keyword_match"] = check(df3,user3)
    df['result'] = result(df)
    return df



df = pd.read_excel("practical_data.xlsx")
df = All(df,1)     #second position : number of question
# df
print(df)

# final = np.round(np.mean(df['result'].values),4)*100
# print(final)































"""import pyttsx3
import speech_recognition as sr 

import pandas as pd
import numpy as np
from time import sleep
import datetime

import nltk
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import string



lemmatizer = WordNetLemmatizer()
wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}
STOPWORDS = set(stopwords.words('english'))



def preprocessing(text):
    PUNCT_TO_REMOVE = '!"#%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    punch = text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))
    pos_tagged_text = nltk.pos_tag(punch.split())
    return  " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])



def prep_all(text):
    punch = string.punctuation
    punch = text.translate(str.maketrans('', '', punch))
    stop = " ".join([word for word in str(punch).split() if word not in STOPWORDS])
    
    pos_tagged_text = nltk.pos_tag(stop.split())
    return  " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])




def Apply(df):
    df["Answer"] = df["Answer"].str.lower()
    df["Answer"] = df["Answer"].apply(lambda text: preprocessing(text))
    return df

def Apply_all(df):
    df["Answer"] = df["Answer"].str.lower()
    df["Answer"] = df["Answer"].apply(lambda text: prep_all(text))
    return df


def random_list(n,count=5):
    l = list()
    for i in range(10000):
        a = np.random.randint(n)
        l.append(a)
        l2 = list(set(l))
        if len(l2)>=count:
            break
    return l2

def random_df(df,l):
    l1,l2 = list(),list()
    for i in l:
        l1.append(df.loc[i,"Question"])
        l2.append(df.loc[i,"Answer"])
    d = pd.DataFrame({"Question":l1,"Answer":l2})
    return d

engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)  
engine.setProperty('rate', 170)
engine.setProperty('volume',1.0) 


def speak(audio):
    engine.say(audio)
    engine.runAndWait()

def recognizer(quesstion):  
    try: 
        r = sr.Recognizer()
        with sr.Microphone() as voice:
            print(quesstion)
            speak(quesstion)
            r.adjust_for_ambient_noise(voice, duration=.2) 
            speak("listening")
            audio = r.listen(voice)
            MyText = r.recognize_google(audio)
            MyText = MyText.lower()
            speak("Did you say..."+MyText)
            print("")
        return MyText
    except Exception as e:
        print('voice not found...')
        speak("voice not found...")
        return " "
    except KeyboardInterrupt:
        pass
    
    
    

def recognizer1(quesstion):  
    try: 
        r = sr.Recognizer()
        with sr.Microphone() as voice:
            print(quesstion)
            speak(quesstion)
            r.adjust_for_ambient_noise(voice, duration=.2) 
            speak("listening")
            audio = r.listen(voice)
            MyText = r.recognize_google(audio)
            MyText = MyText.lower()
#             speak(MyText)
        return MyText
    except Exception as e:
        print('voice not found...')
        speak("voice not found...")
        return " "
    except KeyboardInterrupt:
        pass
    
    
    
    
    
def user_df(df):
    d1,d2 = list(),list()
    for i in df["Question"]:
        temp = True
        while temp:
            a = recognizer(i)
            opt = recognizer1("\n for next say : Yes \n for repeat question say :  No")            
            if "yes" in opt:
                d1.append(i)   
                d2.append(a)
                temp = False
                sleep(1)
                print("")
            elif "no" in opt:  
                temp = True
                sleep(1)
                print("")
            else :
                temp = True
                sleep(1)
                print("")
    user_ans = pd.DataFrame({"Question":d1,"Answer":d2})
    return user_ans

def greed():
    hour = int(datetime.datetime.now().hour)
    if hour>=0 and hour<12:
        speak("Good Morning!")
        return "Good Morning!"
        
        
    elif hour>=12 and hour<18:
        speak("Good Afternoon!")
        return "Good Afternoon!"
        
    else:
        speak("Good Evening!")
        speak("Welcome to Automated interview system This is HR Round....")
        return "Good Evening!","Welcome to Automated interview system This is HR Round...."
    

def bye():
    speak("Thanks for coming your  interview has been finish")
    speak("Your result will be declared soon")

          
def cosin_similarity(X,Y):
    l1,l2 =[],[]
    X_list,Y_list = word_tokenize(X),word_tokenize(Y)  
    global sw
    sw = stopwords.words('english')
    
    X_set = {w for w in X_list if not w in sw}  
    Y_set = {w for w in Y_list if not w in sw} 
    
    rvector = X_set.union(Y_set)  
    for w in rvector: 
        if w in X_set: l1.append(1)
        else: l1.append(0) 
            
        if w in Y_set: l2.append(1) 
        else: l2.append(0) 
            
    c = 0
    for i in range(len(rvector)): 
            c+= l1[i]*l2[i] 
    cosine = (c / ((float((sum(l1)*sum(l2))**0.5)) + .00000001))
    return cosine

def dis_col(df,user):
    return [cosin_similarity(df.loc[i,"Answer"],user.loc[i,"Answer"]) for i in range(len(df))]



def cosin_similarity2(X,Y):
    l1 =[]
    l2 =[]
    X_list = word_tokenize(X)  
    Y_list = word_tokenize(Y)
    global sw
    sw = stopwords.words('english')
    
    X_set = {w for w in X_list if not w in sw}  
    Y_set = {w for w in Y_list if not w in sw} 
    
    rvector = X_set.union(Y_set)  
    for w in rvector: 
        if w in X_set: l1.append(1)
        else: l1.append(0) 
            
        if w in Y_set: l2.append(1) 
        else: l2.append(0) 
            
    c = 0
    for i in range(len(rvector)): 
            c+= l1[i]*l2[i] 
    cosine = (c / ((float((sum(l1)*sum(l2))**0.5)) + .00000001))*.70
    return cosine


def dis_col2(df,user):
    return [cosin_similarity2(df.loc[i,"Answer"],user.loc[i,"Answer"]) for i in range(len(df))]


def must(ans,user):
    temp = 0
    l = [i.replace('$', '') for i in ans.split( ) if i[0]=='$']
    l2 = [1 if l[i] in user else 0 for i in range(len(l))]
    if 1 in  l2: temp = 1
#     if 0 and 1 in l2: temp = 0
    return temp

def check(df,user):
    lst1 = list()
    for i in range(len(df)):  
        c = must(df.loc[i,"Answer"],user.loc[i,"Answer"])
        lst1.append(c)
    return lst1 


def result(df):
    cosine = df['Cosine'].values
    keyword = df['keyword_match'].values
    l= []
    for i in range(len(df)):
        if keyword[i]==1 or keyword[i]==np.nan:
            sum1 = cosine[i]+.30
            l.append(sum1)
        else:
            l.append(cosine[i])
    return l


import copy
def All(df,no_of_question):
    greed()
    l = random_list(len(df),no_of_question)
    df = random_df(df,l)
    df_C = copy.copy(df)

    df2 = Apply_all(df_C)
    user = user_df(df_C)
    user2 = Apply_all(user)
    
    df3 = Apply(df)
    user3 = Apply(user)
    
    bye()
    df['user_answer'] = user['Answer']
    
    df['real_cosine'] = dis_col(df2,user2)
    df['Cosine'] = dis_col2(df2,user2)
    
    df["keyword_match"] = check(df3,user3)
    df['result'] = result(df)
    return df



df = pd.read_excel("practical_data.xlsx")
df = All(df,1)     #second position : number of question

print(df)
final = np.round(np.mean(df['result'].values),4)*100"""
















"""
    #****************************************library**********************************************************
import pyttsx3
import speech_recognition as sr 

import pandas as pd
import numpy as np
from time import sleep
import datetime
import math

import nltk
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
# from nltk.tokenize import word_tokenize
import string




#***********************************preprocessing
lemmatizer = WordNetLemmatizer()
wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}
STOPWORDS = set(stopwords.words('english'))

def preprocessing(text):
    punch = string.punctuation
    punch = text.translate(str.maketrans('', '', punch))
    stop = " ".join([word for word in str(punch).split() if word not in STOPWORDS])
    
    pos_tagged_text = nltk.pos_tag(stop.split())
    return  " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])

def Apply(df):
    df["Answer"] = df["Answer"].str.lower()
    df["Answer"] = df["Answer"].apply(lambda text: preprocessing(text))
    return df



#************************voice and recognize************************************
engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)  
engine.setProperty('rate', 170)
engine.setProperty('volume',1.0) 


def speak(audio):
    engine.say(audio)
    engine.runAndWait()

def recognizer(quesstion):  
    try: 
        r = sr.Recognizer()
        with sr.Microphone() as voice:
            print(quesstion)
            speak(quesstion)
            r.adjust_for_ambient_noise(voice, duration=.2) 
            speak("listening")
            audio = r.listen(voice)
            MyText = r.recognize_google(audio)
            MyText = MyText.lower()
            speak("Did you say..."+MyText)
            print("")
        return MyText
    except Exception as e:
        print('voice not found...')
        speak("voice not found...")
        return " "
    except KeyboardInterrupt:
        pass
    
    
    

def recognizer1(quesstion):  
    try: 
        r = sr.Recognizer()
        with sr.Microphone() as voice:
            print(quesstion)
            speak(quesstion)
            r.adjust_for_ambient_noise(voice, duration=.2) 
            speak("listening")
            audio = r.listen(voice)
            MyText = r.recognize_google(audio)
            MyText = MyText.lower()
#             speak(MyText)
        return MyText
    except Exception as e:
        print('voice not found...')
        speak("voice not found...")
        return " "
    except KeyboardInterrupt:
        pass
    
    
    
    
    
def user_df(df):
    d1,d2 = list(),list()
    for i in df["Question"]:
        temp = True
        while temp:
            a = recognizer(i)
            opt = recognizer1("\n for next say : Yes \n for repeat question say :  No")            
            if "yes" in opt:
                d1.append(i)   
                d2.append(a)
                temp = False
                sleep(1)
                print("")
            elif "no" in opt:  
                temp = True
                sleep(1)
                print("")
            else :
                temp = True
                sleep(1)
                print("")
    user_ans = pd.DataFrame({"Question":d1,"Answer":d2})
    return user_ans

def greed():
    hour = int(datetime.datetime.now().hour)
    if hour>=0 and hour<12:
        speak("Good Morning!")
        return "Good Morning!"
        
        
    elif hour>=12 and hour<18:
        speak("Good Afternoon!")
        return "Good Afternoon!"
        
    else:
        speak("Good Evening!")
        speak("Welcome to Automated interview system This is HR Round....")
        return "Good Evening!","Welcome to Automated interview system This is HR Round...."
    

def bye():
    speak("Thanks for coming your  interview has been finish")
    speak("Your result will be declared soon")


    #*****************************************random list*************************************************
    
def random_list(n,count=5):
    l = list()
    for i in range(10000):
        a = np.random.randint(n)
        l.append(a)
        l2 = list(set(l))
        if len(l2)>=count:
            break
    return l2

def random_df(df,l):
    l1,l2 = list(),list()
    for i in l:
        l1.append(df.loc[i,"Question"])
        l2.append(df.loc[i,"Answer"])
    d = pd.DataFrame({"Question":l1,"Answer":l2})
    return d



#******************************tfidf************************************************

def computeTF(wordDict, doc):
    tfDict = {}
    corpusCount = len(doc)
    for word, count in wordDict.items():
        tfDict[word] = count/float(corpusCount)
    return(tfDict)

def computeIDF(docList):
    idfDict = {}
    N = len(docList)
    
    idfDict = dict.fromkeys(docList[0].keys(), 0)
    for word, val in idfDict.items():
        idfDict[word] = math.log10(N / (float(val) + 1))
        
    return(idfDict)

def computeTFIDF(tfBow, idfs):
    tfidf = {}
    for word, val in tfBow.items():
        tfidf[word] = val*idfs[word]
    return(tfidf)

def cosine_similarity(first_sentence,second_sentence):
    first_sentence = first_sentence.split(" ")
    second_sentence = second_sentence.split(" ")#join them to remove common duplicate words
    total= set(first_sentence).union(set(second_sentence))
    
    wordDictA = dict.fromkeys(total, 0) 
    wordDictB = dict.fromkeys(total, 0)
    for word in first_sentence:
        wordDictA[word]+=1
    
    for word in second_sentence:
        wordDictB[word]+=1
        
        
    tfFirst = computeTF(wordDictA, first_sentence)
    tfSecond = computeTF(wordDictB, second_sentence)
    
    idfs = computeIDF([wordDictA, wordDictB])
    
    idfFirst = computeTFIDF(tfFirst, idfs)
    idfSecond = computeTFIDF(tfSecond, idfs)
    
    x = np.array(list(idfFirst.values()))
    y = np.array(list(idfSecond.values()))
    
    dot_product = np.dot(x, y)
    
    magnitude_x = np.sqrt(np.sum(x**2)) 
    magnitude_y = np.sqrt(np.sum(y**2))
    
    cosine_similarity = dot_product / (magnitude_x * magnitude_y)
    
    return cosine_similarity
#*****************************distance********************************************

def dis_col(df,user):
    return [cosine_similarity(df.loc[i,"Answer"],user.loc[i,"Answer"]) for i in range(len(df))]




def All(df,no_of_question):
    greed()
    l = random_list(len(df),no_of_question)
    df = random_df(df,l)
    
    df2 = Apply(df)
    user = user_df(df)
    user2 = Apply(user)
    
    bye()

    df['user_answer'] = user['Answer']
    df['Cosine'] = dis_col(df2,user2)
    df['result'] = dis_col(df2,user2)
    return df


df = pd.read_excel("practical_data.xlsx")
df = All(df,2)     #second position : number of question

print(df)
final = np.round(np.mean(df['result'].values),4)*100
final
    
    """



# import pyttsx3
# import speech_recognition as sr 

# import pandas as pd
# import numpy as np
# from time import sleep
# import datetime

# import nltk
# from nltk.corpus import stopwords 
# from nltk.stem import WordNetLemmatizer
# from nltk.corpus import wordnet
# from nltk.tokenize import word_tokenize

# from sentence_transformers import SentenceTransformer, util
# model = SentenceTransformer('stsb-roberta-large')

# df = pd.read_excel("practical_data.xlsx")

# def preprocessing(text):
#     global lemmatizer
#     global wordnet_map
#     global cnt 
    
#     STOPWORDS = set(stopwords.words('english'))
#     lemmatizer = WordNetLemmatizer()
#     wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}
    
#     punch = text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))
#     pos_tagged_text = nltk.pos_tag(punch.split())
#     return  " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])

# def random_list(n,count=5):
#     l = list()
#     for i in range(10000):
#         a = np.random.randint(n)
#         l.append(a)
#         l2 = list(set(l))
#         if len(l2)>=count:
#             break
#     return l2

# def random_df(df,l):
#     l1,l2 = list(),list()
#     for i in l:
#         l1.append(df.loc[i,"Question"])
#         l2.append(df.loc[i,"Answer"])
#     d = pd.DataFrame({"Question":l1,"Answer":l2})
#     return d

# def preprocessing(text):
#     global lemmatizer
#     global wordnet_map
#     global PUNCT_TO_REMOVE
    
#     PUNCT_TO_REMOVE = '!"#%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
#     STOPWORDS = set(stopwords.words('english'))
#     lemmatizer = WordNetLemmatizer()
#     wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}
    
#     punch = text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))
#     pos_tagged_text = nltk.pos_tag(punch.split())
#     return  " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])

# def Apply(df):
#     df["Answer"] = df["Answer"].str.lower()
#     df["Answer"] = df["Answer"].apply(lambda text: preprocessing(text))
#     return df

# engine = pyttsx3.init('sapi5')
# voices = engine.getProperty('voices')
# engine.setProperty('voice', voices[1].id)  
# engine.setProperty('rate', 150)
# engine.setProperty('volume',1.0) 
# # engine.runAndWait()
# # engine.stop()

# def speak(audio):
#     engine.say(audio)
#     engine.runAndWait()

# def recognizer(quesstion):  
#     try: 
#         r = sr.Recognizer()
#         with sr.Microphone() as voice:
#             print(quesstion)
#             speak(quesstion)
#             r.adjust_for_ambient_noise(voice, duration=.2) 
#             print("listen.........")
#             audio = r.listen(voice)
#             MyText = r.recognize_google(audio)
#             MyText = MyText.lower()
#             speak("did you say : "+MyText)
#             return MyText
            
#             print("")
#         return MyText
#     except Exception as e:
#         # return 
#         speak("voice not found...")
#         return 'voice not found...'
#     except KeyboardInterrupt:
#         pass
    
    
# def user_df(df):
#     d1,d2 = list(),list()
#     for i in df["Question"]:
#         temp = True
#         while temp:
#             a = recognizer(i)
#             opt = recognizer("\n for next say : Yes \n for repeat question say :  No")
            
#             if "yes" in opt:
#                 d1.append(i)   
#                 d2.append(a)
#                 temp = False
#                 sleep(2)
#                 print("")
#             elif "no" in opt:  
#                 temp = True
#                 sleep(2)
#                 print("")
#             else :
#                 temp = True
#                 sleep(2)
#                 print("")
#     user_ans = pd.DataFrame({"Question":d1,"Answer":d2})
#     return user_ans

# def greed():
#     hour = int(datetime.datetime.now().hour)
#     if hour>=0 and hour<12:
#         speak("Good Morning!")
#         return "Good Morning!"
        
        
#     elif hour>=12 and hour<18:
#         speak("Good Afternoon!")
#         return "Good Afternoon!"
        
#     else:
#         speak("Good Evening!")
#         speak("Welcome to Automated interview system This is HR Round....")
#         return "Good Evening!","Welcome to Automated interview system This is HR Round...."
          
#     # speak("Welcome to Automated interview system This is HR Round....")
#     # return "Welcome to Automated interview system This is HR Round...."
    
    
    
# def cosin_similarity(X,Y):
#     l1 =[]
#     l2 =[]
#     X_list = word_tokenize(X)  
#     Y_list = word_tokenize(Y)
#     global sw
#     sw = stopwords.words('english')
    
#     X_set = {w for w in X_list if not w in sw}  
#     Y_set = {w for w in Y_list if not w in sw} 
    
#     intersection = len(X_set.intersection(Y_set))
    
#     rvector = X_set.union(Y_set)  
#     for w in rvector: 
#         if w in X_set: l1.append(1)
#         else: l1.append(0) 
            
#         if w in Y_set: l2.append(1) 
#         else: l2.append(0) 
            
#     c = 0
#     for i in range(len(rvector)): 
#             c+= l1[i]*l2[i] 
#     cosine = (c / ((float((sum(l1)*sum(l2))**0.5)) + .00000001))*.70
#     return cosine,intersection

# def dis_col(df):
#     dis = []
#     intersecction = []
#     for i in range(len(df)):
#         distance,inter = cosin_similarity(df2.loc[i,"Answer"],user1.loc[i,"Answer"])
#         dis.append(distance)
#         intersecction.append(inter)
#     return (dis),intersecction

# # real cosine
# def cosin1(sentence1,sentence2):
#     embedding1 = model.encode(sentence1, convert_to_tensor=True)
#     embedding2 = model.encode(sentence2, convert_to_tensor=True)
#     cosine_scores = util.pytorch_cos_sim(embedding1, embedding2)
#     return cosine_scores.item()

# def cosin_py1(df):
#     distance = []
#     for i in range(len(df2)):
#         d = cosin1(df2.loc[i,"Answer"],user1.loc[i,"Answer"])
#         distance.append(d)
#     return distance

# def cosin(sentence1,sentence2):
#     embedding1 = model.encode(sentence1, convert_to_tensor=True)
#     embedding2 = model.encode(sentence2, convert_to_tensor=True)
#     cosine_scores = util.pytorch_cos_sim(embedding1, embedding2)
#     return cosine_scores.item()

# def cosin_py(df):
#     distance = []
#     for i in range(len(df2)):
#         d = cosin(df2.loc[i,"Answer"],user1.loc[i,"Answer"])*.70
#         distance.append(d)
#     return distance

# def must(ans,user):
#     temp = 0
#     l = [i.replace('$', '') for i in ans.split( ) if i[0]=='$']
#     l2 = [1 if l[i] in user else 0 for i in range(len(l))]
#     if 1 in  l2: temp = 1
# #     if 0 and 1 in l2: temp = 0
#     return temp

# def check(df):
#     lst1 = list()
#     for i in range(len(df)):  
#         c = must(df2.loc[i,"Answer"],user1.loc[i,"Answer"])
#         lst1.append(c)
#     return lst1  

# def result(df):
#     global cosine1,cosine2,keyword
# #     cosine1 = df['cosine'].values
#     cosine2 = df['cosine2'].values
#     keyword = df['keyword_match'].values
#     l= []
#     for i in range(len(df)):
#         if keyword[i]==1 or keyword[i]==np.nan:
#             sum1 = cosine2[i]+.30
#             l.append(sum1)
#         else:
#             l.append(cosine2[i])
#     return l

# def All(df,no_of_question):
#     greed()
#     global user1,df2
#     threshold = .4
    
#     l = random_list(len(df),no_of_question)
#     df = random_df(df,l)
# #     print(df)
    
#     df2 = Apply(df)
#     user = user_df(df)
#     user1 = Apply(user)
# #     print(user1)
#     dis,intersection = dis_col(df)
#     df['user_answer'] = user['Answer']
#     df['intersection'] = intersection
# #     dis1 = cosin_py(df)
    
#     dis2 = cosin_py1(df)
#     df['real_cosine'] = dis2
    
#     dis1 = cosin_py(df)
#     df['cosine2'] = dis1
#     must = check(df)
#     df["keyword_match"] = must
    
#     result1 = result(df)
#     df['result'] = result1
#     return df

# df = pd.read_excel("practical_data.xlsx")
# df = All(df,1)     #second position : number of question

# print("="*60+"Result"+"="*60)
# print(df)
