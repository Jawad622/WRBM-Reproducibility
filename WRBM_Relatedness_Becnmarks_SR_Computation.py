# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 13:53:26 2023

@author: USER
"""
#nltk.download('wordnet')
#nltk.download("punkt")
#nltk.download("stopwords")
#print('domnload complete......')
from nltk.corpus import wordnet as wn
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import hashedindex
#import time 
#from datetime import timedelta
import nltk
from nltk.corpus import stopwords
import re
from nltk.stem import PorterStemmer

def getToken(text):
    text=text.lower()
    text=re.sub("[!|#|$|%|&|(|)|*|+|,|-|.|/|:|;|<|=|>|?|@|[|\|]|^|_|`|{|}|~]+"," ",text)
    text=re.sub("_"," ",text)
    text=re.sub("-"," ",text)        
    text=re.sub("'"," ",text)
    text=re.sub('"'," ",text)
    text=re.sub('\[\[ '," ",text)
    text=re.sub('[\]|]'," ",text)
    text=re.sub('[0-9]+'," ",text)
    text=re.sub('\n'," ",text)
    text= re.sub(r'[^\x00-\x7f]',r" ",text)
    tokens=nltk.word_tokenize(text)
    filtered = [w for w in tokens if not w in stopwords.words('english')]
    return filtered

def stemedTokens(text):
    tokens=getToken(text)
    stemmer=PorterStemmer()
    stemmed = []
    for item in tokens:
        if len(item) > 1:
            stemmed.append(stemmer.stem(item))
    return stemmed

def get_RelBenchmark(benchMark):# This method returns the Benchmark of specific type.
    df=df_RelBenchmarks.loc[df_RelBenchmarks["benchMark"] == benchMark]
    print(benchMark)
    print(len(df))
    return df

def get_wNet_synset_Gloss(con):
    concept=con.partition('.')[0]
    con=wn.synset(con)
    synGloss=""
    synGloss=con.definition()
    if len(synGloss)==0:
        synGloss = concept
    stem_Gloss=stemedTokens(synGloss)   
    synGloss_tokens = " ".join(stem_Gloss)
    return(synGloss_tokens)

def get_wNet_synset_Example(con):
    concept=con.partition('.')[0]
    con=wn.synset(con)
    example_Gloss=""
    example=con.examples()
    if len(example)==0:
        example_Gloss = concept 
    else:
        example_Gloss=" ".join(example)
    stem_Gloss=stemedTokens(example_Gloss)   
    example_Gloss_tokens = " ".join(stem_Gloss)
    return(example_Gloss_tokens)

def get_wNet_synset_Lemma(con):
    concept=con.partition('.')[0]
    con=wn.synset(con)
    lemmaList=[]
    lemma_Gloss=""
    lemmas = con.lemmas()
    if len(lemmas)==0:
        lemma_Gloss = concept
    else:
        for l in lemmas:
            lemma = l.name()
            lemmaList.append(lemma)
        lemma_Gloss=" ".join(lemmaList)
    stem_Gloss=stemedTokens(lemma_Gloss)   
    lemma_Gloss_tokens = " ".join(stem_Gloss)
    return(lemma_Gloss_tokens)

def get_wNet_holonyms(con):
    concept=con.partition('.')[0]
    con=wn.synset(con)
    HGloss = ""
    holonymsList=[]
    holonyms = con.part_holonyms()
    if len(holonyms)==0:
        holonymsList.append(concept)
    else:
        for holonym in holonyms:
            holonym= holonym.name().split(".")[0].replace('_',' ')
            holonymsList.append(holonym)
    HGloss = " ".join(holonymsList)
    stem_Gloss=stemedTokens(HGloss)   
    Gloss_tokens = " ".join(stem_Gloss)
    return(Gloss_tokens)

def get_wNet_meronyms(con):
    concept=con.partition('.')[0]
    con=wn.synset(con)
    MGloss = ""
    meronymsList=[]
    meronyms = con.part_meronyms()
    if len(meronyms)==0:
        meronymsList.append(concept)
    else:
        for meronym in meronyms:
            meronym= meronym.name().split(".")[0].replace('_',' ')
            meronymsList.append(meronym)
    MGloss = " ".join(meronymsList)
    stem_Gloss=stemedTokens(MGloss)   
    Gloss_tokens = " ".join(stem_Gloss)
    return(Gloss_tokens)

def get_wNet_sister_term(con):
    concept=con.partition('.')[0]
    con=wn.synset(con)
    sister_terms_tokens=""
    sister_terms_Gloss=""
    sisterList=[]
    Hyper_terms = con.hypernyms()
    if len(Hyper_terms)==0:
        sisterList.append(concept)
    else:
        for term in Hyper_terms:
            sister_terms=term.hyponyms()
            if len(sister_terms)==0:
                continue
            else:
                for sister in sister_terms:
                    if (sister==con):
                        continue
                    else:
                        sister=sister.name().split(".")[0].replace('_',' ')
                        sisterList.append(sister)
    sister_terms_Gloss = " ".join(sisterList)
    if len(sister_terms_Gloss)==0:
        sister_terms_Gloss=concept
    stem_Gloss=stemedTokens(sister_terms_Gloss)
    sister_terms_tokens = " ".join(stem_Gloss)
    return (sister_terms_tokens)

def get_wNet_derivationally_term(con):
    concept=con.partition('.')[0]
    con=wn.synset(con)
    formList=[]
    form_Gloss=""
    lemmas=con.lemmas()
    if len(lemmas)==0:
        formList.append(concept)
    else:
        for lemma in lemmas:
            forms = lemma.derivationally_related_forms()
            if len(forms)==0:
                continue
            else:
                for form in forms:
                    form=form.name().split(".")[0].replace('_',' ')
                    formList.append(form)
    form_Gloss=" ".join(formList)
    if len(form_Gloss)==0:
        form_Gloss=concept
    stem_Gloss=stemedTokens(form_Gloss)   
    lemma_Gloss_tokens = " ".join(stem_Gloss)
    return(lemma_Gloss_tokens)

def get_synset_synonyms(orgCon):
    con=str(orgCon)
    synonyms=[]
    for syn in wn.synsets(con):
        for l in syn.lemmas():
            synonyms.append(l.name())
    synonyms_terms = " ".join(synonyms)
    stem_synonyms=stemedTokens(synonyms_terms)
    synonyms_tokens = " ".join(stem_synonyms)
    return (synonyms_tokens)

def get_wNet_Hypo_Hyper_List(con):
    con=wn.synset(con)
    hypoList=[]
    hyperList=[]
    hyponyms=con.hyponyms()
    for h in hyponyms:
        h= h.name().split(".")[0].replace('_',' ')
        hypoList.append(h)
    hypogloss = " ".join(hypoList)
    allhypers= [i for i in con.closure(lambda s:s.hypernyms())]
    for h in allhypers:
        hypers = h.name().split(".")[0].replace('_',' ')
        hyperList.append(hypers)
    hypergloss = " ".join(hyperList)

    Gloss = hypogloss+" "+hypergloss  
    stem_Gloss=stemedTokens(Gloss)   
    Gloss_tokens = " ".join(stem_Gloss)
    return(Gloss_tokens)
    
def get_wNet_Hypo_List(con):
    concept=con.partition('.')[0]
    con=wn.synset(con)
    hypoList=[]
    hyponyms=con.hyponyms()
    if len(hyponyms)==0:
        hypogloss = concept
    else:
        for h in hyponyms:
            h= h.name().split(".")[0].replace('_',' ')
            hypoList.append(h)
        hypogloss = " ".join(hypoList)
    Gloss = hypogloss   
    stem_Gloss=stemedTokens(Gloss)   
    Gloss_tokens = " ".join(stem_Gloss)
    return(Gloss_tokens)

def get_wNet_Hyper_List(con):
    concept=con.partition('.')[0]
    con=wn.synset(con)
    hyperList=[]
    allhypers= [i for i in con.closure(lambda s:s.hypernyms())]
    if len(allhypers)==0:
        hypergloss= concept
    else:
        for h in allhypers:
            hypers = h.name().split(".")[0].replace('_',' ')
            hyperList.append(hypers)
        hypergloss = " ".join(hyperList)

    Gloss = hypergloss   
    stem_Gloss=stemedTokens(Gloss)   
    Gloss_tokens = " ".join(stem_Gloss)
    return(Gloss_tokens)

def get_wNet_Gloss_All_relations_1(wNetCon):
    wNetGloss=(get_wNet_synset_Gloss(wNetCon)+" "+get_wNet_synset_Lemma(wNetCon)
               +" "+get_wNet_synset_Example(wNetCon)+" "+get_wNet_Hypo_List(wNetCon)
               +" "+get_wNet_Hyper_List(wNetCon)+" "+get_wNet_meronyms(wNetCon)
               +" "+get_wNet_holonyms(wNetCon)+" "+get_wNet_sister_term(wNetCon)
               +" "+get_wNet_derivationally_term(wNetCon))
    return (wNetGloss)

def get_wNet_Gloss_SynGloss_2(wNetCon):
    wNetGloss=(get_wNet_synset_Gloss(wNetCon))
    return (wNetGloss)

def get_wNet_Gloss_Lemmas_3(wNetCon):
    wNetGloss=(get_wNet_synset_Lemma(wNetCon))
    return (wNetGloss)

def get_wNet_Gloss_Example_4(wNetCon):
    wNetGloss=(get_wNet_synset_Example(wNetCon))
    return (wNetGloss)

def get_wNet_Gloss_hypers_5(wNetCon):
    wNetGloss=(get_wNet_Hyper_List(wNetCon))
    return (wNetGloss)

def get_wNet_Gloss_hypos_6(wNetCon):
    wNetGloss=(get_wNet_Hypo_List(wNetCon))
    return (wNetGloss)

def get_wNet_Gloss_holo_7(wNetCon):
    wNetGloss=(get_wNet_holonyms(wNetCon))
    return (wNetGloss)

def get_wNet_Gloss_mero_8(wNetCon):
    wNetGloss=(get_wNet_meronyms(wNetCon))
    return (wNetGloss)

def get_wNet_Gloss_der_9(wNetCon):
    wNetGloss=(get_wNet_derivationally_term(wNetCon))
    return (wNetGloss)

def get_wNet_Gloss_sister_10(wNetCon):
    wNetGloss=(get_wNet_sister_term(wNetCon))
    return (wNetGloss)

def get_wNet_Gloss_descriptions_All_11(wNetCon):
    wNetGloss=(get_wNet_synset_Gloss(wNetCon)+" "+get_wNet_synset_Lemma(wNetCon)
               +" "+get_wNet_synset_Example(wNetCon))
    return (wNetGloss)

def get_wNet_Gloss_descriptions_Not_Examples_12(wNetCon):
    wNetGloss=(get_wNet_synset_Gloss(wNetCon)+" "+get_wNet_synset_Lemma(wNetCon))
    return (wNetGloss)
 
def get_wNet_Gloss_hierarchical_relations_13(wNetCon):
    wNetGloss=(get_wNet_Hypo_List(wNetCon)+" "+get_wNet_Hyper_List(wNetCon))
    return (wNetGloss)

def get_wNet_Gloss_non_hierarchical_relations_14(wNetCon):
    wNetGloss=(get_wNet_meronyms(wNetCon)+" "+get_wNet_holonyms(wNetCon)
               +" "+get_wNet_sister_term(wNetCon)+" "+get_wNet_derivationally_term(wNetCon))
    return (wNetGloss)

def get_wNet_Gloss_non_hierarchical_relations_Not_Der_15(wNetCon):
    wNetGloss=(get_wNet_meronyms(wNetCon)+" "+get_wNet_holonyms(wNetCon)
               +" "+get_wNet_sister_term(wNetCon))
    return (wNetGloss)

def get_wNet_Gloss_descriptions_plus_hierarchical_All_16(wNetCon):
    wNetGloss=(get_wNet_synset_Gloss(wNetCon)+" "+get_wNet_synset_Lemma(wNetCon)
               +" "+get_wNet_synset_Example(wNetCon)
               +" "+get_wNet_Hypo_List(wNetCon)+" "+get_wNet_Hyper_List(wNetCon))
    return (wNetGloss)

def get_wNet_Gloss_descriptions_plus_hierarchical_Not_Examlple_17(wNetCon):
    wNetGloss=(get_wNet_synset_Gloss(wNetCon)+" "+get_wNet_synset_Lemma(wNetCon)
               +" "+get_wNet_Hypo_List(wNetCon)+" "+get_wNet_Hyper_List(wNetCon))
    return (wNetGloss)

def get_wNet_Gloss_descriptions_plus_non_hierarchical_All_18(wNetCon):
    wNetGloss=(get_wNet_synset_Gloss(wNetCon)+" "+get_wNet_synset_Lemma(wNetCon)
               +" "+get_wNet_synset_Example(wNetCon)
               +" "+get_wNet_meronyms(wNetCon)+" "+get_wNet_holonyms(wNetCon)
               +" "+get_wNet_sister_term(wNetCon)
               +" "+get_wNet_derivationally_term(wNetCon))
    return (wNetGloss)

def get_wNet_Gloss_descriptions_plus_non_hierarchical_Not_Example_Der_19(wNetCon):
    wNetGloss=(get_wNet_synset_Gloss(wNetCon)+" "+get_wNet_synset_Lemma(wNetCon)
               +" "+get_wNet_meronyms(wNetCon)+" "+get_wNet_holonyms(wNetCon)
               +" "+get_wNet_sister_term(wNetCon))
    return (wNetGloss)

def get_wNet_Gloss_hierarchical_plus_non_hierarchical_All_20(wNetCon):
    wNetGloss=(get_wNet_Hypo_List(wNetCon)+" "+get_wNet_Hyper_List(wNetCon)
               +" "+get_wNet_meronyms(wNetCon)+" "+get_wNet_holonyms(wNetCon)
               +" "+get_wNet_sister_term(wNetCon)+" "+get_wNet_derivationally_term(wNetCon))
    return (wNetGloss)

def get_wNet_Gloss_hierarchical_plus_non_hierarchical_Not_Der_21(wNetCon):
    wNetGloss=(get_wNet_Hypo_List(wNetCon)+" "+get_wNet_Hyper_List(wNetCon)
               +" "+get_wNet_meronyms(wNetCon)+" "+get_wNet_holonyms(wNetCon)
               +" "+get_wNet_sister_term(wNetCon))
    return (wNetGloss)

def get_wNet_Gloss_All_Selected_22(wNetCon):
    wNetGloss=(get_wNet_synset_Gloss(wNetCon)+" "+get_wNet_synset_Lemma(wNetCon)
               +" "+get_wNet_Hypo_List(wNetCon)+" "+get_wNet_Hyper_List(wNetCon)
               +" "+get_wNet_meronyms(wNetCon)+" "+get_wNet_holonyms(wNetCon)
               +" "+get_wNet_sister_term(wNetCon))
    return (wNetGloss)

def get_Gloss_Case1(wNetCon):
    MKRGloss=get_wNet_Gloss_All_relations_1(wNetCon)
    return MKRGloss


def get_Gloss_Case2(wNetCon):
    MKRGloss=get_wNet_Gloss_SynGloss_2(wNetCon)
    return MKRGloss

def get_Gloss_Case3(wNetCon):
    MKRGloss=get_wNet_Gloss_Lemmas_3(wNetCon)
    return MKRGloss

def get_Gloss_Case4(wNetCon):
    MKRGloss=get_wNet_Gloss_Example_4(wNetCon)
    return MKRGloss

def get_Gloss_Case5(wNetCon):
    MKRGloss=get_wNet_Gloss_hypers_5(wNetCon)
    return MKRGloss

def get_Gloss_Case6(wNetCon):
    MKRGloss=get_wNet_Gloss_hypos_6(wNetCon)
    return MKRGloss

def get_Gloss_Case7(wNetCon):
    MKRGloss=get_wNet_Gloss_holo_7(wNetCon)
    return MKRGloss
    
def get_Gloss_Case8(wNetCon):
    MKRGloss=get_wNet_Gloss_mero_8(wNetCon)
    return MKRGloss
    
def get_Gloss_Case9(wNetCon):
    MKRGloss=get_wNet_Gloss_der_9(wNetCon)
    return MKRGloss
    
def get_Gloss_Case10(wNetCon):
    MKRGloss=get_wNet_Gloss_sister_10(wNetCon)
    return MKRGloss
    
def get_Gloss_Case11(wNetCon):
    MKRGloss=get_wNet_Gloss_descriptions_All_11(wNetCon)
    return MKRGloss
    
def get_Gloss_Case12(wNetCon):
    MKRGloss=get_wNet_Gloss_descriptions_Not_Examples_12(wNetCon)
    return MKRGloss
    
def get_Gloss_Case13(wNetCon):
    MKRGloss=get_wNet_Gloss_hierarchical_relations_13(wNetCon)
    return MKRGloss
def get_Gloss_Case14(wNetCon):
    MKRGloss=get_wNet_Gloss_non_hierarchical_relations_14(wNetCon)
    return MKRGloss
    
def get_Gloss_Case15(wNetCon):
    MKRGloss=get_wNet_Gloss_non_hierarchical_relations_Not_Der_15(wNetCon)
    return MKRGloss
    
def get_Gloss_Case16(wNetCon):
    MKRGloss=get_wNet_Gloss_descriptions_plus_hierarchical_All_16(wNetCon)
    return MKRGloss
    
def get_Gloss_Case17(wNetCon):
    MKRGloss=get_wNet_Gloss_descriptions_plus_hierarchical_Not_Examlple_17(wNetCon)
    return MKRGloss
    
def get_Gloss_Case18(wNetCon):
    MKRGloss=get_wNet_Gloss_descriptions_plus_non_hierarchical_All_18(wNetCon)
    return MKRGloss
    
def get_Gloss_Case19(wNetCon):
    MKRGloss=get_wNet_Gloss_descriptions_plus_non_hierarchical_Not_Example_Der_19(wNetCon)
    return MKRGloss
    
def get_Gloss_Case20(wNetCon):
    MKRGloss=get_wNet_Gloss_hierarchical_plus_non_hierarchical_All_20(wNetCon)
    return MKRGloss
    
def get_Gloss_Case21(wNetCon):
    MKRGloss=get_wNet_Gloss_hierarchical_plus_non_hierarchical_Not_Der_21(wNetCon)
    return MKRGloss
    
def get_Gloss_Case22(wNetCon):
    MKRGloss=get_wNet_Gloss_All_Selected_22(wNetCon)
    return MKRGloss


def get_Gloss(wNetCon,Case):
    if Case==1:
        GlossTokens=get_Gloss_Case1(wNetCon)
    elif Case==2:
        GlossTokens=get_Gloss_Case2(wNetCon)
    elif Case==3:
        GlossTokens=get_Gloss_Case3(wNetCon)
    elif Case==4:
        GlossTokens=get_Gloss_Case4(wNetCon)
    elif Case==5:
       GlossTokens=get_Gloss_Case5(wNetCon)
    elif Case==6:
        GlossTokens=get_Gloss_Case6(wNetCon)
    elif Case==7:
        GlossTokens=get_Gloss_Case7(wNetCon)
    elif Case==8:
        GlossTokens=get_Gloss_Case8(wNetCon)
    elif Case==9:
        GlossTokens=get_Gloss_Case9(wNetCon)
    elif Case==10:
        GlossTokens=get_Gloss_Case10(wNetCon)
    elif Case==11:
        GlossTokens=get_Gloss_Case11(wNetCon)
    elif Case==12:
        GlossTokens=get_Gloss_Case12(wNetCon)
    elif Case==13:
        GlossTokens=get_Gloss_Case13(wNetCon)
    elif Case==14:
        GlossTokens=get_Gloss_Case14(wNetCon)
    elif Case==15:
        GlossTokens=get_Gloss_Case15(wNetCon)
    elif Case==16:
        GlossTokens=get_Gloss_Case16(wNetCon)
    elif Case==17:
        GlossTokens=get_Gloss_Case17(wNetCon)
    elif Case==18:
        GlossTokens=get_Gloss_Case18(wNetCon)
    elif Case==19:
        GlossTokens=get_Gloss_Case19(wNetCon)
    elif Case==20:
        GlossTokens=get_Gloss_Case20(wNetCon)
    elif Case==21:
        GlossTokens=get_Gloss_Case21(wNetCon)
    elif Case==22:
        GlossTokens=get_Gloss_Case22(wNetCon)
    return GlossTokens
        
def get_GlossTfidfIndex(wNetCon1,wNetCon2,Case):
    gloss1=get_Gloss(wNetCon1,Case)
    gloss2=get_Gloss(wNetCon2,Case)
    docs=[]
    docs.append(gloss1)
    docs.append(gloss2)
    hindex = hashedindex.HashedIndex()
    i=1
    for d in docs:
        for term in d.split():
            hindex.add_term_occurrence(term,i)
        i+=1
    return hindex

def get_Term_Weight(term,doc, index):
    weight=[]
    df=get_GlossTermDF(term)
    try:
        tf=index.get_term_frequency(term,doc,False)
    except IndexError:
        tf=1.0
        print("warning....")
    weight= tf * np.log2((N)/df) 
    return weight

def get_WordSet(wNetCon1,wNetCon2,Case):
    gloss1=set(get_Gloss(wNetCon1,Case).split())
    gloss2=set(get_Gloss(wNetCon2,Case).split())
    wordCom=gloss1&gloss2
    diff1=gloss1-wordCom
    diff2=gloss2-wordCom
    gloss1=removeRareWordsFromGloss(diff1)|wordCom
    gloss2=removeRareWordsFromGloss(diff2)|wordCom
    wordSet = gloss1|gloss2
    if (len(gloss1)==0 or len(gloss2)==0):
        gloss1=set(get_Gloss(wNetCon1,Case).split())
        gloss2=set(get_Gloss(wNetCon2,Case).split())
        wordSet = gloss1|gloss2
    return gloss1, gloss2, wordSet

def removeRareWordsFromGloss(gloss):
    newGloss=[]
    for w in gloss:
        f = get_GlossTermDF(w)
        if (f > kfreq ):
            newGloss.append(w)
    return set(newGloss)

def get_GlossTermDF(term):
    f=0
    df=df_dstokenic.loc[df_dstokenic["term"]==term]
    for i, r in df.iterrows():
        f= r["df"]
    if (f > 0):
        return f
    else:
        return 1 

def get_GlossSim(wNetCon1,wNetCon2, index, Case):
    gloss1,gloss2,wordSet=get_WordSet(wNetCon1,wNetCon2,Case)
    wordSetList=[]
    vecList=[]
    for w in wordSet:
        wordSetList.append(w)
        vecList.append(w)
    vecDictList1=dict.fromkeys(vecList,0.0)
    vecDictList2=dict.fromkeys(vecList,0.0)
    vec1=get_Vector(gloss1,vecDictList1, index,1)
    vec2=get_Vector(gloss2,vecDictList2, index,2)
    v1=[]
    v2=[]
    for v in vec1:
        v1.append(vec1[v])
    for v in vec2:
        v2.append(vec2[v])
    a1=np.array(v1)
    a2=np.array(v2)
    v11=a1.reshape(1,len(a1))
    v22=a2.reshape(1,len(a2))
    return cosine_similarity(v11,v22)

def get_Vector(gloss, dict1, index,doc):
    wic=0.0
    for w in gloss:
        wic=get_Term_Weight(w,doc,index)
        dict1[w]=np.round(wic,3)
    return dict1

N=5595607
Max_termCount=1192147
kfreq=200
#[353All, MTurk287, MTurk771, MEN3000]
benchMark='353All'
index=hashedindex.HashedIndex()

df_dstokenic = pd.read_csv("df_AlldsTokensDF.txt",sep="\t")
df_RelBenchmarks=pd.read_csv("Relatedness/dfNew_Disamb_MKR_benchMarks_Rel.txt",sep="\t")

df=get_RelBenchmark(benchMark)


with open ("WLFM_Results/"+benchMark+".txt","w",encoding="utf-8") as flw:

    dataline="rid\tWNCon1\tWNCon2\torgCon1\torgCon2\tHJ\tCase1(All)\tCase2(G)\tCase3(L)\tCase4(Ex)\tCase5(Hr)\tCase6(Hp)\tCase7(Ho)\tCase8(M)\tCase9(Der)\tCase10(Sis)\tCase11(D)\tCase12(D(NEx))\tCase13(H)\tCase14(NH)\tCase15(NH(NDer))\tCase16(DH)\tCase17(DH(NEx))\tCase18(DNH)\tCase19(DNH(NEx,Der))\tCase20(HNH)\tCase21(HNH(ND))\tCase22(S)"    
    flw.write(dataline+"\n")
    for i,row in df.iterrows():
        rid=row["rid"]
        wNetCon1=row["wNetCon1"]
        wNetCon2=row["wNetCon2"]
        orgCon1=row["orgCon1"]
        orgCon2=row["orgCon2"]
        hj=row["HJ"]
        if(wNetCon1==wNetCon2):
            sim1=1.0
            sim2=1.0
            sim3=1.0
            sim4=1.0
            sim5=1.0
            sim6=1.0
            sim7=1.0
            sim8=1.0
            sim9=1.0
            sim10=1.0
            sim11=1.0
            sim12=1.0
            sim13=1.0
            sim14=1.0
            sim15=1.0
            sim16=1.0
            sim17=1.0
            sim18=1.0
            sim19=1.0
            sim20=1.0
            sim21=1.0
            sim22=1.0
        else:
            print("Calculating Com1......")
            index=get_GlossTfidfIndex(wNetCon1, wNetCon2,1)
            sim1=round(get_GlossSim(wNetCon1,wNetCon2, index,1)[0][0],3)
            
            print("Calculating Com2......")
            index=get_GlossTfidfIndex(wNetCon1, wNetCon2,2)
            sim2=round(get_GlossSim(wNetCon1,wNetCon2, index,2)[0][0],3)
            
            print("Calculating Com3......")
            index=get_GlossTfidfIndex(wNetCon1, wNetCon2,3)
            sim3=round(get_GlossSim(wNetCon1,wNetCon2, index,3)[0][0],3)
            
            print("Calculating Com4......")
            index=get_GlossTfidfIndex(wNetCon1, wNetCon2,4)
            sim4=round(get_GlossSim(wNetCon1,wNetCon2, index,4)[0][0],3)
            
            print("Calculating Com5......")
            index=get_GlossTfidfIndex(wNetCon1, wNetCon2,5)
            sim5=round(get_GlossSim(wNetCon1,wNetCon2, index,5)[0][0],3)
            
            print("Calculating Com6......")
            index=get_GlossTfidfIndex(wNetCon1, wNetCon2,6)
            sim6=round(get_GlossSim(wNetCon1,wNetCon2, index,6)[0][0],3)
            
            print("Calculating Com7......")
            index=get_GlossTfidfIndex(wNetCon1, wNetCon2,7)
            sim7=round(get_GlossSim(wNetCon1,wNetCon2, index,7)[0][0],3)
            
            print("Calculating Com8......")
            index=get_GlossTfidfIndex(wNetCon1, wNetCon2,8)
            sim8=round(get_GlossSim(wNetCon1,wNetCon2, index,8)[0][0],3)
            
            print("Calculating Com9......")
            index=get_GlossTfidfIndex(wNetCon1, wNetCon2,9)
            sim9=round(get_GlossSim(wNetCon1,wNetCon2, index,9)[0][0],3)
            
            print("Calculating Com10......")
            index=get_GlossTfidfIndex(wNetCon1, wNetCon2,10)
            sim10=round(get_GlossSim(wNetCon1,wNetCon2, index,10)[0][0],3)
            
            print("Calculating Com11......")
            index=get_GlossTfidfIndex(wNetCon1, wNetCon2,11)
            sim11=round(get_GlossSim(wNetCon1,wNetCon2, index,11)[0][0],3)
            
            print("Calculating Com12......")
            index=get_GlossTfidfIndex(wNetCon1, wNetCon2,12)
            sim12=round(get_GlossSim(wNetCon1,wNetCon2, index,12)[0][0],3)
            
            print("Calculating Com13......")
            index=get_GlossTfidfIndex(wNetCon1, wNetCon2,13)
            sim13=round(get_GlossSim(wNetCon1,wNetCon2, index,13)[0][0],3)
            
            print("Calculating Com14......")
            index=get_GlossTfidfIndex(wNetCon1, wNetCon2,14)
            sim14=round(get_GlossSim(wNetCon1,wNetCon2, index,14)[0][0],3)
            
            print("Calculating Com15......")
            index=get_GlossTfidfIndex(wNetCon1, wNetCon2,15)
            sim15=round(get_GlossSim(wNetCon1,wNetCon2, index,15)[0][0],3)
            
            print("Calculating Com16......")
            index=get_GlossTfidfIndex(wNetCon1, wNetCon2,16)
            sim16=round(get_GlossSim(wNetCon1,wNetCon2, index,16)[0][0],3)
            
            print("Calculating Com17......")
            index=get_GlossTfidfIndex(wNetCon1, wNetCon2,17)
            sim17=round(get_GlossSim(wNetCon1,wNetCon2, index,17)[0][0],3)
            
            print("Calculating Com18......")
            index=get_GlossTfidfIndex(wNetCon1, wNetCon2,18)
            sim18=round(get_GlossSim(wNetCon1,wNetCon2, index,18)[0][0],3)
            
            print("Calculating Com19......")
            index=get_GlossTfidfIndex(wNetCon1, wNetCon2,19)
            sim19=round(get_GlossSim(wNetCon1,wNetCon2, index,19)[0][0],3)
            
            print("Calculating Com20......")
            index=get_GlossTfidfIndex(wNetCon1, wNetCon2,20)
            sim20=round(get_GlossSim(wNetCon1,wNetCon2, index,20)[0][0],3)
            
            print("Calculating Com21......")
            index=get_GlossTfidfIndex(wNetCon1, wNetCon2,21)
            sim21=round(get_GlossSim(wNetCon1,wNetCon2, index,21)[0][0],3)
            
            print("Calculating Com22......")
            index=get_GlossTfidfIndex(wNetCon1, wNetCon2,22)
            sim22=round(get_GlossSim(wNetCon1,wNetCon2, index,22)[0][0],3)
            
        dataline=(str(rid)+"\t"+str(wNetCon1)+"\t"+str(wNetCon2)+"\t"
              +str(orgCon1)+"\t"+str(orgCon2)+"\t"+str(hj)+"\t"+str(sim1)+"\t"+str(sim2)+"\t"+str(sim3)
              +"\t"+str(sim4)+"\t"+str(sim5)+"\t"+str(sim6)+"\t"+str(sim7)+"\t"+str(sim8)+"\t"+str(sim9)+"\t"+str(sim10)
              +"\t"+str(sim11)+"\t"+str(sim12)+"\t"+str(sim13)+"\t"+str(sim14)+"\t"+str(sim15)+"\t"+str(sim16)+"\t"+str(sim17)
              +"\t"+str(sim18)+"\t"+str(sim19)+"\t"+str(sim20)+"\t"+str(sim21)+"\t"+str(sim22))
        
        print(dataline)
        flw.write(dataline+"\n")
    flw.close()









