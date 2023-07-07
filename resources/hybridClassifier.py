# -*- coding: utf-8 -*-
"""
created by Liz Workman 2023

This is a generalized version of a program that will classify files of snippets
containing opioid key phrases.  It includes support functions.
It first classifies the snippets using regular expressions and a voting system;
remaining snippets are classified using a trained support vector machine
"""

import re
import csv
import pickle
import numpy as np
from collections import defaultdict
from datetime import datetime
from os import listdir
from os.path import isfile, join
now = datetime.now()
currentDateString = now.strftime("%Y-%m-%d")





# a sample stoplist, in case prebuilt resources are not available
stoplist = ['up','down','for','a','of','the','and','to','to','in','by','is','has','an','was',
'was','not','or','wasn\'t','by','as','on','are','in','with','no','jan','nov','feb',
'mar','apr','may','jun','jul','aug','sep','sept','oct','nov','dec','dr','it','be',
'two','first','second','third','fourth','fifth','sixth','seventh','eighth','ninth',
'tenth','eleventh','twelveth','thirteenth','fourteenth','fifteenth','sixteenth',
'seventeenth','eighteenth','nineteenth','twentieth','one','two','three','four','five',
'six','seven','eight','nine','ten','eleven','twelve','thirteen','fourteen','fifteen',
'sixteen','seventeen','eighteen','nineteen','twenty','etc','about','he','his',
'her','hers','their','theirs','us','ours','ours','yours','your']

# target words here
targetString = 'abstral,actiq,demerol,dependence,dilaudid,dolophine,duragesic,exalgo,fentanyl,fentora,hydrocodone,hydromorphone,hysingla,kadian,lorcet,lortab,meperidine,methadone,methadose,morphine,norco,opiate,opioid abuse,opioid,oxaydo,oxycodone,oxycontin,percocet,roxicet,vicodin,withdrawal,zohydro,opioid dependence,polysubstance abuse,substance abuse,substance dependence'

targets = targetString.split(",")


keyPhraseKeyNumericValue = defaultdict(int)

startN = 1
for kp in targets:
    keyPhraseKeyNumericValue[kp] = startN
    startN += 1




#------------------------------------------------------------------
# cleanString: a convenience text preprocessing function
# customize as needed
#------------------------------------------------------------------
  
def cleanString(s):
    s = s.lower()
    s = re.sub('\n',' ',s) 
    s = re.sub('\r',' ',s)
    s = re.sub(',',' ',s)
    s = re.sub(';',' ',s)
    s = re.sub('\t',' ',s)
    s = re.sub('\s+',' ',s)
    s = re.sub('"','',s)
    txtS = s
    txtS = txtS.replace("\\",' ')
    txtS = txtS.replace("/",' ')
    txtS = txtS.replace(","," ")
    txtS = txtS.replace(";"," ")
    txtS = re.sub("\[\s\]","[]",txtS) 
    txtS = re.sub("\(\s\)","()",txtS)                 
    txtS = re.sub("[-]{3,}","",txtS)   
    txtS = re.sub("[_]{3,}","",txtS)
    txtS = re.sub("[=]{3,}","",txtS) 
    txtS = re.sub("[*]{3,}","",txtS)
    
    parts = txtS.split()
    for p in parts:
        if p == ' ':
            parts.remove(p)
           
    if len(parts) > 0:
        tempString = " ".join(map(str, parts))
        newString = tempString
    else:
        newString = "empty string"
    return newString
    


#------------------------------------------------------------------
# findTemplatePatterns: a function to identify regex patterns
# customize as needed
#------------------------------------------------------------------
          
def findTemplatePatterns(d,s):  
    s = s.lower()
    s = s.replace("'s","")    
    s = s.replace('"','')
    s = s.replace("'","")
    

    # neutral patterns
    mpatterns = ['.*7e\.[\s|\t]+opiates\s+\(*heroin,*\s+.*fentanyl,*[etc]*\.*\?*.*not\s+asked','.*7e\.[\s|\t]+opiates\s+\(*heroin,*\s+.*fentanyl,*[etc]*\.*\?\W+none',
     '.*7e\.[\s|\t]+opiates\s+\(*heroin,*\s+.*fentanyl,*[etc]*\.*\?*\W+0','.*7e\.[\s|\t]+opiates\s+\(*heroin,*\s+.*fentanyl,*[etc]*\.*\?*\W+7f',
     'cocaine\s+dependence', 'alcohol\s+dependence', 'the\s+total\s+score\s+may\s+range\s+from\s+.*lower\s+score\s+indicates\s+a\s+higher\s+level\s+of\s+dependence',
    'nicotine\s+dependence','tobacco\s+dependence','dependence\s+on\s+supplemental\s+oxygen','etoh\s+dependence',
    'dependence\s+on\s+ventilator','continuous\s+drinking\s+behavior','\d{1,2}\)\s+fentanyl',
    'medications:.*:','d{1,2}\)\s+non\-va\s+non\s+va\s+med.*d{1,2}\)','illicit\s+drug\s+\(heroin,\scocaine,\sivdu\)\-*denies',
    'cannabis\s+dependence','tube\s+dependence','alc\s+dependence','marijuana\s+dependence','non\-va\s+non\s+va\s+med','\s+non\-va\s+','allergy:','meds\:',
    '\d{1,2}\)\s*hydrocodone','\[\s*x*\s*\]\s*discussion\s+\[\s*x*\s*\]\s*demonstration\s*\[\s*x*\s*\]\s*handout','\d{1,2}\).*morphine.*\d{1,2}\)','\d{1,2}\).*fentanyl.*\d{1,2}\)','\d{1,2}\).*hydromorphone.*\d{1,2}\)','\d{1,2}\).*methado.*\d{1,2}\)',
    '\d{1,2}\).*oxycodone.*\d{1,2}\)','\d{1,2}\).*hydromorphone.*\d{1,2}\)','pain\s+medication\(s\)\:','alternative\s+pain\s+relief\s+measures\s+offered\:',
    '\s+(allergies|allergic|allergic\s+to)[\s\W]+(?:\w+\W+^except){0,8}?(demerol|opioids|oxycodone|dilaudid|duragesic|exalgo|fentanyl)\s+',
    '\s+(allergies|allergic|allergic\s+to)[\s\W]+(?:\w+\W+^except){0,8}?(fentora|hydrocodone|hydromorphone|hysingla|kadian|lorcet|lortab|meperidine|methadone)\s+',
    '\s+(allergies|allergic|allergic\s+to)[\s\W]+(?:\w+\W+^except){0,8}?(methadose|morphine|norco|opiate|oxycodone|oxycondone|oxycontin|percocet|roxicet|vicodin|zohydro)\s+',
    '\s+(allergies\s+(&|and)\s+sensitivities\:*\s+)[\s\W]+(?:\w+\W+^except){0,8}?(demerol|opioids|oxycodone|dilaudid|duragesic|exalgo|fentanyl)\s+',
    '\s+(allergies\s+(&|and)\s+sensitivities\:*\s+)[\s\W]+(?:\w+\W+^except){0,8}?(fentora|hydrocodone|hydromorphone|hysingla|kadian|lorcet|lortab|meperidine|methadone)\s+',
    '\s+(allergies\s+(&|and)\s+sensitivities\:*\s+)[\s\W]+(?:\w+\W+^except){0,8}?(methadose|morphine|norco|opiate|oxycodone|oxycondone|oxycontin|percocet|roxicet|vicodin|zohydro)\s+',
    'substance\s+abuse\s+or\s+dependence\s+\-+\s+no','substance\s+abuse\s+or\s+dependence\s+\?+\s+no','other\s+substance\s+dependence',
    'this[\s\W]+agreement[\s\W]+is[\s\W]+used[\s\W]+with[\s\W]+all[\s\W]+patients\W+who\W+receive\W+opioid\W+medicines\W+regularly',
    'this\W+agreement\W+does\W+not\W+mean\W+that\W+my\W+provider\W+believes\W+I\W+have\W+abused[\s\W]+or\W+will\W+abuse\W+these\W+medicines',
    'treatment\W+with\W+the\W+opioid\W+medicines\W+may\W+be\W+stopped\W+if',
    'opioid\W+medicine\W+is\W+only\W+one\W+part\W+of\W+my\W+overall\W+treatment\W+plan',
    'opioid\W+medicines\W+can\W+have\W+serious\W+side\W+effects',
    'that\W+are\W+further\W+explained\W+in\W+the\W+attached\W+information\W+sheet',
    'these\W+side\W+effects\W+include\W+constipation,*\W+drowsiness,*\W+nausea,*\W+and\W+itching',
    'i\W+cannot\W+drive\W+or\W+operate\W+machinery\W+whenever\W+i\W+may\W+be\W+sleepy\W+or\W+drowsy',
    'taking\W+too\W+much\W+opioid\W+medicine,*\W+or\W+mixing\W+opioids', 
    'with\W+mood\W+altering\W+substances,*\W+sleeping\W+pills\W+or\W+alcohol',
    'there\W+may\W+be\W+long\-*term\W+risks\W+with\W+taking\W+opioid\W+medicines\W+regularly',
    'regular\W+use\W+of\W+opioid\W+medicines\W+may\W+over\W+time\W+increase\W+my\W+sensitivity\W+t\W+pain',
    'the\W+plan\W+may\W+include\W+diagnostic\W+tests,*\W+physical\W+therapy,*\W+visits\W+with\W+other\W+doctors,*\W+and\W+medicines\W+other\W+than\W+opioids',
    'i\W+will\W+not\W+ask\W+for\W+opioid\W+medicine\W+for\W+my\W+chronic\W+pain\W+condition\W+from',
    'if\W+i\W+receive\W+an\W+opioid\W+prescription\W+from\W+another\W+physician,*\W+i\W+will',
    'i\W+will\W+take\W+my\W+opioid\W+pain\W+medicine\W+as\W+prescribed',
    '\s+(percocet|ms\s+contin|morphine|oxycontin|vicodin|dilaudid|methadone|duragesic|duragesic\s+patch)\W+(?:\w+\W+){0,4}?(never)',
     '\s+(sibs|siblings|mother|father|sister|brother|son|daughter|niece|nephew|friend)\W+(?:\w+\W+){0,4}?(substance\s+abuse|substance\s+dependence|polysubstance\s+abuse|polysubstance\s+dependence)',
    '\s+(sibs|siblings|mother|father|sister|brother|son|daughter|niece|nephew|friend)\W+(?:\w+\W+){0,4}?(opioid\s+abuse|opioid\s+dependence|opiate\s+abuse|opiate\s+dependence)',
    '\s+(stopped|quit|gave up)\W+(?:\w+\W+){0,4}?(heroin|heroine|opioid|opiate)']

    # general positive patterns
    pospatterns = ['polysubstance\s+drug\s+abuse','polysubstance\s+abuse','polysubstance\s+dependence','polysubstance\s+drug\s+dependence',
                   '(opioid|opiate)\s+drug\s+dependence','(opioid|opiate|opiod|opoid)\s+dependence','opioid\s+abuse','opioid\s+type\s+dependence','opioid\s*\-\s*type\s+dependence']
               
    # absolutely positive patterns
    abspospatterns = ['\)\W+longstanding\W+opioid','axis\s+(i|1)\:*\W+(?:\w+\W+){0,10}?opioid',
                      '(opioid\s+abuse|opiod\s+abuse|opoid\s+abuse|opiate\s+abuse)\W+(?:\w+\W+;+){0,7}?(relapse|relapsed|relapses|detox)',
                      'opioid\W+(?:\w+\W+){0,7}?relaps','opiate\W+(?:\w+\W+){0,7}?relaps','longstanding\W+opioid','longstanding\W+opiate','longstanding\W+opiod','longstanding\W+opoid','long\s+standing\W+opiod'
                      'opiod\W+(?:\w+\W+){0,7}?relaps','opoid\W+(?:\w+\W+){0,7}?relaps','polysubstance\W+(?:\w+\W+){0,7}?relaps',
                      'opiod\W+(?:\w+\W+){0,7}?detox','opoid\W+(?:\w+\W+){0,7}?detox','polysubstance\W+(?:\w+\W+){0,7}?detox','opioid\W+(?:\w+\W+){0,7}?detox','opiate\W+(?:\w+\W+){0,7}?detox',
                     '(opioid\s+dependence|opiod\s+dependence|opoid\s+dependence|opiate\s+dependence)\W+(?:\w+\W+;+){0,7}?(relapse|relapsed|relapses|detox)'
                      '\s+(caution|given)\W+(?:\w+\W+){0,10}?(hx|history|h\.o\.|h\\o|h/o)',
                    '\s+if\s+the\s+patient\s+screened\s+postive\s+for\s+tobacco,*\s+alcohol,*\s+opioid,*\s+or\s+illicit\s+substance\s+dependence\s+or\s+abuse\s+within\s+the\s+past\s+30\s+days,*\s+are\s+they\s+interested\s+in\s+a\s+referral\s+for\s+treatment\?\s*yes\W+(?:\w+\W+){0,7}?(opioid|opiate)',
                    '\s+continues\s+to\s+struggle\W+(?:\w+\W+){0,10}?(substance|opioid|opiate)',
                    'substance\s+dependence\s*\(\s*snomed','substance\s+dependence\s*\(\s*icd','opioid\s+use\s+disorder\s*\(\s*icd','substance\s+dependence\s*\(\s*sct',
                     'opioid\s+abuse\s*\(\s*icd', '\W+(?<!of)\sopioid\s+abuse\s*\(\s*snomed','\W+(?<!of)\sopioid\s+abuse\s*\(\s*sct',  
                     'opioid\s+dependence\s*\(\s*icd', '\W+(?<!of)\sopioid\s+dependence\s*\(\s*snomed','\W+(?<!of)\sopioid\s+dependence\s*\(\s*sct',
                    '\)\s+substance\s+abuse\s+\(\s*icd','\)\s+substance\s+abuse\s+\(\s*snomed','\)\s+substance\s+abuse\s+\(\s*sct','substance\s+abuse\s+or\+dependence\:*\s*\(\s*ongoing\s+level\s+of\s+use\s+significantly\s+interferes\s+with\s+ability\s+to\s+function\s*\)\s*yes',                                                                                                                                        
                    '\[\s*x\s*\]\s*substance\s+abuse','\[\s*X\s*\]\s*substance\s+abuse','\[\s*\]\s+psychosis\s+\[\s*x\s*\]\s+substance\s+abuse\s+or\s+dependence\s+\[\s*\]\s+severe\s+mood\s+disorder',
                    'substance\s+abuse/dependence\s+relapse\s+','\s+(longstanding|long\s+standing|long\-standing)\W+(?:\w+\W+){0,5}?(polysubstance|opioid|opiate|)\W+(?:\w+\W+){0,2}?(abuse)',
                    'limitations\:\W+(?:\w+\W+){0,7}?(substance|polysubstance|opioid|opiate)\W+(?!\s+in\s+remission)',
                    'recurrent\s+substance\s+dependence',
                    '(?<!history\sof)\s+heroin\s+dependence', # 'heroin dependence', not including 'history of heroin dependence'
                    '(?<!history\sof)\s+heroine\s+dependence',
                    '\s+substance\s+dependence\:\s*(?:\w+\W+){0,7}?(opioids|opioid|opoid|opiod|opiate|heroin)']      
 
    #canceling patterns
    cnegpatterns = ['(hx|history\s+of|h\.o\.|h\\o|h/o)\W+(?:\w+\W+){0,4}?(polysubstance|opioid|opiate)','\s+hx[\s\W]+(?:\w+\W+){0,4}?(heroin|opioids|oxycodone|polysubstance)\s+',  
                    '\s+(hx|history|h\.o\.|h\\o|h/o)\W+(?:\w+\W+){0,7}?(demerol|dilaudid|duragesic|exalgo|fentanyl|heroin|polysubstance|opioid)\s+',
                     '\s+(hx|history|h\.o\.|h\\o|h/o)[\s\W]+(?:\w+\W+){0,7}?(fentora|hydrocodone|hydromorphone|hysingla|kadian|lorcet|lortab|meperidine|methadone)\s+',
                    '\s+(hx|history|h\.o\.|h\\o|h/o)[\s\W]+(?:\w+\W+){0,7}?(methadose|morphine|norco|opiate|oxycodone|oxycondone|oxycontin|percocet|roxicet|vicodin|zohydro)\s+','former\W+opioid\W+abuse',
                    'former\W+opioid\W+dependence','(polysubstance|opioid|opiod|opoid)\W+(?:\w+\W+){0,10}?remission',
                    'h/o\W+(?:\w+\W+){0,7}?opioid','h/o\W+(?:\w+\W+){0,7}?opioids','h/o\W+(?:\w+\W+){0,7}?opiate',
                    'no\s+evidence\s+of\s+substance\s+abuse',
                    '\s+(no\s+evidence)\W+(?:\w+\W+){0,8}?(illicit|substance\s+abuse)',
                    '(past)\W+(?:\w+\W+){0,10}?(polysubstance|opioid|opiate|substance\s+abuse)',
                    '(hx|history\s+of|h\.o\.|h\\o|h/o)\W+(?:\w+\W+){0,4}?(substance\s+abuse|substance\s+dependence|polysubstance\s+abuse|polysubstance\s+dependence)',
                   '(hx|history\s+of|h\.o\.|h\\o|h/o)\W+(?:\w+\W+){0,4}?(opioid\s+abuse|opioid\s+dependence|opiate\s+abuse|opiate\s+dependence)',
                    'did\s+not\s+screen\s+positive\s+for\s+tobacco,*\s+alcohol,*\s+opioid,*\s+or\s+illicit\s+substance\s+dependence\s+or\s+abuse\s+within\s+the\s+past\s+30\s+days',
                    'denies\s+substance\s+abuse\:*','denies\s+substance\s+use\s+history','\s+substance\s+abuse\:*\s*\(\s*\)\s*yes\s*\(\s*x\s*\)\s*no',
                    'substance\s+abuse\s+or\+dependence\:*\s*\(\s*ongoing\s+level\s+of\s+use\s+significantly\s+interferes\s+with\s+ability\s+to\s+function\s*\)\s*no',
                    'ability\s+to\s+maintain\s+independence\s+in\s+the\s+community','capable\s+of\s+increased\s+independence','\[\s*\]\s+psychosis\s+\[\s*\]\s+substance\s+abuse\s+or\s+dependence\s+\[\s*\]\s+severe\s+mood\s+disorder'
                    ] # cancels positive patterns



    yup = 0
    nope = 0
    cancelVote = 0
    absolutely = 0
    s = s.lower()
    for m in mpatterns:
        if re.search(m,s):
            nope += 1
    for p in pospatterns:
        if re.search(p,s):
            yup += 1           
    for c in cnegpatterns:
        if re.search(c,s):
            cancelVote += 1 
    for a in abspospatterns:
        if re.search(a,s):
            absolutely += 1
    if absolutely > 0:
        return 'pos pattern'
    elif cancelVote > 0:
        return 'neg pattern'  
    elif yup > 0: 
        return 'pos pattern'
    elif nope > 0:        
        return 'neg pattern'
#    return 'no pattern' 
    return ''    
#------------------------------------------------------------------
# getKeyPhraseIndex gets the keyphrase index 
#------------------------------------------------------------------ 

def getKeyPhraseIndex(v,kw):
    ret=0
    kw = kw.strip(' " ')
    if  kw.isalpha():
        if kw in v:
            ret= v.find(kw)
            return ret,kw

    return ret,""



# load the trained SVM model
svm = pickle.load(open('SVM.sav','rb'))


#get ngram bag-of-words features 
featurefilepath='ngram_features.txt'  
uniqueBigrams=[]
with open(featurefilepath) as featurecsvfile:
    featurecsvreader = csv.reader(featurecsvfile, delimiter='\t')
    for line in featurecsvreader:
        uniqueBigrams=line
numUniqueBigrams = len(uniqueBigrams)
line = None
print("got the features")


snipToCount = []
docIDKeySnipNumber = defaultdict(int)
filepathA = './subfolder-with-files/'
fileNamesA = [f1 for f1 in listdir(filepathA) if isfile(join(filepathA, f1))]  # get file names of snippet files
print("THE LIST OF FILE NAMES:")
print(fileNamesA)
for f2 in fileNamesA:
    keywordKeyDocIDValue = defaultdict(list)
    docIDKeyKeywordKeySnipValue = defaultdict(lambda: defaultdict(list))
    fS = str(f2)
    fS = fS[:-4]
    print("file in process:")
    print(fS)    
    outPredictAllFile = open('./file_to_save_results/corpus_predictAll_'+ fS + "_" + currentDateString + '.csv',  'w')
    csvoutPredictAllFile_writer = csv.writer(outPredictAllFile,delimiter=',', quotechar='"')    
    fN = './subfolder-with-files/' + f2  # combine path with file name to open file
    with open(fN) as f2:
        for lineP in f2:
            lineP = lineP.strip()
            line = lineP.split('\t')            
            docID = line[0]
            snipToCount.append(docID)
            keyW = line[1]  
            noteText = line[2] 
            if noteText in docIDKeyKeywordKeySnipValue[docID][keyW]: # prevent duplicates
                continue 
            
            docIDKeyKeywordKeySnipValue[docID][keyW].append(noteText)
            docIDKeySnipNumber[docID] += 1
            templateMatch = findTemplatePatterns(docID, noteText)       
            if templateMatch != '':  # the following separates snippet file contents according to how they are formatted - customize as needed
                 predictA_result=[]
                 predictA_result.append(str(docID))
                 predictA_result.append(keyW)
                 predictA_result.append(docIDKeySnipNumber[docID])   # snippet number      
                 if(templateMatch=='pos pattern'):      # template positive matched
                     predictA_result.append(1)  # probability slot
                     predictA_result.append(1)  # class slot
                 elif(templateMatch=='neg pattern'):      #template negative matched
                     predictA_result.append(0)  # probability slot
                     predictA_result.append(0)  # class slot
                    
                 predictA_result.append("T") # classifying method
                
                 csvoutPredictAllFile_writer.writerow(predictA_result)
                 csvoutPredictAllFile_writer.writerow('\n')       
                 continue
            
            if cleanString(noteText) != "empty string": # additional preprocessing check 
                 noteText = cleanString(noteText) 
                 textSnippet = noteText  
                 tempBigrams = []
    
                 for b in uniqueBigrams:  
                     tempB = b.split("|")
                     bString = " ".join(map(str,tempB))
                     if bString in textSnippet:
                         tempBigrams.append(b)    
                
                
                 dfRow = []
                
                 dfRow.append(docID)
                 dfRow.append(docIDKeySnipNumber[docID])
                
                 keyindex,tWord = getKeyPhraseIndex(textSnippet,keyW)    # tWord is the keyword...            
                 if(keyindex>0):
                     dfRow.append(keyindex)
                
                 else:
                     dfRow.append("0")
                
                 # get the ngram values for this document - use this for binary analysis
                 if len(tempBigrams) > 0:
                     for b in uniqueBigrams:
                         if b in tempBigrams:
                             dfRow.append("1")
                         else:
                             dfRow.append("0")
                 else: 
                     for b in uniqueBigrams:
                         dfRow.append("0")           
                 rowString = "\t".join(map(str,dfRow))        
                
                 dfRow.pop(0) #remove record id
                 dfRow.pop(0) #remove snippet number
                 X = np.array(dfRow).reshape(1,numUniqueBigrams+1)
                 svm_probs = svm.predict_proba(X)  # get predicted probability
                 pos_probs = svm_probs[:, 1]  
                 svm_class = svm.predict(X)  # get predicted class
    
                 predictA_result=[]
                 predictA_result.append(str(docID))
                 predictA_result.append(keyW)
                 predictA_result.append(docIDKeySnipNumber[docID])
                 predictA_result.append(pos_probs[0])
                 predictA_result.append(svm_class[0])
                 predictA_result.append("M")
                 csvoutPredictAllFile_writer.writerow(predictA_result)  
                 csvoutPredictAllFile_writer.writerow('\n')
    outPredictAllFile.close()                


print(str(len(snipToCount)))

    