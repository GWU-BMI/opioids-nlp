# -*- coding: utf-8 -*-
"""
Spyder Editor

created by Liz Workman 2023

This is a general program to extract snippets from a single file
"""


# -*- coding: utf-8 -*-

from collections import defaultdict
import re



# customize this stoplist according to your needs, or use nltk or another resource
stoplist = ['up', 'down','for', 'a', 'of', 'the', 'The', 'and', 'to', 'in', 'by',  'is', 'has', 'an', 'was', 'not', 'or', 'wasn\'t', 'by', 'as', 
            'on',  'are', 'in', 'with', 'no', 'jan', 'nov', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec', 'dr', 'it', 'at', 'be',
            'two', 'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth', 'eleventh', 'twelveth', 'thirteenth', 'fourteenth', 
             'fifteenth', 'sixteenth', 'seventeenth', 'eighteenth', 'nineteenth', 'twentieth', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 
             'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty', 
             'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth', 'eleventh', 'twelveth', 'thirteenth', 'fourteenth', 
             'fifteenth', 'sixteenth', 'seventeenth', 'eighteenth', 'nineteenth', 'twentieth', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 
             'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty']

# counters on targets
targetKeyCountValue = defaultdict(int)


def checkTemplateMarkup(s):
    # this checks for template content
    # customize this function according to your needs
    templateContent = False
    if re.search("(-)\4+",s):  
        templateContent = True
    elif re.search("(_)\4+",s):
        templateContent = True
    elif re.search(".*\].*",s):
        templateContent = True
    elif "[x]" in s or "[ ]" in s or "[]" in s: 
        templateContent = True
    elif re.search(r"\[s*[a-z]s*\]\]",s) or re.search(r"\[s*[A-Z]s*\]\]",s) or re.search(r"\[s*[0-9]s*\]\]",s):
        templateContent = True
    elif "[]" in s: 
        templateContent = True
    elif re.search(".*\[.*\].*",s) or re.search(".*\[.*",s) or re.search(".*\].*",s):
        templateContent = True
    elif "[]" in s or "[" in s or "]" in s:
        templateContent = True
    elif "()" in s or "(x" in s or "x)" in s:
        templateContent = True
    elif "____________________________________________________________________" in s:
        templateContent = True
    return templateContent

def cleanString(s):
    # customize this function according to your needs
    s = s.lower()
    s = s.replace('\t+',' ')
    s = s.replace('\s+',' ')
    s = s.replace('\r+',' ')
    re.sub('_+',' ',s)
    # activate to remove all punctuation
#    s = re.sub(r'[^\w\s]','',s)
    parts = s.split()
    for n,p in enumerate(parts):
        parts[n].strip()

    if len(parts) > 0:
        tempString = " ".join(map(str, parts))
        if tempString != "":
            return tempString

    return "empty string"

DocIDKeysTextValues = defaultdict(str)
DocIDKeysCleanTextValues = defaultdict(str)

targetString = 'abstral,actiq,demerol,dependence,dilaudid,dolophine,duragesic,exalgo,fentanyl,fentora,hydrocodone,hydromorphone,hysingla,kadian,lorcet,lortab,meperidine,methadone,methadose,morphine,norco,opiate,opioid abuse,opioid,oxaydo,oxycodone,oxycontin,percocet,roxicet,vicodin,withdrawal,zohydro,opioid dependence,polysubstance abuse,substance abuse,substance dependence,'

targets = targetString.split(",") 

print("reading in file...")
totalSnippetCount = 0


thisFile = 'your_file'
# this loop builds a dictionary that has record id keys and text values
with open(thisFile, errors='ignore') as f:
    next(f)  # use if file includes a header
    for lineP in f:  # designed for files where the record id is the first element in the first line
        line = lineP.rstrip()
        if re.match("(\d{3}),",line):  # generic regex pattern - alter as needed
            beginningLine = line.split(',')            
            tempID = beginningLine[0]
            if len(beginningLine) > 1:
                beginningLine.remove(tempID)
                theRestP = " ".join(map(str, beginningLine))
                theRest = " " + theRestP + " "
                DocIDKeysTextValues[tempID] += theRest
        else:
            lineA = " " + line + " "
            DocIDKeysTextValues[tempID] += lineA
                  
print("getting clean string values...")
for k,v in DocIDKeysTextValues.items():
    if cleanString(v) != "empty string": # can also use the template markup check function here by adding: checkTemplateMarkup(v)
        DocIDKeysCleanTextValues[k] = cleanString(v)
print("extracting snippets...")
outfile = open('a-name-for-your-file-of-snippets',"w")
sawThese = []
snippetCount = 0
for k,v in DocIDKeysCleanTextValues.items():
    if k not in sawThese:  # to limit to one snippet per record; optional
        sawThese.append(k)
        for t in targets:
            if t in v:
                sParts = v.split(" ")
                targetParts = t.split(" ")
                # if it's a single word phrase...
                if len(targetParts) == 1:  
                    for i,s in enumerate(sParts):
                        if targetParts[0] in sParts[i]:
                            tIndex = i                                      
                            if tIndex > 50 and len(sParts) > (tIndex + 50): 
                                targetKeyCountValue[t] += 1
                                outfile.write(str(k) + "\t" + t + "\t")
                                snipString = " ".join(map(str,sParts[tIndex-50:tIndex+51]))                        
                                outfile.write(snipString + "\n")
                                snippetCount += 1
                                totalSnippetCount += 1
                # if it's a multi-word phrase...
                elif len(targetParts) > 1:                 
                    for i,s in enumerate(sParts):
                        if  i+1 < len(sParts) and targetParts[0] in sParts[i] and targetParts[1] in sParts[i+1]:
                            tIndexFirst = i
                            tIndexLast = i + (len(targetParts) - 1)
                            if tIndexFirst > 50 and len(sParts) > (tIndexLast + 50): 
                                targetKeyCountValue[t] += 1
                                outfile.write(str(k) + "\t" + t + "\t")
                                snipString = " ".join(map(str,sParts[tIndexFirst-50:tIndexLast+51]))
                                outfile.write(snipString + "\n")
                                snippetCount += 1 
                                totalSnippetCount += 1

outfile.close()        
    

print("total snippets: " + str(totalSnippetCount))            

                
            
        
        





