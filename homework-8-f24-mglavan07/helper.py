import string
import re

#Input: words -- a single string that may contain punctuation
#Output: a single string without any punctuation
def remove_punc(words) :
    return re.sub(r'[^\w ]', '', words)
    
if __name__ == "__main__" :
    import nltk
    import nltk.tokenize as tk
    
    nltk.download("punkt")
    
    teststring = "Tyger Tyger, burning bright, In the forests of the night; What immortal hand or eye, Could frame thy fearful symmetry?"
    print(teststring)
    
    print(remove_punc(teststring))
