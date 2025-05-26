import numpy as np
from helper import remove_punc
from hw8_1 import *
import nltk

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
from nltk.corpus import stopwords


# Clean and prepare the contents of a document
def read_and_clean_doc(doc):
    
    '''
    Arguments:
        doc: string, the name of the file to be read.
    Returns:
        all_no_stop: string, a string of all the words in the file, with stopwords removed.
    Notes: Do not append any directory names to doc -- assume we will give you a string representing a file name that will open correctly
    '''     
    
    # 1. Open document, read text into *single* string
    with open(doc, "r") as f:
        allStr = f.read()

    # 2. Filter out punctuation from list of words (use remove_punc)
    all_rm_punc = remove_punc(allStr)

    # 3. Make the words lower case
    all_lower = all_rm_punc.lower()

    # 4. Filter out stopwords
    tok_low = all_lower.split(" ")
    filt_tok = [tok for tok in tok_low if tok not in stopwords.words("english")]
    all_no_stop = "".join(filt_tok)

    return all_no_stop


# Builds a doc-word matrix
def build_doc_word_matrix(doclist, n, normalize=False):
    
    '''
    Arguments:
        doclist: list of strings, each string is the name of a file to be read.
        n: int, the length of each n-gram
        normalize: boolean, whether to normalize the doc-word matrix
    Returns:
        docword: 2-dimensional numpy array, the doc-word matrix for the cleaned documents,
            with one row per document and one column per ngram (there should be as many columns as unique words that appear across *all* documents. 
            Also, Before constructing the doc-word matrix, you should sort the list of ngrams output and construct the doc-word matrix based on the sorted list
        ngramlist: list of strings, the list of ngrams that correspond to the columns in docword
    '''
    # TODO: complete the function
    docword, ngramlist = None, None

    # 1. Create the cleaned string for each doc (use read_and_clean_doc)
    cleaned = []
    for doc in doclist:
        cleaned.append(read_and_clean_doc(doc))

    # 2. Create and use ngram lists to build the doc word matrix
    doc_grams = []

    # create the ngram dictionary for each string
    for clean in cleaned:
        ngram_dict = {}

        # 1. get lines and combine them to a set of keys
        k = get_ngrams(clean, n)

        # 2. use the set to initialize a dict
        ngram_dict = dict.fromkeys(set(k), 0)

        # 3. update the values of the dict
        for key in ngram_dict:
            ngram_dict[key] = k.count(key)

        doc_grams.append(ngram_dict)

    # create a union list of all the dictionaries
    ngramlist = dict_union(doc_grams)

    # initialize the docword matrix
    rows, cols = len(doclist), len(ngramlist)
    docword = np.zeros((rows, cols), dtype= float)

    # fill in the matrix
    for row in range(len(doc_grams)):
        for col in range(len(ngramlist)):
            key = ngramlist[col]
            try:
                docword[row][col] = doc_grams[row][key]
            except KeyError:
                docword[row][col] = 0 # when key does not exist

    # optionally normalize the matrix
    if normalize:
        # get the counts on each row
        row_totals = [sum([value for value in g.values()]) for g in doc_grams]

        # normalize by count
        for row in range(len(doc_grams)):
            for col in range(len(ngramlist)):
                docword[row][col] = round(docword[row][col] / row_totals[row], 4)

    return docword, ngramlist

# Builds a term-frequency matrix
def build_tf_matrix(docword):
    
    '''
    Arguments:
        docword: 2-dimensional numpy array, the doc-word matrix for the cleaned documents, as returned by build_doc_word_matrix
    Returns:
        tf: 2-dimensional numpy array with the same shape as docword, the term-frequency matrix for the cleaned documents  
    HINTs: You may find np.newaxis helpful
    '''
    tf = None
    # TODO: fill in

    # get the counts on each row
    sums = docword.sum(axis=1)

    # shape of docword
    rows, cols = docword.shape
    tf = np.zeros((rows, cols), dtype = float)

    # normalize by count
    for row in range(rows):
        for col in range(cols):
            tf[row][col] = docword[row][col] / sums[row]

    return tf


# Builds an inverse document frequency matrix
def build_idf_matrix(docword):
    
    '''
    Arguments:
        docword: 2-dimensional numpy array, the doc-word matrix for the cleaned documents, as returned by build_doc_word_matrix
    Returns:
        idf: 1-dimensional numpy array, the inverse document frequency matrix for the cleaned documents.
             (should be a 1xW numpy array where W is the number of ngrams in the doc word matrix).
             Don't forget the log factor!
             
    '''
    idf = []
    # TODO: fill in
    n_t = np.count_nonzero(docword, axis=0)
    N, _x_ = docword.shape

    for n in n_t:
        idf.append(np.log10(N / n))

    # convert to 1D array and return
    idf = np.array(idf)
    idf = np.reshape(idf, (1, -1))
    return idf


#   Builds a tf-idf matrix given a doc word matrix
def build_tfidf_matrix(docword):
    
    '''
    Arguments:
        docword: 2-dimensional numpy array, the doc-word matrix for the cleaned documents, as returned by build_doc_word_matrix
    Returns:
        tfidf: 2-dimensional numpy array with the same shape as docword, the tf-idf matrix for the cleaned documents
    '''
    tfidf = None
    #TODO: fill in
    tf_mat = build_tf_matrix(docword)
    idf_mat = build_idf_matrix(docword)
    
    # multiply NON MATRIX WISE
    tfidf = tf_mat * idf_mat

    return tfidf


#   Find the three most distinctive ngrams, according to TFIDF, in each document
def find_distinctive_ngrams(docword, ngramlist, doclist):
    
    '''
    Arguments:
        docword: 2-dimensional numpy array, the doc-word matrix for the cleaned documents, as returned by build_doc_word_matrix
        ngramlist: list of strings, the list of ngrams that correspond to the columns in docword
        doclist: list of strings, each string is the name of a file to be read.
    Returns:
        distinctive_words: dictionary, mapping each document name from doclist to an ordered list of the three most unique ngrams in each document
    '''
    distinctive_words = {}
    # fill in
    # you might find numpy.argsort helpful for solving this problem:
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html
    # HINT: the smallest three of the negative of docword correspond to largest 3 of docword

    # get the tfidf matrix
    tfidf = build_tfidf_matrix(docword)

    # iterate through range of doclist (number of documents and enumerate each name) 
    for i, docname in enumerate(doclist):
        
        # find the top 3 tfidf values using argsort and the negative of tfidf values
        tfidf_max = np.argsort(-tfidf[i, :])[:3]

        # find the associated ngrams to the tfidf maxes
        max_ngram_names = list(np.array(ngramlist)[tfidf_max])

        # create a new dictionary entry for each docname and give the value as the three sliced maxima
        distinctive_words[docname] = max_ngram_names

    # return the words
    return distinctive_words

# broken sorting code 
'''
    for i, doc in enumerate(doclist):

        # make a dictionary from parallel lists
        tfidf_row = list(tfidf[i])
        
        row_dict = dict(zip(ngramlist, tfidf_row))

        # order the tfidf in descending order as in problem 1
        row_dict = dict(sorted(row_dict.items(), key=lambda item: (-item[1], item[0])))

        # take the first 3 appearanced by key name and add to the dictionary
        keys = list(row_dict.keys())
        print({k: row_dict[k] for i, k in enumerate(row_dict) if i < 10})
        distinctive_words[doc] = [keys[0], keys[1], keys[2]]
'''

if __name__ == "__main__":
    from os import listdir
    from os.path import isfile, join, splitext

    ### Test Cases ###
    directory = "lecs"
    path1 = join(directory, "1_vidText.txt")
    path2 = join(directory, "2_vidText.txt")

    print("\n*** Testing build_doc_word_matrix ***") 
    doclist =[path1, path2]
    docword, wordlist = build_doc_word_matrix(doclist, 4)
    print(docword.shape)
    print(len(wordlist))
    print(docword[0][0:10])
    print(wordlist[0:10])
    print(docword[1][0:10])
    print("\n*** Testing build_doc_word_matrix normalization ***") 
    doclist =[path1, path2]
    docword, wordlist = build_doc_word_matrix(doclist, 4, normalize=True)
    print(docword.shape)
    print(len(wordlist))
    print(docword[0][0:10])
    print(wordlist[0:10])
    print(docword[1][0:10])

    # Uncomment the following code block to test build_tf_matrix, builf_idf_matrix, and build_tfidf_matrix
    print("\n*** Testing build_tf_matrix ***")
    doclist =[path1, path2]
    docword, wordlist = build_doc_word_matrix(doclist, 4, normalize=False)
    tf = build_tf_matrix(docword)
    print(tf[0][0:10])
    print(tf[1][0:10])
    print(tf.sum(axis=1)) 
    print("\n*** Testing build_idf_matrix ***")
    idf = build_idf_matrix(docword)
    print(idf[0][0:10])
    print("\n*** Testing build_tfidf_matrix ***")
    tfidf = build_tfidf_matrix(docword)
    print(tfidf.shape)
    print(tfidf[0][0:10])
    print(tfidf[1][0:10])

    # Uncomment the following code block to test find_distinctive_ngrams
    print("\n*** Testing find_distinctive_words ***")
    print(find_distinctive_ngrams(docword, wordlist, doclist))
