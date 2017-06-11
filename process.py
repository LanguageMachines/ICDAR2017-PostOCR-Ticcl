#!/usr/bin/env python3

#-------------------------------------------------------------------
# Processing script for ICDAR 2017 Post-OCR Text Correction challenge
# Uses TICCL ranked correction lists as input
#-------------------------------------------------------------------
#       by Maarten van Gompel
#       Radboud University Nijmegen
#       Licensed under GPLv3

import os
import sys
import argparse
import glob
import json
import colibricore

def loadtext(testfile):
    """Load the text from a test file"""
    with open(testfile,'r',encoding='utf-8') as f:
        for line in f:
            if line.startwith("[OCR_toInput]"):
                return line[len("[OCR_toInput]") + 1:]
            else:
                raise Exception("Unexpected input format, expected [OCR_toInput] on first line")

    raise Exception("No text found")

def buildpatternmodel(testfiles):
    print("Loading test data...", file=sys.stderr)

    with open('inputmodel.txt','w',encoding='utf-8') as f:
        for testfile in testfiles:
            f.write(loadtext(testfile) + "\n")

    print("Building pattern model...", file=sys.stderr)

    classencoder = colibricore.ClassEncoder()
    classencoder.build('inputmodel.txt')
    classencoder.save('inputmodel.colibri.cls')
    classencoder.encodefile('inputmodel.txt','inputmodel.colibri.dat')

    options = colibricore.PatternModelOptions(mintokens=1, maxlength=3)
    patternmodel = colibricore.UnindexedPatternModel()
    patternmodel.train('inputmodel.colibri.dat', options)

    return patternmodel, classencoder


def loadlist(listfile, testfiles):
    """Load correction list, but only loading the patterns we are actually going to use"""

    corrections = {}

    patternmodel, classencoder =  buildpatternmodel(testfiles)

    print("Loading and filtering correction list...", file=sys.stderr)
    found = 0
    total = 0
    with open(listfile,'r',encoding='utf-8') as f:
        for line in f:
            total += 1

            #format: variant hekje frequentie_variant hekje correctie hekje frequentie_correctie hekje Levenshtein_Distance hekje Interne_code (M = manueel geverifieerd/goedgekeurd)
            variant, variantfreq, correction, correctionfreq, ld, code = line.strip().split('#')
            variantfreq = int(variantfreq)
            correctionfreq = int(correctionfreq)
            ld = int(ld)
            variant = variant.replace('_',' ')
            correction = correction.replace('_',' ')

            variantpattern = classencoder.buildpattern(variant, True)
            if not variantpattern.unknown() and variantpattern in patternmodel:
                found += 1
                corrections[variant] = correction

    print("Found " + str(found) + " candidates out of " + str(total) + " patterns...", file=sys.stderr)
    return corrections

def ngrams(text, n):
    """Yields an n-gram (tuple) at each iteration"""
    if isinstance(text, 'str'): text = text.split(' ')
    l = len(text)

    charoffset = 0
    for i in range(-(n - 1),l):
        begin = i
        end = i + n
        if begin >= 0 and end <= l:
            ngram = text[begin:end]
            yield charoffset, begin, end - begin, ngram
            charoffset = len(ngram) + 1

def process(testfiles, listfile):
    corrections = loadlist(listfile, testfiles)

    for testfile in testfiles:
        text = loadtext(testfile)

        tokens = text.split(' ')
        done = [False] * len(tokens) #keep track of which tokens we processed, ensuring we don't process the same token twice

        result = {} #this will store the results

        #greedy match over all 3,2,1-grams, in that order
        for order in (3,2,1):
            for charoffset, tokenoffset, tokenlength, ngram in ngrams(tokens, order):
                assert tokenlength == order
                if not any(done[tokenoffset:tokenoffset+tokenlength]):
                    if ngram in corrections:
                        print(testfile + " @[" + str(charoffset) + "chr," + str(tokenlength) + "toklen]: " + ngram + " -> " + corrections[ngram])
                        result[str(charoffset)+":"+str(tokenlength)] = { corrections[ngram]: 1.0 } #confidence always 1.0, we only output one candidate
                        for i in range(tokenoffset, tokenoffset+tokenlength): done[i] = True

        #Output to JSON
        print("Writing output to " + testfile.replace('.txt','') + '.json', file=sys.stderr)
        with open(testfile.replace('.txt','') + '.json','w',encoding='utf-8') as f:
            json.dump(result, f)


def main():
    parser = argparse.ArgumentParser(description="ICDAR 2017 Post-OCR Processing Script", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i','--input', dest='settype',type="str", help="Input file or directory (*.txt files)", action='store',required=True)
    parser.add_argument('-l','--list ', type=str,help="Ranked correction list from TICCL", action='store',default="",required=True)
    args = parser.parse_args()

    if os.path.isdir(args.input):
        testfiles = []
        for f in glob.glob(args.input + "/*.txt"):
            testfiles.append(f)
    else:
        testfiles = [args.input]

    process(testfiles, args.list)


if __name__ == '__main__':
    main()
