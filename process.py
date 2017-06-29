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
            if line.startswith("[OCR_toInput]"):
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
    with open(listfile,'r',encoding='utf-8',errors='ignore') as f:
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

def readpositiondata(positionfile):
    with open(positionfile,'r',encoding='utf-8') as f:
        positiondata = json.load(f)
    return positiondata

def ngrams(text, n):
    """Yields an n-gram (tuple) at each iteration"""
    if isinstance(text, str): text = text.split(' ')
    l = len(text)

    charoffset = 0
    for i in range(-(n - 1),l):
        begin = i
        end = i + n
        if begin >= 0 and end <= l:
            ngram = " ".join(text[begin:end])
            yield charoffset, begin, end - begin, ngram
            charoffset += len(ngram) + 1

def narrowdownoffsets(corrections, ngram, charoffset, tokenoffset, tokenlength):
    """narrow down the offsets given a match: check where in the match the actual correction is (parts may be equal)"""
    correctioncharoffset = 0
    correctiontokenoffset = 0
    for i, c in enumerate(ngram):
        if c == ' ' and i > 0 and ngram[:i+1] == corrections[ngram][:i+1]:
            correctioncharoffset = i + 1
            correctiontokenoffset += 1

    charoffset += correctioncharoffset
    tokenoffset += correctiontokenoffset
    tokenlength -= correctiontokenoffset

    taillength = 0
    for i, c in enumerate(reversed(ngram)):
        if c == ' ' and i > 0 and ngram[-(i+1):] == corrections[ngram][-(i+1):]:
            tokenlength -= 1
            taillength = i+1

    original = ngram[correctioncharoffset:len(ngram) - taillength]
    correction = corrections[ngram][correctioncharoffset:len(corrections[ngram]) - taillength]

    if tokenlength <= 0:
        print(charoffset,correctioncharoffset,tokenoffset,correctiontokenoffset, tokenlength, taillength,file=sys.stderr)
        print(original,file=sys.stderr)
        print(correction,file=sys.stderr)
        print(ngram,file=sys.stderr)
        print(corrections[ngram],file=sys.stderr)
        raise ValueError("Tokenlength is " + str(tokenlength))

    return original, correction, charoffset, tokenoffset, tokenlength



def process_task1(testfiles, listfile):
    corrections = loadlist(listfile, testfiles)
    result = {}

    for testfile in testfiles:
        text = loadtext(testfile)

        tokens = text.split(' ')
        done = [False] * len(tokens) #keep track of which tokens we processed, ensuring we don't process the same token twice

        result[testfile] = {} #this will store the results

        #greedy match over all 3,2,1-grams, in that order
        for order in (3,2,1):
            for charoffset, tokenoffset, tokenlength, ngram in ngrams(tokens, order):
                assert tokenlength == order
                if ngram in corrections:
                    #we have have a match, now check where in the match the actual correction is (parts may be equal)
                    try:
                        original, correction, charoffset, tokenoffset, tokenlength = narrowdownoffsets(corrections, ngram, charoffset, tokenoffset, tokenlength)
                    except ValueError:
                        print("WARNING: Returned tokenlength was 0? Falling back to entire ngram",file=sys.stderr)
                        original = ngram
                        correction = corrections[ngram]

                    if not any(done[tokenoffset:tokenoffset+tokenlength]):
                        print(testfile + " @[" + str(charoffset) + ":" + str(tokenlength) + "]:\t" + original + " -> " + correction + "\t[" + ngram + " -> " + corrections[ngram]+"]", file=sys.stderr)
                        result[testfile][str(charoffset)+":"+str(tokenlength)] = { correction: 1.0 } #confidence always 1.0, we only output one candidate
                        for i in range(tokenoffset, tokenoffset+tokenlength): done[i] = True

    #Output to JSON
    print("Writing output to stdout", file=sys.stderr)
    print(json.dumps(result))



def process_task2(testfiles, listfile, positionfile):
    positiondata = readpositiondata(positionfile)

    corrections = loadlist(listfile, testfiles)
    result = {}

    for testfile in testfiles:
        text = loadtext(testfile)

        positions = [ (int(positiontuple.split(':')[0]), int(positiontuple.split(':')[1])) for positiontuple in positiondata[testfile] ]

        tokens = text.split(' ')

        result[testfile] = {} #this will store the results

        charoffset = 0
        testwords = []
        mask = []
        found = 0
        skip = 0
        for i, token in enumerate(tokens):
            if skip: #token was already processed as part of multi-token expression, skip:
                skip -= 1
                continue
            #Do we have an explicitly marked token here?
            for position_charoffset, position_tokenlength in positions:
                if charoffset == position_charoffset:
                    token = " ".join(tokens[i:i+position_tokenlength])
                    skip = position_tokenlength -1 #if the token consists of multiple tokens, signal to skip the rest
                    print("[" + testfile + "@" + str(position_charoffset) + ":" + str(position_tokenlength) + "] " +  token, file=sys.stderr)
                    if token in corrections:
                        tokenoffset = i
                        if token != corrections[token]:
                            original, correction, correction_charoffset, correction_tokenoffset, correction_tokenlength = narrowdownoffsets(corrections, token, charoffset, i, position_tokenlength)
                        else:
                            #this is a non-change (probably never occurs?)
                            original = token
                            correction = corrections[token]
                            correction_charoffset = position_charoffset
                            correction_tokenoffset = i
                            correction_tokenlength = position_tokenlength

                        print(testfile + " @[" + str(correction_charoffset) + ":" + str(correction_tokenlength) + "]:\t" + original + " -> " + correction + "\t[" + token + " -> " + corrections[token]+"]", file=sys.stderr)
                        result[testfile][str(correction_charoffset)+":"+str(correction_tokenlength)] = { correction: 1.0 } #confidence always 1.0, we only output one candidate
                    else:
                        print("WARNING: No correction for " + testfile + "[@" + str(position_charoffset) +":" + str(position_tokenlength) + "].. copying input verbatim...",file=sys.stderr)
                        result[testfile][str(position_charoffset)+":"+str(position_tokenlength)] = { token: 0.001 } #just copy input if we don't know
                    found += 1
            charoffset += len(token) + 1

        if found != len(positions):
            raise Exception("One or more positions were not found in the text!")

    #Output to JSON
    print("Writing output to stdout", file=sys.stderr)
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description="ICDAR 2017 Post-OCR Processing Script", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i','--input', type=str, help="Input file or directory (*.txt files)", action='store',required=True)
    parser.add_argument('-l','--list', type=str,help="Ranked correction list from TICCL", action='store',default="",required=True)
    parser.add_argument('--task', type=int, help="Task", action='store',required=True)
    parser.add_argument('--positionfile', type=str, help="Input file with position information (erroneous_tokens_pos.json), required for task 2", action='store',required=False)
    args = parser.parse_args()

    if os.path.isdir(args.input):
        testfiles = []
        for f in glob.glob(args.input + "/*.txt"):
            testfiles.append(f)
    else:
        testfiles = [args.input]

    if args.task == 1:
        process_task1(testfiles, args.list)
    elif args.task == 2:
        process_task2(testfiles, args.list, args.positionfile)


if __name__ == '__main__':
    main()
