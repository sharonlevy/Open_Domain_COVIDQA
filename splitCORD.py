import json
import os
import nltk
import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--metadata-file', type=str)
parser.add_argument('--pmc-path', type=str)
parser.add_argument('--out-path', type=str)

args = parser.parse_args()


metadata = {}
with open(args.metadata_file,'r') as f:
    reader = csv.reader(f)
    headers = next(reader, None)
    for row in reader:
        pmcid = row[5]
        if pmcid not in metadata:
            metadata[pmcid] = {"journal": row[11], "date": row[9], "authors": row[10]}


path = args.pmc_path
dumpPath = args.out_path
count = 0
paragraphs = []
counter = 0


dumpFile = open(dumpPath, 'w')

for filename in os.listdir(path):
    if count % 1000 == 0:
        print(count)
    count += 1
    f = open(path+'/'+filename,'r')
    data = json.load(f)
    f.close()
    #loop through each paragraph in the article
    for paragraph in data['body_text']:
        #typically irrelevant headers
        if len(paragraph['text'].split()) < 8:
            continue
        text = paragraph['text']
        sentences = nltk.sent_tokenize(text)
        chunked = []
        curr_len = 0
        curr_sentences = []
        #split into semi-even chunks
        for sent in sentences:
            if curr_len + len(sent.split()) > 90:
                curr_sentences.append(sent)
                chunked.append(' '.join(curr_sentences))
                curr_len = 0
                curr_sentences = []
            else:
                curr_len += len(sent.split())
                curr_sentences.append(sent)
        if curr_len > 90 or len(chunked) == 0:
            chunked.append(' '.join(curr_sentences))
        else:
            last = chunked.pop()
            chunked.append(' '.join([last]+curr_sentences))
        for chunkedParagraph in chunked:
            entry = {"id": data['paper_id'], "title": data['metadata']['title'], "text": chunkedParagraph, "index": counter, "date": metadata[data['paper_id']]['date'],
                     "journal": metadata[data['paper_id']]['journal'],"authors": metadata[data['paper_id']]['authors']}
            json.dump(entry, dumpFile)
            dumpFile.write('\n')
            counter += 1

dumpFile.close()
