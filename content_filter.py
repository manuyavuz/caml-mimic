import torch
import datasets
import argparse
from dataproc import extract_wvs


def load_vocab_dict(args, vocab_file):
    #reads vocab_file into two lookups (word:ind) and (ind:word)
    vocab = set()
    with open(vocab_file, 'r') as vocabfile:
        for i,line in enumerate(vocabfile):
            line = line.rstrip()
            if line != '':
                vocab.add(line.strip())
    #hack because the vocabs were created differently for these models
    ind2w = {i+1:w for i,w in enumerate(sorted(vocab))}
    w2ind = {w:i for i,w in ind2w.items()}
    return ind2w, w2ind

parser = argparse.ArgumentParser(description="train a neural network on some clinical documents")
args = parser.parse_args()
args.data_path = 'mimicdata/mimic3/train_50.csv'
ind2w, w2ind = load_vocab_dict(args, 'mimicdata/mimic3/vocab.csv')


W = torch.Tensor(extract_wvs.load_embeddings('mimicdata/mimic3/processed_full.embed'))

embedding = torch.nn.Embedding(W.size()[0], W.size()[1], padding_idx=0)

query = 'admission'

text = 'admission date discharge date date of birth sex m service illness rectal perforation history of present illness the patient is a year old male who presented to the emergency department with to hours of lower abdominal pain he was seen in another facility and then transferred here hypotensive and tachycardic with a ct scan suggesting a rectal perforation physical examination the patient appeared ill had a temperature of abdomen was tender with guarding hospital course the patient was taken to the operating room the perforation could not be repaired a proximal colostomy was performed postoperatively the patient was in the intensive care unit for several days but was treated with iv antibiotics and appropriate fluid he then began to progress and was changed to a diet he began rehab screening which took several days he was counseled with enterostomal therapy and was discharged on discharge medications percocet po q 4h p r n metamucil discharge plan return to clinic to see dr last name stitle in weeks first name8 namepattern2 last name namepattern1 md md number dictated by last name namepattern1 medquist36 d t job job number'
text_list = text.split()
# import pdb; pdb.set_trace()
query_embed = embedding(torch.LongTensor([w2ind[query]]))

text_embed = torch.cat([embedding(torch.LongTensor([w2ind[w]])) for w in text_list])


k = 10
sim = torch.nn.functional.cosine_similarity(text_embed, query_embed)

topk_ind = torch.topk(sim, k)[1]
# import pdb; pdb.set_trace()

filtered_text_list = [text_list[index] for index in topk_ind]

print(' '.join(filtered_text_list))

