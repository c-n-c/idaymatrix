from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tag import pos_tag
from nltk.corpus.reader import PlaintextCorpusReader
import math
import os
import pprint
import copy


LIST_CHALLENGE = [
    ['a', 'r', 'i', 'k', 'h', 'i', 'l', 'a', 'f', 'a', 't', 'h'],
    ['m', 'o', 'd', 'a', 's', 'c', 'e', 't', 'p', 'o', 'u', 'g'],
    ['o', 'l', 'a', 'g', 'i', 'f', 'i', 'u', 'a', 'g', 'h', 'a'],
    ['c', 'o', 'n', 's', 't', 'i', 't', 'u', 't', 'i', 'o', 'n'],
    ['k', 'c', 'd', 's', 'i', 'h', 'b', 'a', 'r', 'o', 'l', 'd'],
    ['n', 'i', 'i', 'd', 'r', 'c', 'e', 'f', 'i', 'o', 'g', 'h'],
    ['e', 'r', 'f', 'v', 'b', 's', 'u', 't', 'o', 'l', 'z', 'i'],
    ['m', 't', 'q', 'i', 'd', 'y', 'c', 'a', 't', 'd', 'o', 'j'],
    ['s', 'e', 'v', 'e', 'n', 't', 'y', 'f', 'i', 'v', 'e', 'i'],
    ['i', 'k', 'o', 'p', 'u', 's', 'w', 't', 'c', 'i', 'o', 'q'],
]

F_THRESHOLD = 15
DUMP_DIR = "data"

current_dump_dir = os.path.abspath(os.curdir) + os.path.sep + DUMP_DIR

def corpus_reader():
    reader = PlaintextCorpusReader(current_dump_dir, '.*\.txt')
    return reader


def get_words_from_matrix(row_list):
    bag_of_unknown_words = []
    for each in row_list:
        row_str = "".join(each)
        row_data = []
        for alpha in range(len(row_str)):
            if len(row_str[alpha:]) > 3:
                word_found = row_str[alpha:]
                row_data.append(word_found)
                row_data.append(word_found[::-1])
                # form words of 3 or more letters
                range_var = range(alpha + 2, len(row_str))
                for nums in range_var:
                    possible = row_str[alpha:nums]
                    if len(possible) > 2:
                        row_data.append(possible)
                        row_data.append(possible[::-1])
        bag_of_unknown_words.extend(row_data)
    return bag_of_unknown_words


def transpose_matrix(matrix):
    transpose_mat = []
    row_size = len(matrix[0])
    column_size = len(matrix)
    for col_marker in range(0, (row_size)):
        transpose_row = []
        for row_marker in range(0, (column_size)):
            transpose_row.append(matrix[row_marker][col_marker])
        transpose_mat.append(transpose_row)
    return transpose_mat


def filter_stopwords(word_list):
    filtered_list = set(word_list).difference(set(stopwords.words('english')))
    return filtered_list


reader = corpus_reader()
all_words = reader.words()
word_freq = FreqDist(all_words)


def match_from_wiki_corpus(filtered_words):
    global word_freq, all_words
    found_words = set(all_words).intersection(set(filtered_words))
    return found_words


def trim_words_with_frequency(all_found):
    matched_words={}
    if all_found:
        for word in all_found:
            matched_words[word] = word_freq[word]
    order_words_by_frequency = list(matched_words.keys())
    order_words_by_frequency.sort(key=lambda x: - matched_words[x])
    #print("all words ->", order_words_by_frequency)
    cleaned_words = [each for each in order_words_by_frequency if matched_words[each] >= F_THRESHOLD]
    cleaned_words.sort(key=lambda x: -len(x))
    #print("most common ->", cleaned_words)
    return cleaned_words


def reduce_noise_by_POS_tagging(word_list):
    k = reader.sents()
    survived_words = {}.fromkeys(word_list)
    for sentence in k:
        match = set(sentence).intersection(set(word_list))
        if match:
            tagged = pos_tag(sentence)
            for (each, tag) in tagged:
                if each in word_list:
                    # only nouns, cardinal digits, and foreign words are tagged
                    if tag and tag in ["NN", "NNP", "FW", "CD"]:
                        survived_words[each] = (tag," ".join(sentence))
                        # print(f"tagged->{each, tag, ' '.join(sentence)}")
                        word_list.pop(word_list.index(each))
                    # percepton tagger may need to pass through more examples for plurals, adjectives and verbs
                    elif tag and tag in ["NNPS", "NNS", "JJ", "VB"]:
                        #print("more tries needed for->", each)
                        survived_words[each] = (tag, " ".join(sentence))
                    else:
                        #print(f"removed->{each, tag, ' '.join(sentence)}")
                        survived_words.pop(each)
                        word_list.pop(word_list.index(each))
        if not word_list:
            return survived_words
    return survived_words


def get_TF_IDF(words_list):
    tf = {}.fromkeys(words_list, [])
    # term_total = {}.fromkeys(words_list,0)  # redundant
    paras_list = reader.paras()
    stop_set =  set(stopwords.words('english'))
    for para_index, para in enumerate(paras_list):
        flat_para = []
        for line in para:
            flat_para.extend(line)
        clean_para = [each for each in flat_para if each in set(flat_para).difference(stop_set)]
        freq_para = FreqDist(clean_para)
        intersect = set(words_list).intersection(set(clean_para))
        if intersect:
            for word in intersect:
                current_list = copy.copy(tf[word])
                if current_list:
                    current_list.append((para_index, freq_para.freq(word)))
                    new_list = current_list
                else:
                    new_list = [(para_index, freq_para.freq(word))]
                tf[word] = new_list
                # term_total[word] += 1

    doc_count = len(paras_list)
    idf = {key: math.log(doc_count/len(tf[key])) for key in tf.keys() if len(tf[key]) > 0}
    # return TFXIDF scores for each term
    term_popularity = sorted(tf, key=lambda x: len(tf[x]), reverse=True)
    tf_idf = {}
    for term in term_popularity:
        term_idf = idf[term]
        tf_idf_term = [(item[0], item[1]* term_idf) for item in tf[term]]
        tf_idf_term.sort(key=lambda x: x[1], reverse=True)
        tf_idf[term] = tf_idf_term
    return tf_idf, paras_list


if __name__ == "__main__":
    transpose_mat = transpose_matrix(LIST_CHALLENGE)
    words_list = get_words_from_matrix(LIST_CHALLENGE) + get_words_from_matrix(transpose_mat)
    filtered_words = list(filter_stopwords(words_list))
    print(f"total unique words extracted from matrix ->{len(filtered_words)}")
    all_found = match_from_wiki_corpus(filtered_words)
    # # print(f"total words matching in corpus->{len(all_found)} \n {all_found}")
    # cleaned_words = trim_words_with_frequency(list(all_found))
    survived_words = reduce_noise_by_POS_tagging(list(all_found))
    # # print(f"total survied tags->{len(survived_words)}\n{survived_words}")
    # cascade_frequency_filter_pos_tagging = [(each, *survived_words[each]) for each in cleaned_words\
    # if each in survived_words.keys() and survived_words[each][0] not in ("NNS", "JJ")]
    # print(f"total recommendations from cascade of freq. filter"
    #       f" along with pos tagging filter->{len(cascade_frequency_filter_pos_tagging)}")
    # # pprint.pprint(cascade_frequency_filter_pos_tagging)
    tf_idf, paras_list = get_TF_IDF(list(all_found))
    cascade_tf_idf_pos_tagging = [(each, survived_words[each][0],
    " ".join([" ".join(line) for line in paras_list[tf_idf[each][0][0]]])) for each in tf_idf.keys()\
    if each in survived_words.keys() and survived_words[each][0] not in ("NNS", "JJ")\
    and len(tf_idf[each]) >= F_THRESHOLD]
    print(f"total recommendations from cascade of tfxidf. filter"
          f" along with pos tagging filter->{len(cascade_tf_idf_pos_tagging)}")
    pprint.pprint(cascade_tf_idf_pos_tagging)