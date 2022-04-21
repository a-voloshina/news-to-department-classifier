# -*- coding: utf-8 -*-
import os
import requests
import math
import yake
import summa
import zipfile
import gensim
import logging
import operator
import time
# import nltk

from nltk import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pymystem3 import Mystem
from string import punctuation
from collections import Counter
from rake_nltk import Rake
from keybert import KeyBERT

mystem = Mystem()
# nltk.download('stopwords')
# nltk.download('punkt')
russian_stopwords = stopwords.words("russian") + ['это', 'вне', 'который']

model_unknown_words = set()


def get_text_from_list(list_text):
    return " ".join(list_text)


def get_list_tokens(text):
    tokens = mystem.lemmatize(text.lower())
    return [token for token in tokens if token not in russian_stopwords
            # and token not in departments_stopwords
            and token != " "
            and token.strip() not in punctuation]  # -,


def preprocess_text(text):
    tokens = get_list_tokens(text)
    return list(set(tokens))


def tokenize_text(text):
    tokens = get_list_tokens(text)
    return get_text_from_list(tokens)


mapping = {
    'A': 'ADJ',
    'ADV': 'ADV',
    'ADVPRO': 'ADV',
    'ANUM': 'ADJ',
    'APRO': 'ADJ',  # DET похоже не используется в RusVectores, разобраться
    'COM': 'ADJ',
    'CONJ': 'SCONJ',
    'INTJ': 'INTJ',
    'NONLEX': 'X',
    'NUM': 'NUM',
    'PART': 'PART',
    'PR': 'ADP',
    'S': 'NOUN',
    'SPRO': 'PRON',
    'UNKN': 'X',
    'V': 'VERB'
}


def preprocess_text_tag_mystem(text):
    processed = mystem.analyze(text)
    tagged = []
    for token in processed:
        try:
            lemma = token["analysis"][0]["lex"].lower().strip()
            if lemma not in russian_stopwords:  # and lemma not in departments_stopwords:
                pos = token["analysis"][0]["gr"].split(',')[0]
                pos = pos.split('=')[0].strip()
                if pos in mapping:
                    tag = lemma + '_' + mapping[pos]
                    tagged.append(tag)  # здесь мы конвертируем тэги
                else:
                    tag = lemma + '_X'
                    tagged.append(tag)
        except (KeyError, IndexError):
            continue

    return list(set(tagged))


# def preprocess_keywords_tag_mystem(keywords_list):
#     return preprocess_text_tag_mystem(' '.join([keyword[0] for keyword in keywords_list]))


# def preprocess_text_tag_mystem_with_stopwords(text, stopwords_list):
#     processed = mystem.analyze(text)
#     # print(processed)
#     tagged = []
#     for token in processed:
#         try:
#             lemma = token["analysis"][0]["lex"].lower().strip()
#             if lemma not in russian_stopwords and lemma not in stopwords_list:
#                 pos = token["analysis"][0]["gr"].split(',')[0]
#                 pos = pos.split('=')[0].strip()
#                 if pos in mapping:
#                     tag = lemma + '_' + mapping[pos]
#                     tagged.append(tag)  # здесь мы конвертируем тэги
#                 else:
#                     tag = lemma + '_X'
#                     tagged.append(tag)
#         except KeyError:
#             continue
#
#     return tagged
# return Counter(tagged)


# def api_similarity(model_name, word1, word2):
#     url = '/'.join(['https://rusvectores.org', model_name, word1 + '__' + word2, 'api', 'similarity/'])
#     request = requests.get(url, stream=True)
#     similarity = request.text.split('\t')[0]
#     return float(similarity)


def get_rusvectors_model(model_file_name):
    archive = zipfile.ZipFile(model_file_name, 'r')
    stream = archive.open('model.bin')
    return gensim.models.KeyedVectors.load_word2vec_format(stream, binary=True)


# def get_similarities_for_one_text(text_words, model_name):
#     result = list()
#     for word1 in text_words:
#         for word2 in text_words:
#         similarity = api_similarity(model_name, text1_word, text2_word)
#         if similarity != 1.0:
#             print(text1_word, text2_word, similarity)

# returns key words for text1


def get_similarities_list(text1_words, text2_words, model):
    result = list()
    for text1_word in text1_words:
        mid_similarity = 0
        for text2_word in text2_words:
            similarity = model.similarity(text1_word, text2_word)
            if similarity < 1.0:
                mid_similarity += similarity
        mid_similarity_result = mid_similarity / len(text2_words)
        result.append(mid_similarity_result)
    return result


def get_similarities_for_one_department(department1, departments_list, model):
    similarities_line = {}
    for word1 in department1:
        mid_similarity = 0
        sum_len = 0
        for department2 in departments_list:
            if department1 != department2:
                for word2 in department2:
                    try:
                        similarity = model.similarity(word1, word2)
                        if similarity < 1.0:
                            mid_similarity += similarity
                    except KeyError:
                        model_unknown_words.add(word2)
                        continue
                sum_len += len(department2)
        mid_similarity_result = mid_similarity / sum_len
        similarities_line[word1] = mid_similarity_result
        # similarities_line.append(mid_similarity_result)
    return similarities_line


def get_similarities_table(departments_list, model):
    result_table = []
    i = 1
    for department1 in departments_list:
        similarities = get_similarities_for_one_department(department1, departments_list, model)
        result_table.append(similarities)
        # print(f'Got similarities for {i} department')
        i += 1
    return result_table


def get_departments_keywords(departments_list, model, key_words_count):
    similarities_table = get_similarities_table(departments_list, model)
    # print("Got similarities_table")
    result_table = []
    for line in similarities_table:
        result_words = []
        sorted_similarities = sorted(line.items(), key=operator.itemgetter(1))
        for key_word in sorted_similarities[:key_words_count]:
            result_words.append(key_word[0])
        result_table.append(result_words)
    return result_table


def compare_texts(text1, text2, model_name, key_words_count):
    text1_words = preprocess_text_tag_mystem(text1)  # .keys()
    print(text1_words)
    text2_words = preprocess_text_tag_mystem(text2)  # .keys()
    print(text2_words)
    print()
    similarities_list = get_similarities_list(text1_words, text2_words, model_name)
    sorted_results = sorted(similarities_list)
    result_words = []
    for key in sorted_results[:key_words_count]:
        key_index = similarities_list.index(key)
        key_word = text1_words[key_index]
        result_words.append(key_word)
    return result_words


def compare_preprocessed_texts(text1_words, text2_words, model, key_words_count):
    print(text1_words)
    print(text2_words)
    print()
    similarities_list = get_similarities_list(text1_words, text2_words, model)
    sorted_results = sorted(similarities_list)
    result_words = []
    for key in sorted_results[:key_words_count]:
        key_index = similarities_list.index(key)
        key_word = text1_words[key_index]
        result_words.append(key_word)
    return result_words


# returns key words for file1
def compare_files(text1_file_name, text2_file_name, model_name, key_words_count):
    with open(text1_file_name, encoding='utf-8', mode='r') as file1, \
            open(text2_file_name, encoding='utf-8', mode='r') as file2:
        text1 = file1.read()
        print(text1)
        print()
        text2 = file2.read()
        print(text2)
        print()
        return compare_texts(text1, text2, model_name, key_words_count)


def get_text_dep_similarity(text_words, dep_keywords, model_name):
    similarities = get_similarities_list(text_words, dep_keywords, model_name)
    similarity = 0
    for sim in similarities:
        similarity += sim
    return similarity / len(similarities)


def get_text_department(text_words, deps_map, model_name):
    dep_sims = {}
    for dep in deps_map.items():
        similarity = get_text_dep_similarity(text_words, dep[1], model_name)
        print(dep[0] + ': ' + similarity)
        dep_sims[dep[0]] = similarity
    return sorted(dep_sims.items(), key=lambda item: item[1], reverse=True)


def get_most_frequent_words_from_list(splitted_text, words_number):
    counter = Counter(splitted_text)
    return counter.most_common(words_number)


def get_departments_tasks_texts(departments_texts_path):
    departments = [f for f in os.listdir(departments_texts_path)
                   if os.path.isfile(os.path.join(departments_texts_path, f))]
    departments_tasks_list = []
    for department in departments:
        department_file_name = f"{departments_texts_path}/{department}"
        with open(department_file_name, "r", encoding='utf-8') as department_file:
            department_tasks = department_file.read().replace('\n', ' ')
            departments_tasks_list.append(department_tasks)
    return departments_tasks_list


def get_departments_tasks_list(departments_texts_path, add_tags=True):
    departments = [f for f in os.listdir(departments_texts_path)
                   if os.path.isfile(os.path.join(departments_texts_path, f))]
    departments_tasks_list = []
    for department in departments:
        department_file_name = f"{departments_texts_path}/{department}"
        with open(department_file_name, "r", encoding='utf-8') as department_file:
            department_tasks = department_file.read().replace('\n', ' ')
            if add_tags:
                preprocessed_department = preprocess_text_tag_mystem(department_tasks)
            else:
                preprocessed_department = preprocess_text(department_tasks)
            departments_tasks_list.append(preprocessed_department)
    return departments_tasks_list


def get_departments_stopwords(departments_tasks_list, limit):
    all_text_preprocess = []
    for department_tasks in departments_tasks_list:
        all_text_preprocess.extend(department_tasks)
    most_frequent_words = get_most_frequent_words_from_list(all_text_preprocess, limit)
    return [x[0] for x in most_frequent_words]


def remove_stopwords(departments_tasks_list, stopwords_list):
    new_departments_tasks = []
    for department_tasks in departments_tasks_list:
        new_department_tasks = [word for word in department_tasks if word not in stopwords_list]
        new_departments_tasks.append(new_department_tasks)
    return new_departments_tasks


def compute_tf(input_text):
    tf_text = Counter(input_text)
    for i in tf_text:
        tf_text[i] = tf_text[i] / float(len(input_text))
    return tf_text


def compute_idf(word, corpus):
    return math.log10(len(corpus) / sum([1.0 for i in corpus if word in i]))


def compute_tfidf(corpus):
    documents_list = []
    for text in corpus:
        tf_idf_dictionary = {}
        computed_tf = compute_tf(text)
        for word in computed_tf:
            tf_idf_dictionary[word] = computed_tf[word] * compute_idf(word, corpus)
        documents_list.append(tf_idf_dictionary)
    return documents_list


# def compute_tfidf(text, corpus):
#     tf_idf_dictionary = {}
#     computed_tf = compute_tf(text)
#     for word in computed_tf:
#         tf_idf_dictionary[word] = computed_tf[word] * compute_idf(word, corpus)
#     return tf_idf_dictionary

def tf_idf_keywords_extraction(departments_list, keywords_count, tuple_elements=False):
    departs_corpus = []
    for department in departments_list:
        departs_corpus.append(department)

    departments_tf_idf = []
    tf_idf = compute_tfidf(departs_corpus)
    for keywords in tf_idf:
        sorted_keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)
        if tuple_elements:
            department_keywords = []
            for elem in sorted_keywords[:keywords_count]:
                department_keywords.append(elem[0])
            departments_tf_idf.append(department_keywords)
        else:
            departments_tf_idf.append(sorted_keywords)
    return departments_tf_idf


def rake_keywords_extraction(tokenized_text):
    r = Rake(russian_stopwords)
    r.extract_keywords_from_text(tokenized_text)
    # return r.get_ranked_phrases()
    return r.get_ranked_phrases_with_scores()


def yake_keywords_extraction(text, keywords_count):
    language = "rus"
    max_ngram_size = 1
    deduplication_threshold = 0.9
    custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold,
                                                top=keywords_count, features=None)
    return custom_kw_extractor.extract_keywords(text)


def keybert_keywords_extraction(text):
    kw_extractor = KeyBERT('paraphrase-multilingual-MiniLM-L12-v2')
    keywords = kw_extractor.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words=None)
    return keywords


def print_departments_keywords(departments_keywords):
    i = 1
    for department_keywords in departments_keywords:
        print(f'department {i} keywords: {department_keywords}')
        i += 1


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model_name = "ruwikiruscorpora_upos_cbow_300_10_2021"
    # #"ruwikiruscorpora_upos_skipgram_300_2_2019" #"news_upos_skipgram_300_5_2019"
    model = get_rusvectors_model(f"./model/{model_name}.zip")
    departments_texts_path = ".\departments"
    stopwords_count = 15
    departments_keywords_count = 10

    departments = get_departments_tasks_list(departments_texts_path)
    departments_stopwords = get_departments_stopwords(departments, stopwords_count)
    departments = remove_stopwords(departments, departments_stopwords)

    start = time.time()
    custom_keywords = get_departments_keywords(departments, model, departments_keywords_count)
    end = time.time()
    print_departments_keywords(custom_keywords)
    print(f'custom departments keywords got {end - start} sec')
    # print(f'model {model_name} unknown words count: {len(model_unknown_words)}')
    # print(model_unknown_words)

    print('YAKE')
    departments_texts = get_departments_tasks_list(departments_texts_path, add_tags=False)
    departments_texts_stopwords = get_departments_stopwords(departments_texts, stopwords_count)
    departments_texts = remove_stopwords(departments_texts, departments_stopwords)

    i = 1
    yake_start = time.time()
    for department in departments_texts:
        department_text = get_text_from_list(department)
        yake_keywords = sorted(yake_keywords_extraction(department_text, departments_keywords_count))
        keywords = []
        for keyword in yake_keywords:
            keywords.append(keyword[0])
        print(f'department {i} keywords: {keywords}')
        i += 1
    yake_end = time.time()
    print(f'yake departments keywords got {yake_end - yake_start} sec')

    print('TF-IDF')
    tf_idf_start = time.time()
    tf_idf_keywords = tf_idf_keywords_extraction(departments_texts, departments_keywords_count, tuple_elements=True)
    tf_idf_end = time.time()
    print_departments_keywords(tf_idf_keywords)
    print(f'tf-idf departments keywords got {tf_idf_end - tf_idf_start} sec')
    print()

    print('RAKE')
    # print(rake_keywords_extraction(preprocess_result))
    # education_preprocess = wordpunct_tokenize(education_depart_tasks)
    # ' '.join([word for word in (education_depart_tasks.split()) if word not in punctuation])
    # print(education_preprocess)
    # print(rake_keywords_extraction(education_depart_tasks))
    # print(rake_keywords_extraction(tokenize_text(building_and_architecture_depart_tasks)))
    rake_start = time.time()
    rake_keywords = []
    for department in departments_texts:
        department_text = get_text_from_list(department)
        rake_keywords.append(rake_keywords_extraction(department_text))
    rake_end = time.time()
    print_departments_keywords(rake_keywords)
    print(f'rake departments keywords got {rake_end - rake_start} sec')
    print()

    print('TextRank')
    text_rank_keywords = []
    text_rank_start = time.time()
    for department in departments_texts:
        department_text = get_text_from_list(department)
        text_rank_keywords.append(summa.keywords.keywords(department_text))
    text_rank_end = time.time()
    print_departments_keywords(text_rank_keywords)
    print(f'text rank departments keywords got {text_rank_end - text_rank_start} sec')
    print()

    print('KeyBERT')
    keybert_keywords = []
    keybert_start = time.time()
    for department in departments_texts:
        department_text = get_text_from_list(department)
        department_keywords = keybert_keywords_extraction(department_text)
        department_keywords_alone = []
        for tuple_word in department_keywords:
            department_keywords_alone.append(tuple_word[0])
        keybert_keywords.append(department_keywords_alone)
    keybert_end = time.time()
    print_departments_keywords(keybert_keywords)
    print(f'keybert departments keywords got {keybert_end - keybert_start} sec')
    print()

    departments_tasks_raw = get_departments_tasks_texts(departments_texts_path)
    keybert_keywords = []
    keybert_start = time.time()
    for department in departments_tasks_raw:
        department_keywords = keybert_keywords_extraction(department)
        department_keywords_alone = []
        for tuple_word in department_keywords:
            department_keywords_alone.append(tuple_word[0])
        keybert_keywords.append(department_keywords_alone)
    keybert_end = time.time()
    print_departments_keywords(keybert_keywords)
    print(f'keybert departments keywords got {keybert_end - keybert_start} sec')
    print()
