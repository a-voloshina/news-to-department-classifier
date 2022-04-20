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
from nltk import wordpunct_tokenize

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pymystem3 import Mystem
from string import punctuation
from collections import Counter
from rake_nltk import Rake
from keybert import KeyBERT

mystem = Mystem()
russian_stopwords = stopwords.words("russian")

departments_stopwords = []


# ['организация', 'обеспечение', 'соответствие', 'определение', 'осуществление', 'участие',
#                          'компетенция', 'комитета', 'aнализ', 'прогнозирование', 'полномочие',
#                          'деятельность', 'мэрия', 'город', 'новосибирск', 'российская', 'федерация']


def preprocess_text(text):
    lemmatized_text = mystem.lemmatize(text.lower())
    # print(tokens)
    # processed = mystem.analyze(text)
    # print(processed)
    tokens = [token for token in lemmatized_text if token not in russian_stopwords
              # and token not in departments_stopwords
              and token != " "
              and token.strip() not in punctuation]

    # text = " ".join(tokens)
    return list(set(tokens))  # text


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
            if lemma not in russian_stopwords:  #and lemma not in departments_stopwords:
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
                    except KeyError as err:
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
        print(f'Got similarities for {i} department')
        i += 1
    return result_table


def get_departments_keywords(departments_list, model, key_words_count):
    similarities_table = get_similarities_table(departments_list, model)
    print("Got similarities_table")
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

def tf_idf_keywords_extraction():
    departs_corpus = []
    building_depart_tasks_processed = preprocess_text(building_and_architecture_depart_tasks)
    education_depart_tasks_processed = preprocess_text(education_depart_tasks)
    energy_depart_tasks_processed = preprocess_text(energy_housing_and_communal_services_depart_tasks)
    industry_depart_tasks_processed = preprocess_text(industry_innovation_and_enterprise_depart_tasks)
    transport_depart_tasks = preprocess_text(transport_and_road_improvement_complex_depart_tasks)
    departs_corpus.append(building_depart_tasks_processed)
    departs_corpus.append(education_depart_tasks_processed)
    departs_corpus.append(energy_depart_tasks_processed)
    departs_corpus.append(industry_depart_tasks_processed)
    departs_corpus.append(transport_depart_tasks)
    print()

    keywords_count = 15
    tf_idf = compute_tfidf(departs_corpus)
    for elem in tf_idf:
        sorted_keywords = sorted(elem.items(), key=lambda x: x[1], reverse=True)
        print(sorted_keywords[:keywords_count])


def rake_keywords_extraction(tokenized_text):
    r = Rake(russian_stopwords + ['это', 'вне', 'который'])
    r.extract_keywords_from_text(tokenized_text)
    # return r.get_ranked_phrases()
    return r.get_ranked_phrases_with_scores()


def yake_keywords_extraction(text, keywords_count):
    # kw_extractor = yake.KeywordExtractor()
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


def tokenize_text(text):
    # tokens = text.split()
    tokens = mystem.lemmatize(text.lower())
    # print(tokens)
    tokens = [token for token in tokens if token not in russian_stopwords
              # and token not in departments_stopwords
              and token != " "
              and token.strip() not in punctuation]

    return " ".join(tokens)


def get_most_frequent_words(text, words_number):
    counter = Counter(text.split())
    return counter.most_common(words_number)


def get_most_frequent_words_from_list(splitted_text, words_number):
    counter = Counter(splitted_text)
    return counter.most_common(words_number)


def get_departments_tasks_list(departments_texts_path):
    departments = [f for f in os.listdir(departments_texts_path)
                   if os.path.isfile(os.path.join(departments_texts_path, f))]
    departments_tasks_list = []
    for department in departments:
        department_file_name = f"{departments_texts_path}/{department}"
        with open(department_file_name, "r", encoding='utf-8') as department_file:
            department_tasks = department_file.read().replace('\n', ' ')
            preprocessed_department = preprocess_text_tag_mystem(department_tasks)
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


if __name__ == '__main__':
    start = time.time()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = get_rusvectors_model("./model/182.zip")

    departments = get_departments_tasks_list(".\departments")
    departments_stopwords = get_departments_stopwords(departments, 15)
    departments = remove_stopwords(departments, departments_stopwords)
    print(get_departments_keywords(departments, model, 10))
    end = time.time()
    print(f'departments keywords got {end - start} sec')

    # model = gensim.models.KeyedVectors.load_word2vec_format('ruwikiruscorpora_upos_cbow_300_20_2017.bin.gz',
    #                                                         binary=True)
    # model_name = 'geowac_lemmas_none_fasttextskipgram_300_5_2020'
    # compare_result = compare_files('departments/education-tasks.txt', 'departments/building-and-architecture-tasks.txt',
    #                                model_name, 10)
    # print(compare_result)

    # print('YAKE')
    # building_text = tokenize_text(building_and_architecture_depart_tasks)
    # education_text = tokenize_text(education_depart_tasks)
    # energy_text = tokenize_text(energy_housing_and_communal_services_depart_tasks)
    # industry_text = tokenize_text(industry_innovation_and_enterprise_depart_tasks)
    # transport_text = tokenize_text(transport_and_road_improvement_complex_depart_tasks)
    # building_keywords = sorted(yake_keywords_extraction(building_text, 15))
    # print(building_keywords)
    # education_keywords = sorted(yake_keywords_extraction(education_text, 15))
    # print(education_keywords)
    # energy_keywords = sorted(yake_keywords_extraction(energy_text, 15))
    # print(energy_keywords)
    # industry_keywords = sorted(yake_keywords_extraction(industry_text, 15))
    # print(industry_keywords)
    # transport_keywords = sorted(yake_keywords_extraction(transport_text, 15))
    # print(transport_keywords)
    # print()

    # with open('article.txt', encoding='utf-8', mode='r') as file:
    #     test_text = file.read()
    #     sims = get_text_department(preprocess_text_tag_mystem(test_text),
    #                         {'building': preprocess_keywords_tag_mystem(building_keywords),
    #                          'education': preprocess_keywords_tag_mystem(education_keywords),
    #                          'energy': preprocess_keywords_tag_mystem(energy_keywords),
    #                          'industry': preprocess_keywords_tag_mystem(industry_keywords),
    #                          'transport': preprocess_keywords_tag_mystem(transport_keywords)
    #                          }, model_name)
    #     print(sims)
    # preprocess_result = preprocess_text_tag_mystem(test_text)
    # print(preprocess_result)
    # print(list(preprocess_result.keys()))

    # print('TF-IDF')
    # tf_idf_keywords_extraction()
    # print()

    # print('RAKE')
    # print(rake_keywords_extraction(preprocess_result))
    # education_preprocess = wordpunct_tokenize(education_depart_tasks) #' '.join([word for word in (education_depart_tasks.split())
    #                                  if word not in punctuation])0
    # print(education_preprocess)
    # print(rake_keywords_extraction(education_depart_tasks))
    # print(rake_keywords_extraction(tokenize_text(building_and_architecture_depart_tasks)))
    # print()

    # print('TextRank')
    # building_text_preprocess = ' '.join(preprocess_text(building_and_architecture_depart_tasks))
    # education_text_preprocess = ' '.join(preprocess_text(education_depart_tasks))
    # energy_text_preprocess = ' '.join(preprocess_text(energy_housing_and_communal_services_depart_tasks))
    # industry_text_preprocess = ' '.join(preprocess_text(industry_innovation_and_enterprise_depart_tasks))
    # transport_text_preprocess = ' '.join(preprocess_text(transport_and_road_improvement_complex_depart_tasks))
    # print(summa.keywords.keywords(building_text_preprocess))
    # print()
    # print()
    # print(summa.keywords.keywords(education_text_preprocess))
    # print()
    # print(summa.keywords.keywords(energy_text_preprocess))
    # print()
    # print(summa.keywords.keywords(industry_text_preprocess))
    # print()
    # print(summa.keywords.keywords(transport_text_preprocess))
    # print(summa.keywords.keywords(building_and_architecture_depart_tasks))
    # print()

    # print('KeyBERT')
    # print(keybert_keywords_extraction(building_and_architecture_depart_tasks))
    # print(keybert_keywords_extraction(education_depart_tasks))
    # print(keybert_keywords_extraction(energy_housing_and_communal_services_depart_tasks))
    # print(keybert_keywords_extraction(industry_innovation_and_enterprise_depart_tasks))
    # print(keybert_keywords_extraction(transport_and_road_improvement_complex_depart_tasks))
