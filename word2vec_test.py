import requests

from nltk.corpus import stopwords
from pymystem3 import Mystem
from string import punctuation

mystem = Mystem()
russian_stopwords = stopwords.words("russian")


def preprocess_text(text):
    tokens = mystem.lemmatize(text.lower())
    print(tokens)
    processed = mystem.analyze(text)
    print(processed)
    tokens = [token for token in tokens if token not in russian_stopwords
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
            if lemma not in russian_stopwords:
                pos = token["analysis"][0]["gr"].split(',')[0]
                pos = pos.split('=')[0].strip()
                if pos in mapping:
                    tag = lemma + '_' + mapping[pos]
                    tagged.append(tag)  # здесь мы конвертируем тэги
                else:
                    tag = lemma + '_X'
                    tagged.append(tag)
        except KeyError:
            continue
    return list(set(tagged))


def api_similarity(model_name, word1, word2):
    url = '/'.join(['https://rusvectores.org', model_name, word1 + '__' + word2, 'api', 'similarity/'])
    request = requests.get(url, stream=True)
    similarity = request.text.split('\t')[0]
    return float(similarity)


def get_similarities_list(text1_words, text2_words, model_name):
    result = list()
    for text1_word in text1_words:
        mid_similarity = 0
        for text2_word in text2_words:
            # similarity = model.similarity(text1_word, text2_word)
            # mid_similarity += similarity
            similarity = api_similarity(model_name, text1_word, text2_word)
            print(text1_word, text2_word, similarity)
            mid_similarity += similarity
        mid_similarity_result = mid_similarity / len(text2_words)
        print(text1_word, mid_similarity_result)
        print()
        result.append(mid_similarity_result)

    return result


# returns key words for text1
def compare_texts(text1, text2, model_name, key_words_count):
    text1_words = preprocess_text_tag_mystem(text1)
    print(text1_words)
    text2_words = preprocess_text_tag_mystem(text2)
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


if __name__ == '__main__':
    # model = gensim.models.KeyedVectors.load_word2vec_format('ruwikiruscorpora_upos_cbow_300_20_2017.bin.gz',
    #                                                         binary=True)
    model_name = 'geowac_lemmas_none_fasttextskipgram_300_5_2020'
    result = compare_files('bulding-and-architecture-tasks.txt', './education-tasks.txt', model_name, 10)
    print(result)
