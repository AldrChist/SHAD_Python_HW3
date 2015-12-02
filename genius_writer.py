from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import itertools
from collections import Counter
import sys
import os
import pickle
import time
import copy

special_symbols = {'\n' : 'NEWLINE',
                   '.'  : 'STOP1',
                   '!' : 'STOP2',
                   '?' : 'STOP3',
                   ',' : 'SEPARATOR1',
                   '--' : 'SEPARATOR2',
                   ';' : 'SEPARATOR3'}

class RawTextParser:
    def __init__(self):
        pass
        
    def load_data(self,data_directory):
        books = []
        total_counter = Counter()

        for book in os.listdir(data_directory):
            file_descriptor = open(data_directory + book, 'r')
            raw_book_text = file_descriptor.read()
            file_descriptor.close()
            books.append(raw_book_text)

        print 'All books are loaded'
        sys.stdout.flush()

        raw_accumulated_text = special_symbols['\n'].join(books)
        accumulated_text = self.__filter_text(raw_accumulated_text)

        print 'Text is filtered'
        sys.stdout.flush()

        total_counter = Counter(accumulated_text.split(' '))
        words_number = len(accumulated_text.split(' '))
        print "Total words = {}; Unical words = {}".format(words_number, len(total_counter))
        return accumulated_text
    
    def __is_correct_symbol(self, c):
        return c.isalpha() or c.isdigit() or (c == '-') or (c == "'")

    def __remove_duplicates(self, text, substring):
        text = filter(lambda word: len(word) > 0, text.split(substring))
        return substring.join(text)

    def __filter_text(self, text):
        text = self.__remove_duplicates(text, '\n')
        for (key, value) in special_symbols.items():
            text = text.replace(key, ' ' + value + ' ')
        text = ''.join(map(lambda c: c if self.__is_correct_symbol(c) else ' ', text))
        text = self.__remove_duplicates(text, ' ')
        text = self.__transform_caps_to_lower(text)
        text = self.__remove_incorrect_words(text)
        return text

    def __transform_caps_to_lower(self, text):
        text = text.split(' ');
        words = set(text)
        for (i, word) in enumerate(text):
            if (word.lower() in words) and (word != 'I'):
                text[i] = word.lower()
        return ' '.join(text)

    def __remove_incorrect_words(self, text):
        text = text.split(' ')
        text = filter(lambda word: word[0].isalpha(), text)
        text = ' '.join(text)
        return text
		

class TextNGramDistribution:
    def __init__(self, min_context_counter = 1):
        self.__word2id = {}
        self.__id2word = {}
        self.__hist_unigram = {}
        self.__hist_bigram = {}
        self.__hist_trigram = {}
        self.set_min_counter(min_context_counter)
        
    def print_hist_length(self):
        print len(self.__hist_unigram)
        print len(self.__hist_bigram)
        print len(self.__hist_trigram)
                
    def __repr__(self):
        print 'Word2ID:', self.__word2id
        print 'Id2Word:', self.__id2word
        print 'UNIGRAM:', self.__hist_unigram
        print 'BIGRAM:', self.__hist_bigram
        print 'TRIGRAM:', self.__hist_trigram
        return ''
    
    def set_min_counter(self, min_context_counter):
        self.__min_context_counter = min_context_counter
    
    def fit(self, text):
        # Build dictionaries and histograms for 1-2-3-gramms
        # corresponding to a given sequence of tokens
        self.__setup_words_ids(text)
        self.__build_histograms(text)
        

    def get_predictions(self, context):
        current_hist = {}
        if len(context) > 2:
            context = context[-2:]
        context = self.text_to_ids(context)
        if len(context) == 2:
            context_counter = self.__hist_bigram.get(tuple(context), 0)
            if  context_counter < self.__min_context_counter:
                context = [context[1]]
            else:
                current_hist = self.__hist_trigram
        if len(context) == 1:
            context_counter = self.__hist_unigram.get(tuple(context), 0)
            if context_counter < self.__min_context_counter:
                context = []
            else:
                current_hist = self.__hist_bigram
        if len(context) == 0:
            current_hist = self.__hist_unigram
        
        ids = self.__word2id.values()
        words = self.__word2id.keys()
        possible_continue = [tuple(context + [current_id]) for current_id in ids]
        probabilities = [current_hist.get(ngram, 0) for ngram in possible_continue]
        probabilities = np.array(probabilities, dtype = float)
        probabilities = probabilities + 1 / len(probabilities)
        probabilities /= sum(probabilities)
        
        return (words, probabilities)
        
        
    def save(self, filename):
        file_descriptor = open(filename, "wb")
        
        pickle.dump(self.__word2id, file_descriptor, protocol = 2)
        pickle.dump(self.__id2word, file_descriptor, protocol = 2)
        print 'dictionaries saved'
        sys.stdout.flush()
        
        pickle.dump(self.__hist_unigram, file_descriptor, protocol = 2)
        print 'hist1 saved'
        sys.stdout.flush()
        
        pickle.dump(self.__hist_bigram, file_descriptor,  protocol = 2)
        print 'hist2 saved'
        sys.stdout.flush()
        
        pickle.dump(self.__hist_trigram, file_descriptor,  protocol = 2)
        print 'hist3 saved'
        sys.stdout.flush()
        
        file_descriptor.close()
        
    def load(self, filename):
        file_descriptor = open(filename, "rb")
        self.__word2id = pickle.load(file_descriptor)
        self.__id2word = pickle.load(file_descriptor)
        self.__hist_unigram = pickle.load(file_descriptor)
        self.__hist_bigram = pickle.load(file_descriptor)
        self.__hist_trigram = pickle.load(file_descriptor)
        file_descriptor.close()
        
    def __setup_words_ids(self, text):
        unique_words = list(set(text))
        for (index, word) in enumerate(unique_words):
            self.__word2id[word] = index
            self.__id2word[index] = word
        return
    
    
    def text_to_ids(self, text):
        return map(lambda word: self.__word2id[word], text)
    
    
    def ids_to_text(self, ids):
        return map(lambda current_id: self.__id2word[current_id], ids)
    
    
    def __build_histograms(self, text):
        indexes = self.text_to_ids(text)
        
        self.__hist_unigram = Counter(map(lambda item: tuple([item]), indexes))
        self.__hist_bigram = Counter(zip(indexes[:-1], indexes[1:]))
        self.__hist_trigram = Counter(zip(indexes[:-2],indexes[1:-1],indexes[2:]))
		
		
class PoemWriter:
    def __init__(self):
        self.status_values = ["NEWLINE", "STOP", "SEPARATOR", "TEXT"]
        
        self.available_status_jumps = {"NEWLINE" : ["TEXT"], \
                          "STOP" : ["NEWLINE", "TEXT"], \
                          "SEPARATOR" : ["TEXT"], \
                          "TEXT" : ["STOP","SEPARATOR","TEXT"]}
        self.spec_symbol_representation = {"NEWLINE" : "\n   ", \
                                           "STOP1" : '.', \
                                           "STOP2" : '!', \
                                           "STOP3" : '?', \
                                           "SEPARATOR1" : ',', \
                                           "SEPARATOR2" : ' --', \
                                           "SEPARATOR3" : ';'}
        
    def __get_token_status(self, token):
        if token[-1].isdigit():
            token = token[:-1]
        if token in self.status_values:
            return token
        return "TEXT"
    
    def __get_token_representation(self, token, context_status):
        if token in self.spec_symbol_representation.keys():
            return self.spec_symbol_representation[token]
        if context_status in ["NEWLINE", "STOP"]:
            token = token[0].upper() + token[1:]
        return ' ' + token
    
    def generate_poem(self, ngram_distribution, poem_length = 100):
        poem = []
        t_start = time.time()

        context_status = "NEWLINE"
        for i in range(poem_length):
            (words, weights) = ngram_distribution.get_predictions(poem)
            current_status = "INCORRECT_STATUS"
            current_word = ""
            while current_status not in self.available_status_jumps[context_status]:
                new_word_id = np.random.choice(len(words), 1, p = weights)
                current_word = words[new_word_id[0]]
                current_status = self.__get_token_status(current_word)
                    
            poem += [current_word]
            context_status = current_status
            print i, time.time() - t_start
            
        context_status = "NEWLINE"
        for (i, word) in enumerate(poem):
            poem[i] = self.__get_token_representation(word, context_status)
            context_status = self.__get_token_status(word)
        return "   " + ''.join(poem)
		
		
if __name__ == "__main__":
	data_reader = RawTextParser()
	data_directory = './data/'
	accumulated_text = data_reader.load_data(data_directory)

	ngram_distribution = TextNGramDistribution(1)
	ngram_distribution.fit(accumulated_text.split(' '))

	# WARNING: about 200Mb
	ngram_distribution.save('filename.pcl')
	ngram_distribution = TextNGramDistribution(2)
	ngram_distribution.load('filename.pcl')

	writer = PoemWriter()
	poem_length = 10000
	poem = writer.generate_poem(ngram_distribution, poem_length)
