from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx

''' Takes in a paragraph, returns a list of sentences 

	:param user_input: The paragraph submitted by the user via Flask
	:type: str

	:returns: A list of sentences parsed from the input 
	:rtype: list[str]
'''
def process_input(user_input):
	sentences = user_input.split('. ')
	return sentences

''' Compute the similiarity score of two sentences

	:param s1: The first sentence 
	:type: str

	:param s2: The second sentence
	:type: str

	:param stopwords: The set of stopwords to eliminate
	:type: list[str]

	:returns: 
	:rtype: 
'''
def similiary_score(s1, s2, stopwords):
	s1=[w.lower() for w in s1]
	s2=[w.lower() for w in s2]

	vocabulary = list(set(s1 + s2))

	u, v = (
		[0] * len(vocabulary),
		[0] * len(vocabulary)
	)

	for w in s1:
		if w in stopwords: continue
		u[vocabulary.index(w)] += 1

	for w in s2:
		if w in stopwords: continue
		v[vocabulary.index(w)] += 1

	return 1 - cosine_distance(u, v)


def make_similiarity_matrix(sentences, stopwords=stopwords.words('english')):
	# make an m-by-m matrix where n is the size of the vocabulary
	m = len(sentences)
	Mx = np.zeros((m, m))

	# fill the entries in the matrxi with
	# corresponding similiary scores
	for i in range(m):
		for j in range(m):
			if (i == j): continue
			Mx[i][j] = similiary_score(sentences[i], sentences[j], stopwords)

	return Mx

def generate_summary(text, n=5):
	summary = []

	# Sentence-tokenize text
	sentences = process_input(text)

	# Generate similiarity matrix
	sim_mx = make_similiarity_matrix(sentences)

	# Rank sentences in the matrix
	sim_graph = nx.from_numpy_array(sim_mx)
	scores = nx.pagerank(sim_graph)

	# Sort rank + pick top ranked sentences
	ranked = sorted((
		(scores[i], s) for i,s in enumerate(sentences)), reverse=True
	)

	for i in range(n):
		summary += [ranked[i][1]]

	return '. '.join(summary)

