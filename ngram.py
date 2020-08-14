import os
import pandas as pd
import numpy as np
import requests
import time
import re

def tokenize(book_string):
    rx = r"\w+|[^\w\s]"
    split_all = re.findall(rx, book_string)
    no_whiteblocks = re.sub(r'\n{3,}', '\n\n', book_string)
    replace_with_x0='\x02' + re.sub("\n\n", "\x03\x02", no_whiteblocks) + '\x03'


    result=re.sub("\\n\\n", "\x03", no_whiteblocks)

    if result[0:4]!="\x03":
        result = re.sub("\x03", "\x03 \x02", result)
        result = "\x02"+result
        result = re.findall(rx, result)
    else:
        result = re.sub("\x03", "\x03 \x02", result)
        result = re.findall(rx, result).remove("\x03")

    
    if result[len(result)-1]!="\x03":
        result.append("\x03")
    

    return result

    


class UniformLM(object):

    def __init__(self, tokens):

        """

        Initializes a Uniform languange model using a

        list of tokens. It trains the language model

        using `train` and saves it to an attribute

        self.mdl.

        """

        self.mdl = self.train(tokens)

        

    def train(self, tokens):

        """

        Trains a uniform language model given a list of tokens.

        The output is a series indexed on distinct tokens, and

        values giving the (uniform) probability of a token occuring

        in the language.


        :Example:

        >>> tokens = tuple('one one two three one two four'.split())

        >>> unif = UniformLM(tokens)

        >>> isinstance(unif.mdl, pd.Series)

        True

        >>> set(unif.mdl.index) == set('one two three four'.split())

        True

        >>> (unif.mdl == 0.25).all()

        True

        """

        

        return pd.Series(pd.value_counts(np.unique(tokens),normalize=True))

    

    def probability(self, words):

        """

        probability gives the probabiliy a sequence of words

        appears under the language model.

        :param: words: a tuple of tokens

        :returns: the probability `words` appears under the language

        model.


        :Example:

        >>> tokens = tuple('one one two three one two four'.split())

        >>> unif = UniformLM(tokens)

        >>> unif.probability(('five',))

        0

        >>> unif.probability(('one', 'two')) == 0.0625

        True

        """

        joint_prob = 1
        for i in words:
            try:
                prob = self.mdl[i]
                joint_prob *= prob
            except KeyError:
                return 0

        return joint_prob

        

    def sample(self, M):

        """

        sample selects tokens from the language model of length M, returning

        a string of tokens.


        :Example:

        >>> tokens = tuple('one one two three one two four'.split())

        >>> unif = UniformLM(tokens)

        >>> samp = unif.sample(1000)

        >>> isinstance(samp, str)

        True

        >>> len(samp.split()) == 1000

        True

        >>> s = pd.Series(samp.split()).value_counts(normalize=True)

        >>> np.isclose(s, 0.25, atol=0.05).all()

        True

        """

        return " ".join(np.random.choice(self.mdl.index.values,M,p=self.mdl.values,replace=True))


            

# ---------------------------------------------------------------------

# Question #4

# ---------------------------------------------------------------------



class UnigramLM(object):

    

    def __init__(self, tokens):

        """

        Initializes a Unigram languange model using a

        list of tokens. It trains the language model

        using `train` and saves it to an attribute

        self.mdl.

        """

        self.mdl = self.train(tokens)

    

    def train(self, tokens):

        """

        Trains a unigram language model given a list of tokens.

        The output is a series indexed on distinct tokens, and

        values giving the probability of a token occuring

        in the language.


        :Example:

        >>> tokens = tuple('one one two three one two four'.split())

        >>> unig = UnigramLM(tokens)

        >>> isinstance(unig.mdl, pd.Series)

        True

        >>> set(unig.mdl.index) == set('one two three four'.split())

        True

        >>> unig.mdl.loc['one'] == 3 / 7

        True

        """


        return pd.Series(pd.value_counts(tokens,normalize=True))

    

    def probability(self, words):

        """

        probability gives the probabiliy a sequence of words

        appears under the language model.

        :param: words: a tuple of tokens

        :returns: the probability `words` appears under the language

        model.


        :Example:

        >>> tokens = tuple('one one two three one two four'.split())

        >>> unig = UnigramLM(tokens)

        >>> unig.probability(('five',))

        0

        >>> p = unig.probability(('one', 'two'))

        >>> np.isclose(p, 0.12244897959, atol=0.0001)

        True

        """

        

        joint_prob = 1

        for i in words:

            try:

                prob = self.mdl[i]

                joint_prob *= prob

            except KeyError:

                return 0

        

        return joint_prob

        

    def sample(self, M):

        """

        sample selects tokens from the language model of length M, returning

        a string of tokens.


        >>> tokens = tuple('one one two three one two four'.split())

        >>> unig = UnigramLM(tokens)

        >>> samp = unig.sample(1000)

        >>> isinstance(samp, str)

        True

        >>> len(samp.split()) == 1000

        True

        >>> s = pd.Series(samp.split()).value_counts(normalize=True).loc['one']

        >>> np.isclose(s, 0.41, atol=0.05).all()

        True

        """


        return " ".join(np.random.choice(self.mdl.index.values,M,p=self.mdl.values,replace=True))

        

    

# ---------------------------------------------------------------------

# Question #5,6,7,8

# ---------------------------------------------------------------------


class NGramLM(object):

    

    def __init__(self, N, tokens):

        """

        Initializes a N-gram languange model using a

        list of tokens. It trains the language model

        using `train` and saves it to an attribute

        self.mdl.

        """


        self.N = N

        ngrams = self.create_ngrams(tokens)


        self.ngrams = ngrams

        self.mdl = self.train(ngrams)


        if N < 2:

            raise Exception('N must be greater than 1')

        elif N == 2:

            self.prev_mdl = UnigramLM(tokens)

        else:

            mdl = NGramLM(N-1, tokens)

            self.prev_mdl = mdl


    def create_ngrams(self, tokens):

        """

        create_ngrams takes in a list of tokens and returns a list of N-grams. 

        The START/STOP tokens in the N-grams should be handled as 

        explained in the notebook.


        :Example:

        >>> tokens = tuple('\x02 one two three one four \x03'.split())

        >>> bigrams = NGramLM(2, [])

        >>> out = bigrams.create_ngrams(tokens)

        >>> isinstance(out[0], tuple)

        True

        >>> out[0]

        ('\\x02', 'one')

        >>> out[2]

        ('two', 'three')

        """

        result = []

        for i in range(len(tokens)-(self.N-1)):

            result.append(tuple(tokens[i:i+self.N]))

        return result

        

    def train(self, ngrams):

        """

        Trains a n-gram language model given a list of tokens.

        The output is a dataframe indexed on distinct tokens, with three

        columns (ngram, n1gram, prob).


        :Example:

        >>> tokens = tuple('\x02 one two three one four \x03'.split())

        >>> bigrams = NGramLM(2, tokens)

        >>> set(bigrams.mdl.columns) == set('ngram n1gram prob'.split())

        True

        >>> bigrams.mdl.shape == (6, 3)

        True

        >>> bigrams.mdl['prob'].min() == 0.5

        True

        """


        temp_ngram = pd.Series(ngrams)

        ngram = pd.Series(temp_ngram.unique())

        temp1=dict(pd.value_counts(ngrams,sort=False))

        # n-1 gram counts C(w_1, ..., w_(n-1))

        n1gram = ngram.apply(lambda x: x[0:self.N-1])

        temp2= dict(pd.value_counts(temp_ngram.apply(lambda x: x[0:self.N-1]),sort=False))

        # Create the conditional probabilities

        prob = ngram.map(temp1) / n1gram.map(temp2)

        # Put it all together

        out = pd.DataFrame({"ngram":ngram,"n1gram":n1gram,"prob":prob})

        return out

    

    def probability(self, words):

        """

        probability gives the probabiliy a sequence of words

        appears under the language model.

        :param: words: a tuple of tokens

        :returns: the probability `words` appears under the language

        model.


        :Example:

        >>> tokens = tuple('\x02 one two one three one two \x03'.split())

        >>> bigrams = NGramLM(2, tokens)

        >>> p = bigrams.probability('two one three'.split())

        >>> np.isclose(p, (1/4)*(1/2)*(1/3))

        True

        >>> bigrams.probability('one two five'.split()) == 0

        True

        """

        

        n = self.N

        result = 1

        prev = self.prev_mdl

        for i in range(len(words)):

            prev = self.prev_mdl

            for j in range(n-i-2):

                prev = prev.prev_mdl

            if i < (n - 1):

                if (n == 2) | (i == 0):

                    result *= prev.probability(tuple(words[:i+1]))

                elif n > 2:

                    print(prev)

                    result *= prev.mdl.prob.loc[prev.mdl.ngram == tuple(words[0:i+1])].values[0]

            else:

                try:

                    result *= self.mdl.prob.loc[self.mdl.ngram == tuple(words[i-(n-1):i+1])].values[0]

                except IndexError:

                    result *= 0

                    return result

        return result


    def sample(self, M):

        """

        sample selects tokens from the language model of length M, returning

        a string of tokens.


        :Example:

        >>> tokens = tuple('\x02 one two three one four \x03'.split())

        >>> bigrams = NGramLM(2, tokens)

        >>> samp = bigrams.sample(3)

        >>> len(samp.split()) == 4  # don't count the initial START token.

        True

        >>> samp[:2] == '\\x02 '

        True

        >>> set(samp.split()) <= {'\\x02', '\\x03', 'one', 'two', 'three', 'four'}

        True

        """

            

        # Use a helper function to generate sample tokens of length `length`

        def helper(self, tokens):

            prev = self

            n = self.N 

            while n > (len(tokens) + 1):

                prev = self.prev_mdl

                n = prev.N

            rows = prev.mdl[prev.mdl['n1gram'] == tokens[-(self.N-1):]]

            

            try:

                out = tokens + (np.random.choice(rows.ngram, p=rows.prob,replace=True)[-1], )

            except ValueError:

                out = tokens + ('\x03',)

            return out

        

        # Transform the tokens to strings

        if M == 1:

            return '\x02'

        start = helper(self, ('\x02', ))

        for i in range(M-1):

            start = helper(self, start)

        return " ".join(start)

    