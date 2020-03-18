import csv
import math

indexmap = {1: 0, -1: 1} # Map from class numbers to list indexes.

def readln(file):
	""" Method: readln
	Reads and returns the next line in the provided file separated on whitespace.
	"""
	lines = file.readlines()
	for ln in lines:
		yield ln.split(' ')

def ftopvals(fdict, valindex):
	""" Method: ftopvals
	Prints out the top 5 key-value pairs for class valindex in dict.
	"""
	import operator
	result = dict(sorted(fdict.items(), key=lambda i: i[1][indexmap[valindex]], reverse=True)[:5])
	print('Top five frequencies for class %d:' % valindex)
	for i in result:
		print('\t%s %d' % (i, result[i][indexmap[valindex]]))

def tf(data):
	""" Method: tf
	Creates a .csv file containing term frequencies for words in the provided
	 data file.
	"""
	with open(data) as dd, open('tf.csv', 'w', newline='') as tf:

		tfdict = {}

		for doc in readln(dd):
			cnum = int(doc[0])
			words = doc[1:]
			for w in words:
				if w not in tfdict:
					tfdict[w] = [0, 0]
				tfdict[w][indexmap[cnum]] += 1

		tfwriter = csv.writer(tf, delimiter=',')
		tfwriter.writerow(['term', 'class 1 frequency', 'class -1 frequency'])
		for k in tfdict:
			tfwriter.writerow([k] + tfdict[k])

		ftopvals(tfdict, 1)
		ftopvals(tfdict, -1)

def tfgrep(data):
	""" Method: tfgrep
	(1) Finds most discriminating term in the tf data and
	(2) uses said term to generate a confusion matrix for the provided
	 data.
	"""
	with open(data) as dd, open('tf.csv') as tf:

		confusionmtx = [[0, 0], [0, 0]]

		tfreader = csv.reader(tf, delimiter=',')
		next(tfreader) # Skip header.

		discriminator = None

		for row in tfreader:
			yesclassf = int(row[1])
			noclassf = int(row[2])
			if not discriminator or math.fabs(yesclassf - noclassf) > math.fabs(discriminator[1] - discriminator[2]):
				discriminator = (row[0], yesclassf, noclassf)

		guessinfo = (discriminator[0], (discriminator[2] - discriminator[1] + 1)//(math.fabs(discriminator[2] - discriminator[1]) + 1))

		for doc in readln(dd):
			cnum = int(doc[0])
			words = doc[1:]
			pred = -guessinfo[1]
			if guessinfo[0] in words:
				pred = -pred
			confusionmtx[indexmap[cnum]][indexmap[pred]] += 1

		return confusionmtx

def countclasses(data):
	""" Method: countclasses
	Counts the number of instances of classes in a given data file.
	"""
	with open(data) as dd:
		classtotals = [0, 0]
		for doc in readln(dd):
			cnum = int(doc[0])
			classtotals[indexmap[cnum]] += 1
		return classtotals

def priors(data, train):
	""" Method: priors
	(1) Finds the most probable class in the training data and
	(2) uses the results to generate a confusion matrix for the
	 provided data using 0-R.
	"""
	with open(data) as dd, open(train) as td:

		confusionmtx = [[0, 0], [0, 0]]
		classtots = countclasses(train)
		guessindex = classtots.index(max(classtots))

		for doc in readln(dd):
			cnum = int(doc[0])
			confusionmtx[indexmap[cnum]][guessindex] += 1

		return confusionmtx

def csvgetdata(csvfile, func):
	""" Method: csvgetdata
	Reads data from a specified csv file and then executes a given
	 function on said data and returns the result.
	"""
	with open(csvfile) as cf:
		cfreader = csv.reader(cf, delimiter=',')
		next(cfreader) # Skip header.
		return func(cfreader)

def mnb(data, train):
	""" Method: mnb
	Predicts the most likely class given a document using the tf data 
	 and the Multinomial Naive Bayes Model.
	"""
	with open(data) as dd:

		# priorsmtx = priors(data, train)
		confusionmtx = [[0, 0], [0, 0]]

		classtots = countclasses(train)
		cyes = classtots[0]/sum(classtots)
		cno = classtots[1]/sum(classtots)

		terms = csvgetdata('tf.csv', lambda reader: [r[0] for r in reader])
		freqs = csvgetdata('tf.csv', lambda reader: [[int(r[1]), int(r[2])] for r in reader])
		freqyes = sum([f[0] for f in freqs])
		freqno = sum([f[1] for f in freqs])
		condprobs = [[(1 + f[0])/(1 + freqyes), (1 + f[1])/(1 + freqno)] for f in freqs]
		probdict = dict(zip(terms, condprobs))

		for doc in readln(dd):
			classprobs = [cyes, cno]
			cnum = int(doc[0])
			words = doc[1:]
			cmpwords = {}
			for w in words:
				if w in probdict:
					if w not in cmpwords:
						cmpwords[w] = 1
					else:
						cmpwords[w] += 1
			for i in range(0, len(classprobs)):
				classprobs[i] *= sum(list(map(lambda item: math.pow(probdict[item[0]][i], item[1]), cmpwords.items())))
			guessindex = classprobs.index(max(classprobs))
			confusionmtx[indexmap[cnum]][guessindex] += 1

		return confusionmtx

def df(data):
	""" Method: df
	Creates a .csv file containing document frequencies for words in the 
	 provided data file.
	"""
	with open(data) as dd, open('df.csv', 'w', newline='') as df:

		dfdict = {}

		for doc in readln(dd):
			tempdict = {}
			cnum = int(doc[0])
			words = doc[1:]
			for w in words:
				if w not in dfdict:
					dfdict[w] = [0, 0]
				if w not in tempdict:
					tempdict[w] = [0, 0]
				tempdict[w][indexmap[cnum]] += 1
				if tempdict[w][indexmap[cnum]] <= 1:
					dfdict[w][indexmap[cnum]] += 1

		tfwriter = csv.writer(df, delimiter=',')
		tfwriter.writerow(['term', 'class 1 frequency', 'class -1 frequency'])
		for k in dfdict:
			tfwriter.writerow([k] + dfdict[k])

		ftopvals(dfdict, 1)
		ftopvals(dfdict, -1)

def nb(data, train):
	""" Method: nb
	Predicts the most likely class given a document using the df data 
	 and the Multi-variate Bernoulli Model.
	"""
	with open(data) as dd:

		# priorsmtx = priors(data, train)
		confusionmtx = [[0, 0], [0, 0]]

		classtots = countclasses(train)
		cyes = classtots[0]/sum(classtots)
		cno = classtots[1]/sum(classtots)

		terms = csvgetdata('df.csv', lambda reader: [r[0] for r in reader])
		freqs = csvgetdata('df.csv', lambda reader: [[int(r[1]), int(r[2])] for r in reader])

		for doc in readln(dd):
			condprobs = [[1 - ((1 + f[0])/(2 + classtots[0])), 1 - ((1 + f[1])/(2 + classtots[1]))] for f in freqs]
			probdict = dict(zip(terms, condprobs))
			classprobs = [cyes, cno]
			cnum = int(doc[0])
			words = doc[1:]
			checked = []
			for w in words:
				if w not in checked:
					checked.append(w)
					if w in probdict:
						val = probdict[w]
						val[0] = 1 - val[0]
						val[1] = 1 - val[1]
			for i in range(0, len(classprobs)):
				classprobs[i] += sum(list(map(lambda item: math.log(item[1][i]), probdict.items())))
			guessindex = classprobs.index(max(classprobs))
			print(classprobs, guessindex)
			confusionmtx[indexmap[cnum]][guessindex] += 1

		return confusionmtx

def tfmine(data):
	""" Method: tfmine
	A modified version of the tf function which ignores words in the
	 provided data with hyphens in them.
	"""
	with open(data) as dd, open('tf.csv', 'w', newline='') as tf:

		tfdict = {}

		for doc in readln(dd):
			cnum = int(doc[0])
			words = doc[1:]
			for w in words:
				if '-' not in w:
					if w not in tfdict:
						tfdict[w] = [0, 0]
					tfdict[w][indexmap[cnum]] += 1

		tfwriter = csv.writer(tf, delimiter=',')
		tfwriter.writerow(['term', 'class 1 frequency', 'class -1 frequency'])
		for k in tfdict:
			tfwriter.writerow([k] + tfdict[k])

def mine(data, train):
	""" Method: mine
	Uses a modified version of tf.csv (see tfmine) alongside the 
	 Multinomial Naive Bayes Model to predict the most likely class.
	"""
	print('mine confusion matrix for %s:' % data, mnb(data, train))

# its yer boi main
if __name__ == '__main__':

	import sys
	args = sys.argv[1:]

	if len(args) != 3:
		print('HELP: classify.py takes three arguments',
			'\t(1) a file of training data',
			'\t(2) a file of testing data',
			'\t(3) a method to execute',
			sep='\n')
		sys.exit(1)
	if args[2] == 'tf':
		tf(args[0])
	elif args[2] == 'tfgrep':
		print('tfgrep confusion matrix for %s:' % args[0], tfgrep(args[0]))
		print('tfgrep confusion matrix for %s:' % args[1], tfgrep(args[1]))
	elif args[2] == 'priors':
		print('priors confusion matrix for %s:' % args[0], priors(args[0], args[0]))
		print('priors confusion matrix for %s:' % args[1], priors(args[1], args[0]))
	elif args[2] == 'mnb':
		print('mnb confusion matrix for %s:' % args[0], mnb(args[0], args[0]))
		print('mnb confusion matrix for %s:' % args[1], mnb(args[1], args[0]))
	elif args[2] == 'df':
		df(args[0])
	elif args[2] == 'nb':
		print('nb confusion matrix for %s:' % args[0], nb(args[0], args[0]))
		print('nb confusion matrix for %s:' % args[1], nb(args[1], args[0]))
	elif args[2] == 'mine':
		print('The mine method reduces the dictionary of words by',
			'removing words that start with tags. During testing',
			'I noticed that the words with the top frequencies',
			'were always a part of the body of the ad documents.',
			'Because of this, I thought it may improve performance',
			'to remove other words, deeming them unnecessary.', sep='\n')
		tfmine(args[0])
		mine(args[0], args[0])
		mine(args[1], args[0])
	else:
		print('HELP: the third argument must be a valid method',
			'\tOPTIONS: tf, tfgrep, priors, mnb, df, nb, mine', sep='\n')
		sys.exit(1)
