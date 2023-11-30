import sys
import os
import random
from math import floor

def main():
	if len(sys.argv) < 3:
		print("usage: generate_dna.py <output file> <number of characters>")
		return 1
	arr = 'CGTA'
	with open(sys.argv[1], "w") as f:
		for x in range(0 ,int(sys.argv[2])):
			f.write(arr[floor(random.random()*4)-1])


if __name__ == '__main__':
	main()