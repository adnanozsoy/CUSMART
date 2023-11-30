#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import os

if len(sys.argv) < 4:
	print("""
 dupefile.py: text duplication script to generate larger test samples

 Usage: dupefile.py <infile> <outfile> [multiplier]

	<infile>        path to input file
	<outfile>       path to output file 
	[multiplier]    This value can be a decimal for simple size multiplication
	                Or can include "M" for megabytes and "G" for gigabytes.
 Examples:

	dupefile.py in out 5  	copies in file 5 times to out file
	dupefile.py in out 50M	copies in file to out file until it is at least 50 megabytes
	dupefile.py in out 1G	copies in file to out file until it is at least 1 gigabytes

""");
	exit(1)


def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

infile = sys.argv[1]
infile_size = os.path.getsize(sys.argv[1])

outfile = sys.argv[2]
multiplier_arg = sys.argv[3].upper()

factor = 1
if   multiplier_arg[-1] == "M":
	outfile_size = int(multiplier_arg[:-1]) * pow(2,20)
elif multiplier_arg[-1] == "G":
	outfile_size = int(multiplier_arg[:-1]) * pow(2,30)
else:
	outfile_size = int(multiplier_arg) * infile_size

fi = open(infile, "rb")
fo = open(outfile, "wb")

infile_text = fi.read()
written = float(0)
while written < outfile_size:
	print("%{:<3.0f} done ({})".format(written/outfile_size*100, sizeof_fmt(written)))
	fo.write(infile_text)
	written += infile_size
print("%100 done ({})".format(sizeof_fmt(written)))
print('File "{}" generated successfully.'.format(outfile))
fi.close()
fo.close()
