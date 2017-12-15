import sys

for line in sys.stdin:
	items = line.strip().split()
	assert len(items) == 3
	print ' '.join([items[2], items[0], items[1]]);
