
a = range(100000)
b = {str(i):i for i in a}
c = [b[str(i)] for i in a]