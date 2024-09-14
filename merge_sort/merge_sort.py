import numpy as np

def pivot_sort(a):
    if len(a) <= 1: 
        return a
    pivot = a[0]
    fh = [x for x in a[1:] if x < pivot]
    sh = [x for x in a[1:] if x >= pivot]
    return pivot_sort(fh) + [pivot] + pivot_sort(sh)

def find_median(a, offset = 0):
    if len(a) == 1:
        return a[0]
    pivot = a[0]
    
    lower = []
    higher = []
        
    while len(a) > 0:
        if a[-1] < pivot:
            lower.append(a[-1])
        else:
            higher.append(a[-1])
        a.pop()

    if len(lower) > len(higher):
        find_median()

    return find_median()


if __name__=="__main__":
    a = np.random.randint(0,10,size=10)
    f = pivot_sort(a)
    print(f)
