#FORTY + TEN + TEN = SIXTY
#FORTYSIX
#12346789
#01234 + 35x10 + 35x10 = 45634

def sum_aritmetica(values):
    forty = int(values[0] + values[1]+ values[2] + values[3] + values[4])
    ten = int(values[3] + '5' + '0')
    sixty = int(values[4] + values[5] + values[6] + values[3]+  values[4])

    if forty + ten + ten == sixty:
        print(f"{forty} + {ten} + {ten} = {sixty}")
        return False
    return True


digits = ['1','2','3','4','6','7','8','9']
for a in range(8):
    for b in range(8):
        if b == a:
            continue
        for c in range(8):
            if c == a or c == b:
                continue
            for d in range(8):
                if d == a or d == b or d == c:
                    continue
                for e in range(8):
                    if e == a or e == b or e == c or e == d:
                        continue
                    for f in range(8):
                        if f == a or f == b or f == c or f == d or f == e:
                            continue
                        for g in range(8):
                            if g == a or g == b or g == c or g == d or g == e or g == f:
                                continue
                            for h in range(8):
                                if h == a or h == b or h == c or h == d or h == e or h == f or h == g:
                                    continue
                                test = [digits[a], digits[b], digits[c],digits[d], digits[e], digits[f],digits[g], digits[h]]
                                sum_aritmetica(test)
