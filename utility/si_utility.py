# Angka Acak dengan interval
## Fungsi untuk menghasilkan angka acak dengan interval dari min_val sampai max_val (inklusif). Mengembalikan numpy
## array dengan ukuran row x col.
def interval_random(row, col, min_val, max_val):
    ran = abs(max_val - min_val)
    return ran * np.random.random((row, col)) + min_val

# Fungsi Objective Maximize
## Fungsi objective dengan tujuan mencari nilai fitness paling besar. Mengembalikan nilai absolute dari fitness
def maximize(val):
    return abs(val)

# Fungsi Objective Minimize
## Fungsi objective dengan tujuan mencari nilai fitness paling kecil. Mengembalikan nilai absoulute dari 1/fitness
def minimize(val):
    return abs(1/(1+val))

# Pemotongan Nilai
## Fungsi untuk memastikan nilai berada pada rentang min_value sampai max_value (inklusif).
def value_clip(val, min_value, max_value):
    if val < min_value:
        return min_value
    if val > max_value:
        return max_value
    return val
