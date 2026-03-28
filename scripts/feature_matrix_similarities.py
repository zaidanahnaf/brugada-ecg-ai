import pandas as pd

f1 = pd.read_csv('features/feature_matrix_new.csv')
f2 = pd.read_csv('features/feature_matrix_frans.csv')

print("Shape f1:", f1.shape)
print("Shape f2:", f2.shape)

print("Shape sama:", f1.shape == f2.shape)

cols_f1 = set(f1.columns)
cols_f2 = set(f2.columns)

print("Jumlah kolom f1:", len(cols_f1))
print("Jumlah kolom f2:", len(cols_f2))

print("Kolom sama (set):", cols_f1 == cols_f2)

# Kalau ada yang beda
print("Kolom hanya di f1:", cols_f1 - cols_f2)
print("Kolom hanya di f2:", cols_f2 - cols_f1)

same_order = list(f1.columns) == list(f2.columns)

print("Urutan kolom sama:", same_order)

# Cari posisi yang beda
diff_positions = [
    i for i, (c1, c2) in enumerate(zip(f1.columns, f2.columns))
    if c1 != c2
]

print("Jumlah posisi kolom beda:", len(diff_positions))

# Tampilkan contoh
for i in diff_positions[:10]:
    print(f"Index {i}: f1={f1.columns[i]} | f2={f2.columns[i]}")

dtype_diff = []

for col in f1.columns:
    if col in f2.columns:
        if f1[col].dtype != f2[col].dtype:
            dtype_diff.append((col, f1[col].dtype, f2[col].dtype))

print("Jumlah kolom beda tipe:", len(dtype_diff))
print("Contoh:", dtype_diff[:10])

same_index = f1.index.equals(f2.index)
print("Index sama:", same_index)