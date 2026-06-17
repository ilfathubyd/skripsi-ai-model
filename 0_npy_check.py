import numpy as np
from pathlib import Path

folder_siap_train = Path("/mnt/7024399524395EF2/proyek_tcnn/dataset_siap_train")

print(f"Membaca file di dalam folder: {folder_siap_train.name}\n")

for file_npy in folder_siap_train.glob("*.npy"):
    # Buka file .npy
    data = np.load(file_npy)
    
    print("-" * 50)
    print(f"Nama File : {file_npy.name}")
    print(f"Bentuk/Shape: {data.shape}")
    
    # Tampilkan cuplikan isi baris pertama (data ke-0) saja agar terminal tidak kepenuhan
    if len(data) > 0:
        print(f"Cuplikan isi data pertama (Indeks ke-0):")
        if 'y_' in file_npy.name:
            # Jika ini file label (y_train atau y_test)
            print(data[0]) 
            print("(Ini adalah format One-Hot Encoding)")
        else:
            # Jika ini file fitur (X_train atau X_test), tampilkan sedikit dari frame pertama
            print(data[0][0][:10], "... (dst)")
    
print("-" * 50)
print("Selesai mengecek seluruh file.")