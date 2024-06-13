import os

def rename_files_in_directory(directory_path):
    # Pobieranie listy wszystkich plików w katalogu
    files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

    # Sortowanie plików w celu utrzymania porządku
    files.sort()

    # Iterowanie przez wszystkie pliki i zmiana ich nazw
    for i, filename in enumerate(files):
        # Ustawienie nowej nazwy pliku
        new_name = f"{i}.png"
        
        # Uzyskanie pełnej ścieżki do pliku
        old_file = os.path.join(directory_path, filename)
        new_file = os.path.join(directory_path, new_name)
        
        # Zmiana nazwy pliku
        os.rename(old_file, new_file)
        print(f"Renamed: {old_file} -> {new_file}")

# Użycie funkcji z wybranym katalogiem
directory_path = 'data/5'
rename_files_in_directory(directory_path)
