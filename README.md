# Python Spektrometr

Aplikacja do kontroli spektrometru z kamerą PixeLink i silnikami sterowymi.

## Wymagania systemowe

- **Python 3.7+**
- **Windows 10/11** (dla PixeLink SDK)
- **PixeLink SDK** (opcjonalnie, dla pełnej funkcjonalności kamery)

## Szybka instalacja

### 1. Sklonuj/pobierz projekt
```bash
git clone <repository-url>
cd Spektrometr
```

### 2. Uruchom automatyczną instalację
```bash
python setup.py
```

### 3. Uruchom aplikację
```bash
python "index copy.py"
```

## Ręczna instalacja

### 1. Zainstaluj wymagane pakiety
```bash
pip install -r requirements.txt
```

### 2. Utwórz katalogi robocze
```bash
mkdir measurement_data
mkdir backup  
mkdir logs
```

### 3. (Opcjonalnie) Zainstaluj PixeLink SDK
- Pobierz PixeLink SDK z oficjalnej strony
- Zainstaluj zgodnie z instrukcjami producenta
- Upewnij się, że `pixelinkWrapper.py` jest dostępny

## Konfiguracja

### Porty szeregowe
Edytuj `options.json`:
```json
{
  "port_x": "COM10",
  "port_y": "COM11",
  "step_x": 2,
  "step_y": 2,
  "exposure_time": 1000.0
}
```

### Kalibracja
1. Otwórz zakładkę "Camera & Controls"
2. Przeprowadź kalibrację osi X i Y
3. Zapisz kalibrację

## Struktura projektu

```
Spektrometr/
├── index copy.py          # Główna aplikacja
├── addons.py             # Dodatkowe funkcje
├── options.json          # Konfiguracja
├── requirements.txt      # Zależności Python
├── setup.py             # Skrypt instalacyjny
├── README.md            # Ta instrukcja
├── measurement_data/    # Zapisane pomiary
├── backup/              # Kopie zapasowe
├── logs/               # Logi aplikacji
└── samples/            # Przykłady PixeLink SDK
```

## Funkcjonalności

### ✅ Gotowe
- Kontrola kamery PixeLink (ekspozycja, gain)
- Automatyczne obliczanie spektrum z obrazu
- Sterowanie silnikami krokowymi
- Sekwencje pomiarowe z inteligentną synchronizacją
- Eksport danych do CSV
- Kalibracja długości fali (400-700nm)

### 🔧 Konfiguracja
- Skanowanie obszaru o zadanych wymiarach
- Automatyczna adaptacja czasu oczekiwania do ekspozycji
- Zapisywanie pełnego spektrum (2048 punktów)
- Powrót do pozycji wyjściowej po skanowaniu

## Rozwiązywanie problemów

### Brak PixeLink SDK
```
⚠️ PixeLink SDK not found - camera features will be limited
```
**Rozwiązanie**: Zainstaluj PixeLink SDK lub użyj aplikacji bez kamery

### Błędy portów szeregowych
```
ERROR: Motors are not connected!
```
**Rozwiązanie**: 
1. Sprawdź połączenia USB
2. Zaktualizuj porty w Settings → Port Settings
3. Kliknij "Refresh Ports"

### Problemy z pakietami Python
```
ModuleNotFoundError: No module named 'xyz'
```
**Rozwiązanie**:
```bash
pip install --upgrade -r requirements.txt
```

## Użycie

### Podstawowy pomiar
1. **Settings** → skonfiguruj porty i parametry skanowania
2. **Camera & Controls** → przeprowadź kalibrację  
3. **Spectrum** → uruchom sekwencję pomiarową
4. **Results** → przeglądaj i eksportuj wyniki

### Sekwencja pomiarowa
- Automatycznie skanuje zadany obszar
- Zapisuje pełne spektrum (2048 punktów) dla każdej pozycji
- Adaptuje czas oczekiwania do czasu ekspozycji kamery
- Format CSV: `x_pixel, y_pixel, spectrum_value_0, spectrum_value_1, ...`

## Kontakt

W razie problemów sprawdź:
1. Logi w konsoli aplikacji
2. Plik `options.json` - poprawność konfiguracji
3. Połączenia sprzętowe (USB, kamera)

---
**Wersja**: 2025.09
**Python**: 3.7+
**Platforma**: Windows