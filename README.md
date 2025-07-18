# Obesity Prediction using Machine Learning 🧠🥗

Projekt wykorzystujący dane dotyczące stylu życia i nawyków żywieniowych do przewidywania poziomu otyłości u użytkownika. Został stworzony przy użyciu bibliotek Pythona takich jak `pandas`, `scikit-learn`, `seaborn` oraz z prostym interfejsem API w `Flask`.

## 📊 Dane

Zbiór danych zawiera informacje takie jak:

- Płeć (`Gender`)
- Wiek, wzrost, waga
- Historia otyłości w rodzinie
- Nawyki żywieniowe i styl życia (spożycie warzyw, aktywność fizyczna, picie wody itd.)
- Środek transportu (`MTRANS`)
- Cel: **Obesity Level** – kategoria otyłości (`Normal`, `Obesity_Type_I`, `Insufficient_Weight`, itp.)

Źródło danych: Kaggle

## ⚙️ Użyte technologie

- Python 3.12
- Pandas, NumPy
- Scikit-learn (modelowanie, preprocessing)
- Matplotlib & Seaborn (wizualizacje)
- Flask (API)
- HTML + CSS (frontend)
- joblib (zapisywanie modeli i encoderów)

## 📈 Etapy pracy

1. **Eksploracja danych (EDA)**  
   - Sprawdzenie braków, typów danych
   - Analiza korelacji
   - Wizualizacje rozkładów i zależności

2. **Przetwarzanie danych**  
   - Kodowanie zmiennych kategorycznych (`OrdinalEncoder`, `LabelEncoder`)
   - Skalowanie zmiennych (opcjonalnie)
   - Podział na zbiór treningowy i testowy

3. **Modelowanie**  
   - Trening wybranego modelu (np. `RandomForestClassifier`, `GradientBoosting`, itp.)
   - Ocena: accuracy, classification report

4. **Zapis modelu i encoderów**  
   - `joblib.dump()` dla modelu oraz encoderów

5. **API w Flask**  
   - Endpoint `/prediction` przyjmujący dane formularza i zwracający klasę otyłości

6. **Prosty frontend**  
   - Formularz HTML do wprowadzania danych użytkownika
   - Wynik predykcji zwracany po stronie klienta

## 🚀 Jak uruchomić projekt lokalnie

1. Sklonuj repozytorium:
   ```bash
   git clone https://github.com/MaciekVys/ObesityPrediction.git
   cd ObesityPrediction
   ```
## 👤 Autor
Maciej Wysocki

LinkedIn: https://www.linkedin.com/in/maciej-wysocki-b13826267/
