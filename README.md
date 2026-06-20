# Klasifikacija Insekata - Konvoluciona Neuralna Mreža u Pythonu

Projekat implementira model konvolucione neuralne mreže (CNN) za automatsku klasifikaciju slika insekata u 5 kategorija. Uključuje kompletnu pipeline obradu podataka: balansiranje klasa, augmentaciju, trening, evaluaciju i vizuelizaciju rezultata.

# Opis

Model prima sliku insekta dimenzija 64×64 piksela i klasifikuje je u jednu od 5 klasa (kategorija insekata). Skup podataka je podeljen na trening, validacioni i test skup, pri čemu se balansiranje klasa vrši nasumičnim uzorkovanjem uz augmentaciju duplikata kako bi se izbegle identične kopije slika u skupu.

# Arhitektura modela

```
Data Augmentation (RandomFlip, RandomRotation, RandomZoom, RandomContrast, RandomBrightness)
→ Rescaling (1/255)
→ Conv2D(32, 3×3, relu) + L2 regularizacija
→ MaxPooling2D
→ Conv2D(64, 3×3, relu) + L2 regularizacija
→ MaxPooling2D
→ Conv2D(128, 3×3, relu) + L2 regularizacija
→ MaxPooling2D
→ Conv2D(128, 3×3, relu) + L2 regularizacija
→ MaxPooling2D
→ Dropout(0.2)
→ Flatten
→ Dense(512, relu) + L2 regularizacija
→ Dense(256, relu) + L2 regularizacija
→ Dense(5, softmax)
```

# Parametri treninga

| Parametar | Vrednost |
|-----------|---------|
| Veličina slike | 64×64 px |
| Batch size | 64 |
| Podela podataka | 60% trening / 20% validacija / 20% test |
| Optimizer | Adam (lr=0.001) |
| Loss funkcija | SparseCategoricalCrossentropy |
| Maks. epohe | 50 |
| Early stopping | patience=16, monitor=val\_loss |
| L2 regularizacija | 0.001 (sve Conv2D i Dense slojeve) |

# Tok programa

1. Učitavanje slika iz `./podaci/` foldera i podela na trening/validacioni/test skup
2. Analiza i vizuelizacija distribucije klasa
3. Prikaz po jednog primera svake klase
4. Balansiranje klasa — svaka klasa dobija isti broj odabiraka (nasumično uzorkovanje)
5. Augmentacija duplikata — ponavljajuće slike se zamenjuju augmentovanim verzijama
6. Trening CNN modela sa early stopping-om
7. Vizuelizacija krivulja tačnosti i gubitka
8. Generisanje confusion matrice na trening skupu i test skupu
9. Prikaz 5 primera uspešne i 5 primera neuspešne klasifikacije sa test skupa

# Struktura podataka

```
podaci/
├── NazivKlase1/
│   ├── slika1.jpg
│   └── ...
├── NazivKlase2/
│   └── ...
└── ...
```

Klase se automatski preuzimaju iz naziva foldera unutar `podaci/` direktorijuma.

# Pokretanje

# Zahtevi
- Python 3.12+

### Instalacija

```bash
pip install -r requirements.txt
```

### Pokretanje

```bash
python main.py
```

**Napomena:** Trening traje duže (do 50 epoha). Za brzo testiranje ispisa, privremeno smanji `epochs=50` na `epochs=2` u kodu.

# Zavisnosti

```
numpy==2.4.6
matplotlib==3.11.0
tensorflow==2.21.0
keras==3.14.1
scikit-learn==1.9.0
```

