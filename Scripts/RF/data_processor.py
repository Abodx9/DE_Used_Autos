# Scripts/data_processor.py

import pandas as pd
import numpy as np


class DataProcessor:
    """
    Eine Klasse zur Durchführung von Datenbereinigung und Feature Engineering.

    Attributes:
        df (pandas.DataFrame): Der zu verarbeitende DataFrame.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialisiert den DataProcessor mit einem DataFrame.
        Eine Kopie des DataFrames wird erstellt, um den Original-DataFrame nicht zu modifizieren.
        """
        self.df = df.copy()
        print("DataProcessor initialisiert. Eine Arbeitskopie des DataFrames wurde erstellt.")

    def fill_missing_values(self):
        """
        Füllt fehlende Werte in spezifischen Spalten.
        """
        print("\n--- Fülle fehlende Werte ---")

        if 'model' in self.df.columns:
            mode_model = self.df['model'].mode()[0]
            self.df['model'] = self.df['model'].fillna(mode_model)
            print(f"- Fehlende Werte in 'model' mit dem Modus '{mode_model}' gefüllt.")
        else:
            print("- Spalte 'model' nicht gefunden. Überspringe Füllung von fehlenden Werten.")

        if 'hp' in self.df.columns:
            median_hp = self.df['hp'].median()
            self.df['hp'] = self.df['hp'].fillna(median_hp)
            print(f"- Fehlende Werte in 'hp' mit dem Median '{median_hp:.2f}' gefüllt.")
        else:
            print("- Spalte 'hp' nicht gefunden. Überspringe Füllung von fehlenden Werten.")

        # Lösche Zeilen mit fehlenden Werten in anderen relevanten Spalten
        cols_to_check_for_nan = ['make', 'fuel', 'gear', 'offerType']
        existing_cols_to_check_for_nan = [col for col in cols_to_check_for_nan if col in self.df.columns]

        initial_rows = self.df.shape[0]
        if existing_cols_to_check_for_nan:
            self.df.dropna(subset=existing_cols_to_check_for_nan, inplace=True)
            if initial_rows - self.df.shape[0] > 0:
                print(
                    f"- {initial_rows - self.df.shape[0]} Zeilen mit fehlenden Werten in kritischen kategorialen Spalten ({existing_cols_to_check_for_nan}) entfernt.")
            else:
                print("- Keine Zeilen durch NaN-Prüfung in kritischen kategorialen Spalten entfernt.")
        else:
            print("- Keine der angegebenen kritischen Spalten zum Überprüfen auf NaN gefunden. Keine Zeilen entfernt.")

        return self  # Ermöglicht Chaining

    def clean_outliers_and_filters(self):
        """
        Entfernt Ausreißer und wendet grundlegende Filter an.
        """
        print("\n--- Bereinige Ausreißer und wende Filter an ---")
        initial_rows_before_filters = self.df.shape[0]

        if 'price' in self.df.columns:
            self.df = self.df[self.df['price'] > 0].copy()
            print("- Zeilen mit Preis <= 0 entfernt.")
        else:
            print("- Spalte 'price' nicht gefunden. Überspringe Preisfilter.")

        if 'mileage' in self.df.columns:
            self.df = self.df[self.df['mileage'] >= 0].copy()
            print("- Zeilen mit negativem Kilometerstand entfernt.")
        else:
            print("- Spalte 'mileage' nicht gefunden. Überspringe Kilometerstandfilter.")

        if 'hp' in self.df.columns:
            min_hp = 20
            max_hp = 1500
            self.df = self.df[(self.df['hp'] >= min_hp) & (self.df['hp'] <= max_hp)].copy()
            print(f"- Zeilen mit PS < {min_hp} oder PS > {max_hp} entfernt.")
        else:
            print("- Spalte 'hp' nicht gefunden. Überspringe PS-Filter.")

        if 'year' in self.df.columns:
            current_year = pd.Timestamp.now().year  # Dynamisches aktuelles Jahr
            min_year = 1990
            self.df = self.df[(self.df['year'] >= min_year) & (self.df['year'] <= current_year)].copy()
            print(f"- Zeilen mit Baujahr < {min_year} oder Baujahr > {current_year} entfernt.")
        else:
            print("- Spalte 'year' nicht gefunden. Überspringe Baujahr-Filter.")

        rows_removed_by_filters = initial_rows_before_filters - self.df.shape[0]
        if rows_removed_by_filters > 0:
            print(f"- Insgesamt {rows_removed_by_filters} Zeilen durch spezifische Spaltenfilter entfernt.")

        # Umgang mit Ausreißern im Preis (IQR-Methode)
        if 'price' in self.df.columns:
            Q1_price = self.df['price'].quantile(0.25)
            Q3_price = self.df['price'].quantile(0.75)
            IQR_price = Q3_price - Q1_price

            untergrenze_price = Q1_price - 1.5 * IQR_price
            obergrenze_price = Q3_price + 1.5 * IQR_price

            initial_rows_outlier_check = self.df.shape[0]
            self.df = self.df[(self.df['price'] >= untergrenze_price) & (self.df['price'] <= obergrenze_price)].copy()

            if initial_rows_outlier_check - self.df.shape[0] > 0:
                print(
                    f"- {initial_rows_outlier_check - self.df.shape[0]} Preis-Ausreißer mittels IQR-Methode entfernt.")
            else:
                print("- Keine Preis-Ausreißer mittels IQR-Methode entfernt.")
        else:
            print("- Spalte 'price' nicht gefunden. Überspringe Ausreißerbehandlung für Preis.")

        return self  # Ermöglicht Chaining

    def engineer_features(self):
        """
        Erstellt neue Features wie 'age' und 'price_per_mileage'.
        """
        print("\n--- Erzeuge neue Features ---")

        if 'year' in self.df.columns:
            aktuelles_jahr = pd.Timestamp.now().year  # Dynamisches aktuelles Jahr
            self.df['age'] = aktuelles_jahr - self.df['year']
            print("- 'age' Spalte hinzugefügt.")
        else:
            print("- Spalte 'year' nicht gefunden. Kann 'age' nicht berechnen.")

        if 'price' in self.df.columns and 'mileage' in self.df.columns:
            self.df['price_per_mileage'] = self.df['price'] / (self.df['mileage'] + 1)
            print("- 'price_per_mileage' Spalte hinzugefügt.")
        else:
            print("- 'price' oder 'mileage' Spalte(n) nicht gefunden. Kann 'price_per_mileage' nicht berechnen.")

        return self  # Ermöglicht Chaining

    def get_processed_data(self) -> pd.DataFrame:
        """
        Gibt den aktuell verarbeiteten DataFrame zurück.
        """
        print(f"\nFinaler DataFrame Form nach Bereinigung und Feature Engineering: {self.df.shape}")
        print("DataProcessor: Datenverarbeitung abgeschlossen.")
        return self.df


# Beispielhafte Verwendung der Klasse (optional, nur für isolierte Tests)
if __name__ == '__main__':
    # Erstelle einen Dummy-DataFrame mit den tatsächlichen Spaltennamen
    data = {
        'mileage': [100000, 50000, 1000, 200000, -100, 75000, 150000, 0, 120000, 90000],
        'make': ['VW', 'Audi', 'BMW', 'Mercedes', 'VW', 'Audi', 'BMW', 'Mercedes', 'VW', 'Audi'],
        'model': ['Golf', 'A3', None, 'C-Class', 'Polo', 'A4', 'X5', 'E-Class', 'Passat', 'Q5'],
        'fuel': ['Diesel', 'Gasoline', 'Gasoline', 'Diesel', 'Gasoline', None, 'Diesel', 'Gasoline', 'Diesel',
                 'Gasoline'],
        'gear': ['Manual', 'Automatic', 'Manual', 'Automatic', 'Manual', 'Automatic', 'Manual', 'Automatic', 'Manual',
                 'Automatic'],
        'offerType': ['Used', 'Used', 'Used', 'Pre-registered', 'Used', 'Used', 'Used', 'Used', 'Used', 'Used'],
        'price': [15000, 25000, 40000, 35000, 8000, 20000, 10000, 50000, 12000, 180000],  # 180000 ist ein Ausreißer
        'hp': [150, 180, None, 200, 90, 160, 250, 220, 130, 800],  # None und 800 (Ausreißer)
        'year': [2015, 2018, 2022, 2024, 2013, 2016, 2010, 2020, 2014, 1985],  # 1985 ist zu alt
    }
    dummy_df = pd.DataFrame(data)
    print("Dummy DataFrame vor Bereinigung (Test):")
    print(dummy_df.info())
    print("\n----------------------------------")

    # Verwendung der Klasse
    processor = DataProcessor(dummy_df)
    processed_df = processor.fill_missing_values() \
        .clean_outliers_and_filters() \
        .engineer_features() \
        .get_processed_data()

    print("\nDummy DataFrame nach Bereinigung (Test):")
    print(processed_df.info())
    print(processed_df.head())