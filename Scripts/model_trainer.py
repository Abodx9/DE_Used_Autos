# Scripts/model_trainer.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class ModelTrainer:
    """
    Eine Klasse zum Trainieren, Bewerten und Analysieren eines Random Forest Regressors.
    """

    def __init__(self, df: pd.DataFrame, features: list, target: str):
        """
        Initialisiert den ModelTrainer mit einem DataFrame, Features und einer Zielvariable.

        Args:
            df (pandas.DataFrame): Der bereinigte DataFrame.
            features (list): Eine Liste der Spaltennamen, die als Features verwendet werden sollen.
            target (str): Der Spaltenname der Zielvariable (z.B. 'price').
        """
        self.df = df.copy()  # Arbeitskopie des DataFrames
        self.features = [col for col in features if col in self.df.columns]
        self.target = target

        missing_features = [col for col in features if col not in self.df.columns]
        if missing_features:
            print(
                f"Warnung: Folgende Features sind nicht im DataFrame vorhanden und werden ignoriert: {missing_features}")

        if self.target not in self.df.columns:
            raise ValueError(f"Fehler: Die Zielvariable '{self.target}' ist nicht im DataFrame vorhanden.")

        self.X = self.df[self.features]
        self.y = self.df[self.target]

        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.full_pipeline = None
        self.y_pred = None
        self.metrics = {}
        self.plots = {}
        self.feature_importances_df = None

        print(f"ModelTrainer initialisiert. Features: {self.features}, Ziel: {self.target}")
        print(f"Datenpunkte: {self.X.shape[0]}, Features: {self.X.shape[1]}")

    def split_data(self, test_size: float = 0.2, random_state: int = 42):
        """
        Teilt die Daten in Trainings- und Testsets auf.
        """
        print(f"\n--- Teile Daten in Trainings- und Testsets auf (Testgröße: {test_size * 100}%) ---")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        print(f"Trainingsdaten: {self.X_train.shape[0]} Beispiele, Testdaten: {self.X_test.shape[0]} Beispiele.")
        return self  # Ermöglicht Chaining

    def build_and_train_model(self, n_estimators: int = 100, random_state: int = 42):
        """
        Erstellt die Vorverarbeitungs-Pipeline und trainiert den Random Forest Regressor.
        """
        print("\n--- Erstelle und trainiere das Modell ---")

        numerische_spalten = self.X_train.select_dtypes(include=np.number).columns.tolist()
        kategoriale_spalten = self.X_train.select_dtypes(include='object').columns.tolist()

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerische_spalten),
                ('cat', OneHotEncoder(handle_unknown='ignore'), kategoriale_spalten)
            ],
            remainder='passthrough'
        )

        print(f"Numerische Spalten für Skalierung: {numerische_spalten}")
        print(f"Kategoriale Spalten für One-Hot-Encoding: {kategoriale_spalten}")

        rf_model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)

        self.full_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                             ('regressor', rf_model)])

        print("Random Forest Regressor Modell in Pipeline definiert.")
        print("Starte das Training der Pipeline auf den Trainingsdaten...")
        self.full_pipeline.fit(self.X_train, self.y_train)
        print("Modelltraining abgeschlossen.")
        return self  # Ermöglicht Chaining

    def evaluate_model(self):
        """
        Bewertet das trainierte Modell auf dem Testset und speichert die Metriken.
        """
        print("\n--- Bewerte das Modell auf dem Testset ---")
        if self.full_pipeline is None:
            raise RuntimeError("Modell wurde noch nicht trainiert. Bitte rufen Sie 'build_and_train_model' zuerst auf.")

        self.y_pred = self.full_pipeline.predict(self.X_test)

        mae = mean_absolute_error(self.y_test, self.y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, self.y_pred))
        r2 = r2_score(self.y_test, self.y_pred)

        self.metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2
        }
        print(f"  Mean Absolute Error (MAE): {mae:,.2f} €")
        print(f"  Root Mean Squared Error (RMSE): {rmse:,.2f} €")
        print(f"  R-squared (R²): {r2:.4f}")
        return self  # Ermöglicht Chaining

    def generate_plots(self):
        """
        Erstellt Visualisierungen der Modellvorhersagen und Feature Importances.
        Die Figuren-Objekte werden im Attribut 'plots' gespeichert.
        """
        print("\n--- Erzeuge Modell-Visualisierungen ---")
        if self.y_pred is None:
            raise RuntimeError("Modell wurde noch nicht bewertet. Bitte rufen Sie 'evaluate_model' zuerst auf.")

        # Scatter Plot: Tatsächliche vs. vorhergesagte Preise
        fig_scatter, ax_scatter = plt.subplots(figsize=(10, 8))
        ax_scatter.scatter(self.y_test, self.y_pred, alpha=0.5, color='blue', s=15)
        ax_scatter.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        ax_scatter.set_xlabel('Tatsächlicher Preis (€)', fontsize=14)
        ax_scatter.set_ylabel('Vorhergesagter Preis (€)', fontsize=14)
        ax_scatter.set_title('Tatsächlicher Preis vs. Vorhergesagter Preis', fontsize=16)
        ax_scatter.grid(True, linestyle='--', alpha=0.7)
        self.plots['scatter_plot'] = fig_scatter
        plt.close(fig_scatter)  # Wichtig, um Matplotlib-Speicher zu verwalten

        # Residuen Histogramm
        residuals = self.y_test - self.y_pred
        fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
        sns.histplot(residuals, bins=50, kde=True, color='purple', ax=ax_hist)
        ax_hist.set_title('Verteilung der Residuen (Vorhersagefehler)', fontsize=16)
        ax_hist.set_xlabel('Residuen (Tatsächlicher Preis - Vorhergesagter Preis) (€)', fontsize=14)
        ax_hist.set_ylabel('Anzahl', fontsize=14)
        self.plots['residuals_hist'] = fig_hist
        plt.close(fig_hist)

        # Residualplot
        fig_residuals, ax_residuals = plt.subplots(figsize=(10, 6))
        ax_residuals.scatter(self.y_pred, residuals, alpha=0.5, color='darkorange', s=15)
        ax_residuals.axhline(y=0, color='r', linestyle='--', lw=2)
        ax_residuals.set_xlabel('Vorhergesagter Preis (€)', fontsize=14)
        ax_residuals.set_ylabel('Residuen (€)', fontsize=14)
        ax_residuals.set_title('Residualplot: Vorhergesagter Preis vs. Residuen', fontsize=16)
        ax_residuals.grid(True, linestyle='--', alpha=0.7)
        self.plots['residuals_plot'] = fig_residuals
        plt.close(fig_residuals)

        # Feature Importance
        if self.full_pipeline and hasattr(self.full_pipeline.named_steps['regressor'], 'feature_importances_'):
            preprocessor_trained = self.full_pipeline.named_steps['preprocessor']
            all_transformed_feature_names = preprocessor_trained.get_feature_names_out(self.X.columns)
            importances = self.full_pipeline.named_steps['regressor'].feature_importances_

            self.feature_importances_df = pd.DataFrame(
                {'Feature': all_transformed_feature_names, 'Importance': importances})
            self.feature_importances_df = self.feature_importances_df.sort_values(by='Importance', ascending=False)

            fig_feature_imp, ax_feature_imp = plt.subplots(figsize=(12, 8))
            sns.barplot(x='Importance', y='Feature', data=self.feature_importances_df.head(15), palette='Reds_d',
                        ax=ax_feature_imp)
            ax_feature_imp.set_title('Top 15 Feature Importances (Random Forest)', fontsize=18)
            ax_feature_imp.set_xlabel('Wichtigkeit (Mean Decrease Impurity)', fontsize=14)
            ax_feature_imp.set_ylabel('Feature', fontsize=14)
            fig_feature_imp.tight_layout()
            self.plots['feature_importance_plot'] = fig_feature_imp
            plt.close(fig_feature_imp)
        else:
            print(
                "Feature Importance kann nicht generiert werden (Modell nicht trainiert oder hat keine feature_importances_).")

        print("Modell-Visualisierungen erfolgreich erzeugt.")
        return self  # Ermöglicht Chaining

    def get_results(self):
        """
        Gibt die gesammelten Ergebnisse des Modelltrainings und der Bewertung zurück.
        """
        return self.metrics, self.plots, self.feature_importances_df, self.full_pipeline, (self.y_test, self.y_pred)


# Beispielhafte Verwendung der Klasse (optional, nur für isolierte Tests)
if __name__ == '__main__':
    # Erstelle einen Dummy-DataFrame wie für den DataProcessor
    data = {
        'mileage': np.random.randint(10000, 200000, 100),
        'make': np.random.choice(['BMW', 'Audi', 'VW'], 100),
        'model': np.random.choice(['A3', 'A4', '3 Series', '5 Series', 'Golf', 'Passat'], 100),
        'fuel': np.random.choice(['Gasoline', 'Diesel'], 100),
        'gear': np.random.choice(['Manual', 'Automatic'], 100),
        'hp': np.random.randint(80, 300, 100),
        'year': np.random.randint(2010, 2023, 100),
        'offerType': np.random.choice(['Used', 'Pre-registered'], 100),
        'price': np.random.randint(5000, 50000, 100)
    }
    dummy_df = pd.DataFrame(data)
    dummy_df['age'] = 2025 - dummy_df['year']  # Feature Engineering wie im Preprocessing
    dummy_df['price_per_mileage'] = dummy_df['price'] / (dummy_df['mileage'] + 1)

    test_features = ['mileage', 'make', 'model', 'fuel', 'gear', 'hp', 'age', 'offerType', 'year']
    test_target = 'price'

    # Verwendung der Klasse
    trainer = ModelTrainer(dummy_df, test_features, test_target)
    metrics, plots, feature_importances, pipeline, (y_test, y_pred) = \
        trainer.split_data() \
            .build_and_train_model() \
            .evaluate_model() \
            .generate_plots() \
            .get_results()

    if metrics:
        print("\nMetriken (aus Test):", metrics)
        for name, fig in plots.items():
            print(f"Plot '{name}' erzeugt.")
            # fig.show() # Würde Plots anzeigen, wenn direkt ausgeführt
        print("\nFeature Importances (aus Test):\n", feature_importances.head())