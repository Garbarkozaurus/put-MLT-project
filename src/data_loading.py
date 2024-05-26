import pandas as pd

def load_penguins() -> pd.DataFrame:
    PENGUINS_PATH = "./data/penguins.csv"
    df = pd.read_csv(PENGUINS_PATH)
    df = df[["bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g","species"]]
    df = df.dropna()
    df.iloc[:,:-1] = df.iloc[:,:-1].apply(lambda x: x/x.max(), axis=0)
    return df

def load_wine() -> pd.DataFrame:
    WINE_PATH = "./data/winequality-red.csv"
    df = pd.read_csv(WINE_PATH, sep=';')
    df = df.dropna()
    df.iloc[:,:-1] = df.iloc[:,:-1].apply(lambda x: x/x.max(), axis=0)
    return df
