import pandas as pd


def load_data():
    data = pd.read_csv("datasets/TCGA_BRCA_tpm.tsv", sep="\t")
    clinic = pd.read_csv("datasets/brca_tcga_pan_can_atlas_2018_clinical_data_filtered.tsv", sep="\t")
    return data, clinic


def preprocess(data, clinic):
    """Transpose gene data, clean clinic data, merge, and return final df."""

    # Transpose genetic sequencing data
    transposed = (
        data.set_index("Ensembl_ID")
        .T.reset_index()
        .rename_axis(None, axis=1)
        .rename(columns={"index": "Ensembl_ID"})
    )

    # Clean up clinic data
    clinic = clinic.rename(columns={"Sample ID": "Ensembl_ID"})
    clinic.Ensembl_ID += "A"

    # Merge clinic data with gene expression data
    final_data = pd.merge(transposed, clinic, how="inner", on="Ensembl_ID")

    # Keep only Subtype and gene expression columns
    df = (
        final_data.drop(
            columns=["Ensembl_ID", "Patient ID", "Diagnosis Age", "Cancer Type", "Sex", "Tumor Type"]
        )
        .set_index("Subtype")
        .reset_index()
        .dropna()
    )

    return df


def get_Xy(df):
    """Split dataframe into features and target."""
    X = df.drop("Subtype", axis=1)
    y = df.Subtype
    return X, y


if __name__ == "__main__":
    data, clinic = load_data()
    df = preprocess(data, clinic)
    print("Final dataset shape:", df.shape)
    print(df.groupby("Subtype").size().reset_index().rename(columns={0: "Count"}))
