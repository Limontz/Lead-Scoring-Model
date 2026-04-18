from lead_scoring.data.cleaning import clean_data
from lead_scoring.data.io import read_lead_data


def score_lead(path: str):
    lead_score_df = read_lead_data(path)
    clean_lead_score_df = clean_data
    return lead_score_df, clean_lead_score_df


if __name__ == "__main__":
    score_lead("data/lead_data.csv")
