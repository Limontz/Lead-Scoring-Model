from dataclasses import dataclass, field


@dataclass
class ScoringModelConfig:
    target: str = "target_closed_won"
    categorical_features: list[str] = field(
        default_factory=lambda: [
            "DEAL_DEALSOURCE",
            "DEAL_SOURCE_DETAIL",
            "UTM_SOURCE",
            "LEAD_TYPE",
            "DEAL_INDUSTRY",
            "CONTACT_ROLE",
            "COMPANY_STATE",
            "DEAL_HRIS_TECH_STACK",
            "DEAL_CCNL_MACRO",
        ]
    )
    numerical_features: list[str] = field(default_factory=list)
    training_set_portion: float = 0.7
    random_state: int = 42

    @property
    def total_features(self) -> list[str]:
        return [*self.categorical_features, *self.numerical_features]


def get_scoring_model_config() -> ScoringModelConfig:
    return ScoringModelConfig()
