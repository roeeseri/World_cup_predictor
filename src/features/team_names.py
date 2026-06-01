"""Team-name normalization helpers."""

TEAM_NAME_MAP = {
    "USA": "United States",
        "Türkiye": "Turkey",
    "Turkiye": "Turkey",
    "Czech Republic": "Czechia",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Korea Republic": "South Korea",
    "IR Iran": "Iran",
    "Cabo Verde": "Cape Verde",
    "Congo DR": "DR Congo",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Curacao": "Curaçao",
    "Macedonia": "North Macedonia",
    "Swaziland": "Eswatini",
    "Sao Tome and Principe": "São Tomé and Príncipe",
}


def normalize_team_name(team_name: str) -> str:
    """Normalize a team name across fixtures, ELO, and Transfermarkt joins."""
    return TEAM_NAME_MAP.get(str(team_name), str(team_name))
