"""Team-name normalization helpers for joining ELO data with Transfermarkt data."""

TEAM_NAME_MAP = {
    "USA": "United States",
    "Turkey": "Turkiye",
    "Czech Republic": "Czechia",
    "Cote d'Ivoire": "Ivory Coast",
    "Democratic Republic of the Congo": "DR Congo",
    "Korea Republic": "South Korea",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Cape Verde Islands": "Cape Verde",
    "Curacao": "Curaçao",
    "Macedonia": "North Macedonia",
    "Swaziland": "Eswatini",
    "Sao Tome and Principe": "São Tomé and Príncipe",
}


def normalize_team_name(team_name: str) -> str:
    """Normalize a team name for Transfermarkt joins."""
    return TEAM_NAME_MAP.get(team_name, team_name)
