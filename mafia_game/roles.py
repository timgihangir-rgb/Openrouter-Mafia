"""Role definitions and night-action data structures."""

from dataclasses import dataclass
from enum import Enum


class Role(Enum):
    MAFIA = "Mafia"
    DETECTIVE = "Detective"
    DOCTOR = "Doctor"
    CIVILIAN = "Civilian"

    @property
    def team(self) -> str:
        if self == Role.MAFIA:
            return "mafia"
        return "town"

    @property
    def has_night_action(self) -> bool:
        return self in (Role.MAFIA, Role.DETECTIVE, Role.DOCTOR)

    @property
    def display_name(self) -> str:
        """Russian display name for this role."""
        return {
            Role.MAFIA: "Мафия",
            Role.DETECTIVE: "Детектив",
            Role.DOCTOR: "Доктор",
            Role.CIVILIAN: "Мирный",
        }[self]

    @property
    def team_display_name(self) -> str:
        """Russian team name: Мафия or Город."""
        if self == Role.MAFIA:
            return "Мафия"
        return "Город"


@dataclass
class NightAction:
    """A resolved night action taken by a player."""
    role: Role
    target: int
    result: str


