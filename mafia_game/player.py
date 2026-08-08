"""Player dataclass — represents an AI model assigned a Mafia role."""

from dataclasses import dataclass

from mafia_game.roles import Role


@dataclass
class Player:
    player_id: int
    model_slug: str
    is_alive: bool = True
    role: Role | None = None
    role_revealed: bool = False

    @property
    def team(self) -> str:
        if self.role is None:
            return "unknown"
        return self.role.team

    def reveal_role(self) -> Role:
        """Mark the player's role as publicly known (on elimination)."""
        self.role_revealed = True
        return self.role

    @property
    def is_mafia(self) -> bool:
        return self.role == Role.MAFIA

    def __repr__(self) -> str:
        status = "ALIVE" if self.is_alive else "DEAD"
        role = self.role.value if self.role else "unassigned"
        return f"Player({self.player_id}, {self.model_slug}, {role}, {status})"
