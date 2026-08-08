"""Main game engine — orchestrates the full Мафия game loop."""

import asyncio
import json
import logging
import random
import sys
from collections import Counter

from mafia_game.config import Config
from mafia_game.openrouter_client import OpenRouterClient, RateLimitError
from mafia_game.player import Player
from mafia_game.prompts import (
    DISCUSSION_PROMPT,
    INVALID_RESPONSE_PROMPT,
    NIGHT_DETECTIVE_PROMPT,
    NIGHT_DOCTOR_PROMPT,
    NIGHT_MAFIA_PROMPT,
    ROLE_ASSIGNMENT_PROMPT,
    SYSTEM_PROMPT,
    VOTING_PROMPT,
    VOTE_REVOTE_PROMPT,
)
from mafia_game.roles import Role
from mafia_game.utils import extract_json, validate_target, validate_vote

logger = logging.getLogger(__name__)


class MafiaGame:
    """Orchestrates a full Мафия game driven by LLMs via the OpenRouter API."""

    def __init__(self, config: Config):
        self.config = config
        self.client = OpenRouterClient(
            api_key=config.openrouter_api_key,
            referer=config.http_referer,
            app_title=config.app_title,
            timeout=config.timeout,
            max_retries=config.max_retries,
            reasoning_enabled=config.reasoning_enabled,
            max_tokens=config.max_tokens,
        )
        self.players: list[Player] = []
        self.day_count = 0
        self.night_count = 0
        self.discussion_history: list[str] = []
        self.game_log: list[str] = []
        self.eliminated: list[Player] = []
        self.player_messages: dict[int, list[dict]] = {}
        self.known_events: list[str] = []
        self.investigation_results: dict[int, list[dict]] = {}
        self.protected_player: int | None = None
        self.doctor_saved_history: list[int] = []
        self.killed_players: list[int] = []


    def alive_players(self) -> list[Player]:
        return [p for p in self.players if p.is_alive]

    def alive_player_ids(self) -> list[int]:
        return [p.player_id for p in self.players if p.is_alive]

    def dead_players(self) -> list[Player]:
        return [p for p in self.players if not p.is_alive]

    def announce(self, message: str):
        print(message)
        self.game_log.append(message)

    def _player_name(self, player_id: int) -> str:
        """Get the display name (model slug) for a player ID."""
        for p in self.players:
            if p.player_id == player_id:
                return p.model_slug
        return f"player_{player_id}"

    def _format_alive_players(self, exclude: int | None = None) -> str:
        return ", ".join(
            p.model_slug
            for p in self.alive_players()
            if p.player_id != exclude
        )

    def _format_dead_players(self) -> str:
        if not self.dead_players():
            return "(нет)"
        parts = []
        for p in self.dead_players():
            role_str = p.role.display_name if p.role else "неизвестно"
            parts.append(f"{p.model_slug} ({role_str})")
        return ", ".join(parts)

    def build_game_state(self, player: Player) -> dict:
        """Build a serializable dict of public + private game state for a player."""
        state = {
            "day": self.day_count,
            "night": self.night_count,
            "alive_players": [
                {"id": p.player_id, "model": p.model_slug, "role_revealed": p.role_revealed}
                for p in self.alive_players()
            ],
            "dead_players": [
                {
                    "id": p.player_id,
                    "model": p.model_slug,
                    "role": p.role.display_name if p.role else "неизвестно",
                }
                for p in self.dead_players()
            ],
            "known_events": list(self.known_events),
            "my_player_id": player.player_id,
            "my_role": player.role.display_name if player.role else None,
        }
        if player.role == Role.DETECTIVE:
            state["investigation_results"] = self.investigation_results.get(
                player.player_id, []
            )
        if player.role == Role.DOCTOR:
            state["doctor_saved_history"] = list(self.doctor_saved_history)
        return state

    def _format_game_state_for_prompt(self, state: dict) -> str:
        lines = [f"День: {state['day']}, Ночь: {state['night']}"]
        lines.append("Живые игроки: " + json.dumps(state["alive_players"], indent=2))
        if state["dead_players"]:
            lines.append("Мёртвые игроки: " + json.dumps(state["dead_players"], indent=2))
        else:
            lines.append("Мёртвые игроки: (нет)")
        if state["known_events"]:
            lines.append("Известные события: " + "\n".join(f"  - {e}" for e in state["known_events"]))
        else:
            lines.append("Известные события: (нет)")
        return "\n".join(lines)


    async def get_model_response(
        self,
        player: Player,
        prompt: str,
        validator_fn=None,
        validator_error_hint: str = "",
    ) -> dict | None:
        """Send a prompt to a single player's model, retry on bad JSON/validation.

        Appends the user prompt and assistant response to the player's private
        conversation history so context is preserved.

        Returns the parsed JSON dict, or ``None`` after all retries are
        exhausted (caller must handle fallback).
        """
        messages = self.player_messages[player.player_id]
        messages.append({"role": "user", "content": prompt})

        alive_ids = self.alive_player_ids()
        attempt = 0
        while attempt < self.config.max_retries:
            attempt += 1
            try:
                message = await self.client.chat(player.model_slug, messages)
                content = message.get("content", "")
                parsed = extract_json(content)
                if validator_fn:
                    validator_fn(parsed, alive_ids)
                assistant_msg = {"role": "assistant"}
                if content:
                    assistant_msg["content"] = content
                if message.get("reasoning_details"):
                    assistant_msg["reasoning_details"] = message["reasoning_details"]
                messages.append(assistant_msg)
                return parsed
            except RateLimitError:
                logger.warning(
                    "Rate limit hit for player_%d (%s) — using fallback",
                    player.player_id, player.model_slug,
                )
                return None
            except Exception as exc:
                logger.warning(
                    "Attempt %d/%d for player_%d (%s): %s",
                    attempt, self.config.max_retries,
                    player.player_id, player.model_slug, exc,
                )
                messages.append(
                    {"role": "assistant", "content": f"[FAILED: {exc}]"}
                )
                retry_hint = validator_error_hint or (
                    f"ERROR: {exc}. Please respond with valid JSON in the required format."
                )
                messages.append(
                    {"role": "user", "content": retry_hint}
                )

        logger.error(
            "All %d retries exhausted for player_%d (%s)",
            self.config.max_retries, player.player_id, player.model_slug,
        )
        return None


    async def setup(self):
        """Initialize players, assign roles, and send private role info."""
        self.announce("=== ИГРА МАФИЯ ===")
        self.announce(f"Всего игроков: {self.config.num_players}")
        for i, model in enumerate(self.config.models[: self.config.num_players]):
            player = Player(player_id=i, model_slug=model)
            self.players.append(player)
            self.player_messages[i] = [
                {"role": "system", "content": SYSTEM_PROMPT.format(
                    total_players=self.config.num_players
                )}
            ]

        role_list: list[Role] = []
        for role_name, count in self.config.role_distribution.items():
            role = Role(role_name)
            role_list.extend([role] * count)

        if len(role_list) > self.config.num_players:
            role_list = role_list[: self.config.num_players]
        while len(role_list) < self.config.num_players:
            role_list.append(Role.CIVILIAN)

        random.shuffle(role_list)
        for player, role in zip(self.players, role_list):
            player.role = role

        mafia_count = sum(1 for p in self.players if p.role == Role.MAFIA)
        town_count = self.config.num_players - mafia_count
        self.announce(
            f"Роли назначены: {mafia_count} Мафии, {town_count} Мирных"
            f" (Детектив={sum(1 for p in self.players if p.role == Role.DETECTIVE)},"
            f" Доктор={sum(1 for p in self.players if p.role == Role.DOCTOR)},"
            f" Мирный={sum(1 for p in self.players if p.role == Role.CIVILIAN)})"
        )
        self.announce("Отправка приватных ролей...")

        async def assign_role(player: Player):
            state = self.build_game_state(player)
            prompt = ROLE_ASSIGNMENT_PROMPT.format(
                player_id=player.player_id,
                role=player.role.display_name,
                total_players=self.config.num_players,
                total_players_minus_1=self.config.num_players - 1,
                role_distribution=json.dumps(
                    self.config.role_distribution, indent=2
                ),
            )
            parsed = await get_model_response_simple(
                self, player, prompt,
                expect_field="action", expect_value="understood",
            )
            if parsed is None:
                self.announce(
                    f"  {player.model_slug}:"
                    " роль подтверждена (резервный вариант)"
                )
            else:
                self.announce(
                    f"  {player.model_slug}:"
                    f" {parsed.get('thoughts', '')[:60]}"
                )

        await asyncio.gather(*[assign_role(p) for p in self.players])
        self.announce("Все игроки готовы. Игра начинается...")


    async def night_phase(self):
        """Execute night actions: Мафия kill, Detective investigate, Doctor save."""
        self.night_count += 1
        self.announce(f"\n=== НОЧЬ {self.night_count} ===")

        night_actors = [
            p for p in self.alive_players()
            if p.role and p.role.has_night_action
        ]

        mafia_actions: list[tuple[int, int]] = []
        detective_target: int | None = None
        doctor_target: int | None = None

        async def do_mafia_action(mafia: Player):
            state = self.build_game_state(mafia)
            already_targets = [
                (pid, t) for pid, t in mafia_actions if pid != mafia.player_id
            ]
            other_targets_str = (
                ", ".join(f"{self._player_name(pid)} целится в {self._player_name(t)}" for pid, t in already_targets)
                if already_targets
                else "(none yet)"
            )
            prompt = NIGHT_MAFIA_PROMPT.format(
                night_num=self.night_count,
                alive_players=self._format_alive_players(exclude=mafia.player_id),
                dead_players=self._format_dead_players(),
                killed_players=", ".join(self._player_name(p) for p in self.killed_players) or "(нет)",
                mafia_other_targets=other_targets_str,
            )

            def validate_kill(parsed, alive):
                t = parsed.get("target")
                if t is None:
                    raise ValueError("Missing 'target' field")
                target = validate_target(t, alive)
                if target == mafia.player_id:
                    raise ValueError("You cannot target yourself as Мафия")

            parsed = await self.get_model_response(
                mafia, prompt,
                validator_fn=validate_kill,
                validator_error_hint="Your target must be a valid alive player ID. "
                "Respond with JSON: {\"thoughts\": \"...\", \"target\": <id>}",
            )
            if parsed is None:
                choices = [p.player_id for p in self.alive_players() if p.player_id != mafia.player_id]
                target = random.choice(choices) if choices else mafia.player_id
                parsed = {"thoughts": "[ОШИБКА — случайная цель]", "target": target}
            mafia_actions.append((mafia.player_id, validate_target(parsed["target"], self.alive_player_ids())))
            self.announce(
                f"  [Мысль Мафии] {self._player_name(mafia.player_id)}: {parsed.get('thoughts', '')[:80]}"
            )

        async def do_detective_action(detective: Player):
            state = self.build_game_state(detective)
            prompt = NIGHT_DETECTIVE_PROMPT.format(
                night_num=self.night_count,
                alive_players=self._format_alive_players(exclude=detective.player_id),
                dead_players=self._format_dead_players(),
            )

            def validate_investigate(parsed, alive):
                t = parsed.get("target")
                if t is None:
                    raise ValueError("Missing 'target' field")
                validate_target(t, alive)

            parsed = await self.get_model_response(
                detective, prompt,
                validator_fn=validate_investigate,
                validator_error_hint="Your target must be a valid alive player ID. "
                "Respond with JSON: {\"thoughts\": \"...\", \"target\": <id>}",
            )
            if parsed is None:
                choices = [p.player_id for p in self.alive_players() if p.player_id != detective.player_id]
                target = random.choice(choices) if choices else detective.player_id
                parsed = {"thoughts": "[ОШИБКА — случайная цель]", "target": target}
            nonlocal detective_target
            detective_target = validate_target(parsed["target"], self.alive_player_ids())
            self.announce(
                f"  [Мысль Детектива] {self._player_name(detective.player_id)}: {parsed.get('thoughts', '')[:80]}"
            )

        async def do_doctor_action(doctor: Player):
            state = self.build_game_state(doctor)
            prompt = NIGHT_DOCTOR_PROMPT.format(
                night_num=self.night_count,
                alive_players=self._format_alive_players(exclude=doctor.player_id),
                dead_players=self._format_dead_players(),
                doctor_saved_history=", ".join(
                    self._player_name(p) for p in self.doctor_saved_history[-3:]
                ) or "(нет)",
            )

            def validate_save(parsed, alive):
                t = parsed.get("target")
                if t is None:
                    raise ValueError("Missing 'target' field")
                target = validate_target(t, alive)
                if target in self.doctor_saved_history[-1:]:
                    raise ValueError("You cannot save the same player two nights in a row")

            parsed = await self.get_model_response(
                doctor, prompt,
                validator_fn=validate_save,
                validator_error_hint="Your save target must be a valid alive player ID "
                "and not the same as last night. Respond with JSON: "
                "{\"thoughts\": \"...\", \"target\": <id>}",
            )
            if parsed is None:
                choices = [p.player_id for p in self.alive_players() if p.player_id != doctor.player_id]
                target = random.choice(choices) if choices else doctor.player_id
                parsed = {"thoughts": "[ОШИБКА — случайная цель]", "target": target}
            nonlocal doctor_target
            doctor_target = validate_target(parsed["target"], self.alive_player_ids())
            self.announce(
                f"  [Мыслит Доктора] {self._player_name(doctor.player_id)}: {parsed.get('thoughts', '')[:80]}"
            )

        tasks = []
        for actor in night_actors:
            if actor.role == Role.MAFIA:
                tasks.append(do_mafia_action(actor))
            elif actor.role == Role.DETECTIVE:
                tasks.append(do_detective_action(actor))
            elif actor.role == Role.DOCTOR:
                tasks.append(do_doctor_action(actor))

        if tasks:
            await asyncio.gather(*tasks)

        if mafia_actions:
            targets = [t for _, t in mafia_actions]
            target_counts = Counter(targets)
            max_count = max(target_counts.values())
            top_targets = [t for t, c in target_counts.items() if c == max_count]
            mafia_target = top_targets[0]
            self.announce(f"  Мафия голосует: {targets} → убить {self._player_name(mafia_target)}")
        else:
            mafia_target = None
            self.announce("  Мафии нет в живых — убийство не выполнено.")

        if doctor_target is not None:
            self.doctor_saved_history.append(doctor_target)
            self.protected_player = doctor_target
            self.announce(f"  Доктор спас: {self._player_name(doctor_target)}")
        else:
            self.protected_player = None

        killed_player = None
        if mafia_target is not None:
            if mafia_target == self.protected_player:
                self.announce(
                    f"  Доктор спас player_{mafia_target} — убийство Мафии предотвращено!"
                )
            else:
                killed_player = mafia_target
                self.killed_players.append(killed_player)
                victim = next(p for p in self.players if p.player_id == killed_player)
                victim.is_alive = False
                self.announce(f"  {victim.model_slug} убит ночью.")
                self.known_events.append(
                    f"Ночь {self.night_count}: {self._player_name(killed_player)} был убит"
                )

        if detective_target is not None:
            target_player = next(p for p in self.players if p.player_id == detective_target)
            is_mafia = target_player.is_mafia
            result = "МАФИЯ" if is_mafia else "ГОРОД"
            self.announce(
                f"  Детектив расследует {self._player_name(detective_target)}: {result}"
            )
            for det in self.alive_players():
                if det.role == Role.DETECTIVE and det.player_id != detective_target:
                    pass
            det_players = [p for p in self.players if p.role == Role.DETECTIVE and p.is_alive]
            for det in det_players:
                self.investigation_results.setdefault(det.player_id, []).append({
                    "night": self.night_count,
                    "target": detective_target,
                    "result": result,
                })

        self.protected_player = None


    async def day_phase(self):
        """Execute day: announcement, discussion rounds, voting, elimination."""
        self.day_count += 1
        self.announce(f"\n=== ДЕНЬ {self.day_count} ===")

        alive = self.alive_players()
        if not alive:
            return

        for round_num in range(1, self.config.discussion_rounds + 1):
            self.announce(f"\n  — Раунд обсуждения {round_num} —")
            for player in alive:
                state = self.build_game_state(player)
                state_str = self._format_game_state_for_prompt(state)
                disc_hist = "\n".join(self.discussion_history) if self.discussion_history else "(пока нет заявлений)"
                prompt = DISCUSSION_PROMPT.format(
                    day_num=self.day_count,
                    alive_players=self._format_alive_players(),
                    dead_players=self._format_dead_players(),
                    known_events="\n".join(f"  - {e}" for e in self.known_events) or "(нет)",
                    discussion_history=disc_hist,
                    round_num=round_num,
                )
                parsed = await self.get_model_response(
                    player, prompt,
                    validator_fn=None,
                    validator_error_hint="Respond with JSON: "
                    "{\"thoughts\": \"...\", \"statement\": \"...\"}",
                )
                if parsed is None:
                    parsed = {
                        "thoughts": "[ОШИБКА]",
                        "statement": f"У меня нечего добавить ({player.model_slug}).",
                    }
                thoughts = parsed.get("thoughts", "")
                statement = parsed.get("statement", "")
                self.announce(
                    f"    {player.model_slug}: "
                    f'"{statement}"'
                )
                self.announce(f"    [Мысль] {thoughts}")
                self.discussion_history.append(
                    f"{player.model_slug}: {statement}"
                )

        self.announce("\n  — Голосование —")
        votes: list[tuple[int, int]] = []

        for player in alive:
            state = self.build_game_state(player)
            disc_summary = "\n".join(self.discussion_history) or "(нет обсуждения)"
            prompt = VOTING_PROMPT.format(
                day_num=self.day_count,
                alive_players=self._format_alive_players(),
                discussion_summary=disc_summary,
            )

            def validate_vote_fn(parsed, alive_ids):
                v = parsed.get("vote")
                if v is None:
                    raise ValueError("Missing 'vote' field")
                validate_vote(v, alive_ids)

            parsed = await self.get_model_response(
                player, prompt,
                validator_fn=validate_vote_fn,
                validator_error_hint="Respond with JSON: "
                "{\"thoughts\": \"...\", \"vote\": <player_id>}",
            )
            if parsed is None:
                choices = self.alive_player_ids()
                vote = random.choice(choices) if choices else player.player_id
                parsed = {"thoughts": "[ОШИБКА — случайный голос]", "vote": vote}
            vote = validate_vote(parsed["vote"], self.alive_player_ids())
            self.announce(f"    [Мысль] {self._player_name(player.player_id)}: {parsed.get('thoughts', '')[:80]}")
            votes.append((player.player_id, vote))
            self.announce(f"    {self._player_name(player.player_id)} голосует за исключение {self._player_name(vote)}")

        eliminated = await self._resolve_votes(votes, alive)
        if eliminated is not None:
            victim = next(p for p in self.players if p.player_id == eliminated)
            victim.is_alive = False
            role = victim.reveal_role()
            self.eliminated.append(victim)
            self.announce(f"\n  >> {victim.model_slug} ИСКЛЮЧЁН. Роль: {role.display_name}")
            self.known_events.append(
                f"День {self.day_count}: {victim.model_slug} "
                f"исключён, роль была {role.display_name}"
            )

    async def _resolve_votes(self, votes: list[tuple[int, int]], alive: list[Player]) -> int | None:
        """Tally votes and handle ties. Returns the eliminated player's ID."""
        target_counts = Counter(v for _, v in votes)
        if not target_counts:
            return None
        max_count = max(target_counts.values())
        top_targets = [t for t, c in target_counts.items() if c == max_count]

        if len(top_targets) > 1:
            return await self.handle_tie(top_targets, votes, alive)
        return top_targets[0]

    async def handle_tie(
        self, tied_players: list[int], votes: list[tuple[int, int]], alive: list[Player]
    ):
        """Re-vote among tied players only."""
        self.announce(f"  НИЧЬЯ между игроками: {[self._player_name(p) for p in tied_players]}. Переголосование.")
        tied_str = ", ".join(self._player_name(p) for p in tied_players)
        revote_votes: list[tuple[int, int]] = []

        for player in alive:
            prompt = VOTE_REVOTE_PROMPT.format(
                tied_players=tied_str,
                tied_players_list=tied_str,
            )

            def validate_tied_vote(parsed, alive_ids):
                v = parsed.get("vote")
                if v is None:
                    raise ValueError("Missing 'vote' field")
                target = validate_vote(v, alive_ids)
                if target not in tied_players:
                    raise ValueError(f"You must vote for one of the tied players: {tied_players}")

            parsed = await self.get_model_response(
                player, prompt,
                validator_fn=validate_tied_vote,
                validator_error_hint=f"Respond with JSON: "
                f"{{\"thoughts\": \"...\", \"vote\": <one of {tied_players}>}}",
            )
            if parsed is None:
                vote = random.choice(tied_players)
                parsed = {"thoughts": "[ОШИБКА — случайный переголос]", "vote": vote}
            vote = validate_vote(parsed["vote"], self.alive_player_ids())
            self.announce(f"    {self._player_name(player.player_id)} голосует: {self._player_name(vote)}")
            revote_votes.append((player.player_id, vote))

        target_counts = Counter(v for _, v in revote_votes)
        max_count = max(target_counts.values())
        top = [t for t, c in target_counts.items() if c == max_count]
        if len(top) > 1:
            winner = random.choice(top)
            self.announce(f"  Переголосование всё ещё в ничью ({top}). Устраняем случайно.")
        else:
            winner = top[0]
        return winner


    def check_win_condition(self) -> tuple[bool, str | None]:
        """Check if Мафия or Город has won. Returns (game_over, winner)."""
        alive_mafia = [p for p in self.players if p.is_alive and p.is_mafia]
        alive_town = [
            p for p in self.players
            if p.is_alive and not p.is_mafia
        ]

        if len(alive_mafia) == 0:
            return True, "Город"
        if len(alive_mafia) >= len(alive_town):
            return True, "Мафия"
        return False, None


    async def play_game(self, log_file: str | None = None):
        """Main game loop: setup → while not won → night → day → check wins."""
        async with self.client:
            await self.setup()

            while True:
                await self.night_phase()
                game_over, winner = self.check_win_condition()
                if game_over:
                    break
                if len(self.alive_players()) <= 1:
                    break

                await self.day_phase()
                game_over, winner = self.check_win_condition()
                if game_over:
                    break
                if len(self.alive_players()) <= 1:
                    break

            if game_over:
                self.announce(f"\n=== ИГРА ОКОНЧЕНА: {winner.upper()} ПОБЕДИЛА! ===")
            else:
                self.announce("\n=== ИГРА ОКОНЧЕНА: Победитель не определён ===")

            self.announce("\n--- Итоги ---")
            for p in self.players:
                status = "В ЖИВЫХ" if p.is_alive else "УБИТ"
                role = p.role.display_name if p.role else "неизвестно"
                self.announce(f"  {p.model_slug}: {role} — {status}")

        if log_file:
            with open(log_file, "w", encoding="utf-8") as f:
                f.write("\n".join(self.game_log))
            self.announce(f"Игровой лог сохранён в {log_file}")



async def get_model_response_simple(
    game: MafiaGame,
    player: Player,
    prompt: str,
    expect_field: str | None = None,
    expect_value: str | None = None,
) -> dict | None:
    """Simplified wrapper around get_model_response for setup-phase calls.

    Accepts the full responsibility of message management and only validates
    the presence (and optionally value) of a single field.
    """
    messages = game.player_messages[player.player_id]
    messages.append({"role": "user", "content": prompt})

    for attempt in range(game.config.max_retries):
        try:
            message = await game.client.chat(player.model_slug, messages)
            content = message.get("content", "")
            parsed = extract_json(content)
            if expect_field:
                val = parsed.get(expect_field)
                if val is None:
                    raise ValueError(f"Missing '{expect_field}' field")
                if expect_value and str(val) != expect_value:
                    raise ValueError(
                        f"Expected '{expect_field}'='{expect_value}', got '{val}'"
                    )
            assistant_msg = {"role": "assistant"}
            if content:
                assistant_msg["content"] = content
            if message.get("reasoning_details"):
                assistant_msg["reasoning_details"] = message["reasoning_details"]
            messages.append(assistant_msg)
            return parsed
        except RateLimitError:
            logger.warning(
                "Rate limit hit for player_%d (%s) — using fallback",
                player.player_id, player.model_slug,
            )
            return None
        except Exception as exc:
            logger.warning(
                "Setup attempt %d/%d for player_%d: %s",
                attempt + 1, game.config.max_retries,
                player.player_id, exc,
            )
            messages.append({"role": "assistant", "content": f"[FAILED: {exc}]"})
            messages.append({
                "role": "user",
                "content": f"ERROR: {exc}. Please respond with valid JSON "
                f"containing '{expect_field}' field. Try again.",
            })

    logger.error(
        "All retries exhausted for player_%d in setup", player.player_id
    )
    return None
