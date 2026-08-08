"""CLI entry point for the Mafia AI Game."""

import argparse
import asyncio
import logging
import os
import sys

from mafia_game.config import Config
from mafia_game.game import MafiaGame


async def main():
    parser = argparse.ArgumentParser(description="Mafia AI Game — Игра Мафия на LLM")
    parser.add_argument(
        "--models", "-m", nargs="+",
        help="Model slugs from OpenRouter (at least N for N players)",
    )
    parser.add_argument(
        "--players", "-p", type=int, default=None,
        help="Number of players (default: 7, or NUM_PLAYERS env var)",
    )
    parser.add_argument(
        "--discussion-rounds", type=int, default=None,
        help="Number of discussion rounds per day (default: 2)",
    )
    parser.add_argument(
        "--log-file", "-l",
        help="Write full game transcript to this file",
    )
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    os.makedirs("logs", exist_ok=True)
    debug_log_path = os.path.join("logs", "debug.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(debug_log_path, mode="a", encoding="utf-8"),
        ],
    )
    logging.getLogger().setLevel(logging.INFO)

    kwargs = {}

    env_models = os.environ.get("MODELS")
    if args.models:
        models = args.models
    elif env_models:
        models = [m.strip() for m in env_models.split(",") if m.strip()]
    else:
        models = None

    env_players = os.environ.get("NUM_PLAYERS")
    if args.players:
        num_players = args.players
    elif env_players:
        num_players = int(env_players)
    else:
        num_players = 7

    env_rounds = os.environ.get("DISCUSSION_ROUNDS")
    if args.discussion_rounds is not None:
        discussion_rounds = args.discussion_rounds
    elif env_rounds:
        discussion_rounds = int(env_rounds)
    else:
        discussion_rounds = 2

    if models is not None:
        kwargs["models"] = models
    kwargs["num_players"] = num_players
    kwargs["discussion_rounds"] = discussion_rounds

    try:
        config = Config(**kwargs)
    except ValueError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        sys.exit(1)

    game = MafiaGame(config)
    await game.play_game(log_file=args.log_file)


if __name__ == "__main__":
    asyncio.run(main())
