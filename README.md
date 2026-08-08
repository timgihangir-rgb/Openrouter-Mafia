# OpenrouterMafia — Игра Мафия на LLM

Игра Мафия, где llm-модели из openrouter играют за ролей

## Быстрый старт

```bash
export OPENROUTER_API_KEY=sk-or-v1-ВАШ_КЛЮЧ
python3 run_game.py --log-file logs/game_log.txt
```

API-ключ читается из `OPENROUTER_API_KEY`

## Флаги
```
 `--models M1 M2 ...` или `-m`  Модели openrouter 7 для 7 игроков 
 `--players N` или `-p`  Количество игроков по умолч. 7
 `--discussion-rounds N`  Раунды обсуждения в день по умолч. 2
 `--log-file PATH` или `-l`  Сохранить транскрипт игры в файл 
  `--help` или `-h`  Справка 
```
## Переменные окружения
```
`OPENROUTER_API_KEY`  *(обязательно)*  API-ключ OpenRouter 
`MODELS`  встроенные  Comma-separated модели 
`NUM_PLAYERS`  `7`  Количество игроков 
`DISCUSSION_ROUNDS`  `2` | Раунды обсуждения 
`REASONING`  `1`  Включить reasoning (1/0) 
```
## Команда Для виртуального Окружения

```bash
export OPENROUTER_API_KEY=sk-or-v1-ВАШ_КЛЮЧ
.venv/bin/python run_game.py --log-file logs/game.txt
```

## Модели по умолчанию

```
poolside/laguna-s-2.1
nvidia/nemotron-3-ultra-550b-a55b
google/gemma-4-31b-it
deepseek/deepseek-v4-flash-0731
qwen/qwen3.7-flash
google/gemini-3.5-flash-lite
mistralai/mistral-medium-3
```

## Файлы

```
mafia_game/         # Исходный код
run_game.py         # Точка входа
.env.example        # Пример env (скопировать → .env)
logs/debug.log      # Технические логи
```

## Пример вывода

```
=== ИГРА МАФИЯ ===
Всего игроков: 7
Роли назначены: 2 Мафии, 5 Мирных (Детектив=1, Доктор=1, Мирный=3)

=== НОЧЬ 1 ===
  Мафия голосует: → убить deepseek/deepseek-v4-flash-0731
  Доктор спас: google/gemma-4-31b-it
  deepseek/deepseek-v4-flash-0731 убит ночью.
  Детектив расследует poolside/laguna-s-2.1: МАФИЯ

=== ДЕНЬ 1 ===
  — Раунд обсуждения 1 —
    poolside/laguna-s-2.1: "Нам нужно больше конкретики в обвинениях."
  — Голосование —
    >> poolside/laguna-s-2.1 ИСКЛЮЧЁН. Роль: Мафия

=== ИГРА ОКОНЧЕНА: ГОРОД ПОБЕДИЛА! ===
```
# Openrouter-Mafia
