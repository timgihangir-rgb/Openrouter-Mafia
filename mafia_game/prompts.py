"""Prompt template strings for all model interactions.

Every prompt that expects a structured reply tells the model to respond with
valid JSON containing at least ``"thoughts"`` and an action field.
"""


JSON_FORMAT_INSTRUCTION = """\
IMPORTANT: You MUST respond with valid JSON only. Do not include any text
before or after the JSON. Your response must be parseable by a JSON parser.

Required fields in your JSON:
  - "thoughts": Your private reasoning (string, 2-3 sentences max).
  - One additional field depending on your action (see examples below).

If your response is not valid JSON, it will be rejected and you will be asked
to try again. Make sure your output starts with {{ and ends with }}.

IMPORTANT: Write all your text in RUSSIAN language. The "thoughts" and
"statement" fields must be in Russian. The JSON field names ("thoughts",
"action", "target", "vote", "statement") must remain in English.
"""

SYSTEM_PROMPT = (
    "You are a player in a game of Mafia (also known as Werewolf). "
    "There are {total_players} players total. "
    "The game alternates between Night phases (secret actions) and Day "
    "phases (discussion and voting). "
    "Roles: Mafia (kill people, work together), Detective (investigate "
    "one player each night to learn their team), Doctor (save one player "
    "from death each night), Civilian (no special powers, just vote). "
    "Your goal depends on your role: Mafia want to eliminate all town; "
    "town wants to eliminate all Mafia. "
    "You are an AI player — make decisions based on the information given. "
    + JSON_FORMAT_INSTRUCTION
)


ROLE_ASSIGNMENT_PROMPT = """\
=== PRIVATE ROLE ASSIGNMENT ===
Your player ID is: {player_id}
Your role is: {role}

Role abilities:
  - Mafia: Each night, coordinate with other Mafia to choose a player to kill.
    You do NOT want anyone to know you are Mafia.
  - Detective: Each night, investigate one player to learn if they are
    Mafia or Town.
  - Doctor: Each night, save one player (including yourself) from the
    Mafia kill.
  - Civilian: No special night action. Help the town deduce who the
    Mafia are during the day.

Total players: {total_players}
Role distribution: {role_distribution}
Player IDs: 0 through {total_players_minus_1}

Remember: if you are Mafia, you are on the same team as other Mafia.
If you are Town (Detective, Doctor, Civilian), you are on the same team
and want to eliminate all Mafia.

Отвечайте на русском языке.
Your thoughts and statements should be in Russian.
Reply with: {{"thoughts": "понял", "action": "understood"}}
"""


NIGHT_MAFIA_PROMPT = """\
=== NIGHT {night_num} — MAFIA TURN ===
You are Mafia. It is night. You and any other alive Mafia must secretly
choose one player to kill.

Alive players: {alive_players}
Dead players: {dead_players}
Previously killed players: {killed_players}

Other alive Mafia players and their stated targets (if any):
{mafia_other_targets}

Your target this night:
Reply with: {{"thoughts": "...", "action": "kill", "target": <player_id>}}
Only target an alive player (not yourself, not already dead).
"""

NIGHT_DETECTIVE_PROMPT = """\
=== NIGHT {night_num} — DETECTIVE TURN ===
You are the Detective. Investigate one alive player to learn if they are
Mafia or Town.

Alive players: {alive_players}
Dead players: {dead_players}

Reply with: {{"thoughts": "...", "action": "investigate", "target": <player_id>}}
Only target an alive player (not yourself).
"""

NIGHT_DOCTOR_PROMPT = """\
=== NIGHT {night_num} — DOCTOR TURN ===
You are the Doctor. Choose one alive player to save from tonight's Mafia
kill. You may save yourself or another player.

Alive players: {alive_players}
Dead players: {dead_players}

Note: you cannot save the same player two nights in a row (game rule).
Previously saved players: {doctor_saved_history}

Reply with: {{"thoughts": "...", "action": "save", "target": <player_id>}}
Only target an alive player.
"""


DISCUSSION_PROMPT = """\
=== DAY {day_num} — DISCUSSION ROUND {round_num} ===
The sun rises. Players debate who the Mafia are.

Alive players: {alive_players}
Dead / revealed players: {dead_players}
Known events so far: {known_events}

Discussion history (previous statements):
{discussion_history}

Your statement (public — other players will see this):
Write your statement in RUSSIAN.
Reply with: {{"thoughts": "...", "statement": "Что вы хотите, чтобы город знал"}}
Be strategic. Share your analysis, suspicions, or defend yourself.
Keep your statement concise (1-2 sentences). Write in Russian.
"""

VOTING_PROMPT = """\
=== DAY {day_num} — VOTING ===
It is time to vote to eliminate a player. Based on the discussion, decide
who is most likely Mafia and vote to eliminate them.

Alive players: {alive_players}
Discussion summary: {discussion_summary}

Reply with: {{"thoughts": "...", "vote": <player_id>}}
Only vote for an alive player. The player with the most votes is eliminated.
In case of a tie, a re-vote is held among tied players. Write your thoughts in Russian.
"""

VOTE_REVOTE_PROMPT = """\
=== RE-VOTE ===
There was a tie in the previous vote between players: {tied_players}
You must vote for one of these tied players.

Reply with: {{"thoughts": "...", "vote": <player_id>}}
Only vote for one of the tied players: {tied_players_list}
"""

INVALID_RESPONSE_PROMPT = """\
=== ERROR IN YOUR LAST RESPONSE ===
Your last response was:
{previous_response}

Error: {error}

Please re-read the prompt and respond with valid JSON in the required format.
{original_prompt}
"""
