import os
from supabase import create_client, Client
import pandas as pd
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# Constants
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
FPL_BASE_URL = 'https://fantasy.premierleague.com/api/'

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_fpl_data(endpoint):
    """Helper function to get data from FPL API"""
    url = FPL_BASE_URL + endpoint
    return requests.get(url).json()

def sync_players():
    data = get_fpl_data("bootstrap-static")
    players = data['elements']
    success, fail = 0, 0

    for player in players:
        try:
            supabase.table("players").upsert({
                "id": player["id"],
                "web_name": player["web_name"],
                "first_name": player["first_name"],
                "second_name": player["second_name"],
                "team_id": player["team"],
                "element_type": player["element_type"],
                "now_cost": player["now_cost"],
                "total_points": player["total_points"],
                "goals_scored": player["goals_scored"],
                "assists": player["assists"],
                "minutes": player["minutes"],
                "form": player["form"],
                "selected_by_percent": player["selected_by_percent"],
                "transfers_in": player["transfers_in"],
                "transfers_out": player["transfers_out"],
                "player_code": player["code"],
                "status": player["status"],
                "chance_of_playing_next_round": player["chance_of_playing_next_round"],
                "chance_of_playing_this_round": player["chance_of_playing_this_round"],
                "now_cost_rank": player.get("now_cost_rank"),
                "now_cost_rank_type": player.get("now_cost_rank_type"),
                "cost_change_event": player.get("cost_change_event"),
                "cost_change_event_fall": player.get("cost_change_event_fall"),
                "cost_change_start": player.get("cost_change_start"),
                "cost_change_start_fall": player.get("cost_change_start_fall"),
                "selected_rank": player.get("selected_rank"),
                "selected_rank_type": player.get("selected_rank_type"),
                "event_points": player["event_points"],
                "points_per_game": player["points_per_game"],
                "points_per_game_rank": player.get("points_per_game_rank"),
                "points_per_game_rank_type": player.get("points_per_game_rank_type"),
                "form_rank": player.get("form_rank"),
                "form_rank_type": player.get("form_rank_type"),
                "value_form": player["value_form"],
                "value_season": player["value_season"],
                "dreamteam_count": player["dreamteam_count"],
                "transfers_in_event": player.get("transfers_in_event"),
                "transfers_out_event": player.get("transfers_out_event"),
                "ep_next": player.get("ep_next"),
                "ep_this": player.get("ep_this"),
                "expected_goals": player.get("expected_goals"),
                "expected_assists": player.get("expected_assists"),
                "expected_goal_involvements": player.get("expected_goal_involvements"),
                "expected_goals_conceded": player.get("expected_goals_conceded"),
                "expected_goals_per_90": player.get("expected_goals_per_90"),
                "expected_assists_per_90": player.get("expected_assists_per_90"),
                "expected_goal_involvements_per_90": player.get("expected_goal_involvements_per_90"),
                "expected_goals_conceded_per_90": player.get("expected_goals_conceded_per_90"),
                "influence_rank": player.get("influence_rank"),
                "influence_rank_type": player.get("influence_rank_type"),
                "creativity_rank": player.get("creativity_rank"),
                "creativity_rank_type": player.get("creativity_rank_type"),
                "threat_rank": player.get("threat_rank"),
                "threat_rank_type": player.get("threat_rank_type"),
                "ict_index_rank": player.get("ict_index_rank"),
                "ict_index_rank_type": player.get("ict_index_rank_type"),
                "corners_and_indirect_freekicks_order": player.get("corners_and_indirect_freekicks_order"),
                "direct_freekicks_order": player.get("direct_freekicks_order"),
                "penalties_order": player.get("penalties_order"),
                "gw": None,  # Not available in API
                # last_updated will be set by DB default
            }).execute()
            success += 1
        except Exception as e:
            print(f"Failed to upsert player {player['id']}: {e}")
            fail += 1

    print(f"sync_players: {success} succeeded, {fail} failed")

def sync_teams():
    data = get_fpl_data("bootstrap-static")
    teams = data['teams']
    success, fail = 0, 0

    for team in teams:
        try:
            supabase.table("teams").upsert({
                "id": team["id"],
                "name": team["name"],
                "short_name": team["short_name"],
                "strength": team["strength"],
                "strength_overall_home": team["strength_overall_home"],
                "strength_overall_away": team["strength_overall_away"],
                "strength_attack_home": team["strength_attack_home"],
                "strength_attack_away": team["strength_attack_away"],
                "strength_defence_home": team["strength_defence_home"],
                "strength_defence_away": team["strength_defence_away"],
                "code": team["code"],
                "pulse_id": team["pulse_id"],
                "elo": None,  # Not available in API
                # last_updated will be set by DB default
            }).execute()
            success += 1
        except Exception as e:
            print(f"Failed to upsert team {team['id']}: {e}")
            fail += 1

    print(f"sync_teams: {success} succeeded, {fail} failed")

def sync_fixtures():
    fixtures = get_fpl_data("fixtures")
    success, fail = 0, 0

    for fixture in fixtures:
        try:
            supabase.table("fixtures").upsert({
                "id": fixture["id"],
                "name": f"GW {fixture['event']}" if fixture.get("event") else "TBD",
                "deadline_time": fixture.get("kickoff_time"),
                "average_entry_score": (fixture.get("team_a_score") or 0) + (fixture.get("team_h_score") or 0),
                "finished": fixture.get("finished", False),
                "data_checked": fixture.get("data_checked", False),
                "highest_scoring_entry": None,  # Not in fixture data
                "is_previous": fixture.get("is_previous", False),
                "is_current": fixture.get("is_current", False),
                "is_next": fixture.get("is_next", False),
                "chip_deadline_passed": fixture.get("kickoff_time") is not None,
                # last_updated will be set by DB default
            }).execute()
            success += 1
        except Exception as e:
            print(f"Failed to upsert fixture {fixture['id']}: {e}")
            fail += 1

    print(f"sync_fixtures: {success} succeeded, {fail} failed")

def sync_gameweek_history():
    players = supabase.table("players").select("id").execute().data
    total_success, total_fail = 0, 0

    for player in players:
        player_id = player["id"]
        try:
            history = get_fpl_data(f"element-summary/{player_id}")["history"]
            for gw in history:
                try:
                    supabase.table("gameweek_history").upsert({
                        "player_id": player_id,
                        "round": gw["round"],
                        "total_points": gw["total_points"],
                        "minutes": gw["minutes"],
                        "goals_scored": gw["goals_scored"],
                        "assists": gw["assists"],
                        "clean_sheets": gw["clean_sheets"],
                        "goals_conceded": gw["goals_conceded"],
                        "saves": gw["saves"],
                        "bonus": gw["bonus"],
                        "bps": gw["bps"],
                        "influence": gw["influence"],
                        "creativity": gw["creativity"],
                        "threat": gw["threat"],
                        "ict_index": gw["ict_index"],
                        "was_home": gw["was_home"],
                        "kickoff_time": gw["kickoff_time"],
                        "opponent_team": gw["opponent_team"],
                        "clearances": gw.get("clearances", 0),
                        "blocks": gw.get("blocks", 0),
                        "interceptions": gw.get("interceptions", 0),
                        "tackles": gw.get("tackles", 0),
                        "ball_recoveries": gw.get("ball_recoveries", 0),
                        "defensive_contributions": gw.get("defensive_contributions", 0),
                        # last_updated will be set by DB default
                    }).execute()
                    total_success += 1
                except Exception as e:
                    print(f"Failed GW {gw['round']} for player {player_id}: {e}")
                    total_fail += 1
        except Exception as e:
            print(f"Failed to fetch history for player {player_id}: {e}")
            total_fail += 1

    print(f"sync_gameweek_history: {total_success} succeeded, {total_fail} failed")

def main():
    sync_players()
    sync_teams()
    sync_fixtures()
    sync_gameweek_history()

if __name__ == '__main__':
    main()