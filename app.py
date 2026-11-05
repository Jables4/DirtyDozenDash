"""
Sleeper fantasy league dashboard.
Run locally:  streamlit run app.py
Deploy:       push app.py + requirements.txt to GitHub, then deploy on share.streamlit.io
"""
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
LEAGUE_ID = "1389372059129229312"
BASE = "https://api.sleeper.app/v1"
MAX_WEEK = 18
SIMULATIONS = 10000

SLOT_ELIGIBILITY = {
    "QB": {"QB"}, "RB": {"RB"}, "WR": {"WR"}, "TE": {"TE"}, "K": {"K"}, "DEF": {"DEF"},
    "FLEX": {"RB", "WR", "TE"},
    "SUPER_FLEX": {"QB", "RB", "WR", "TE"},
    "REC_FLEX": {"WR", "TE"},
    "WRRB_FLEX": {"RB", "WR"},
    "DL": {"DL", "DE", "DT"}, "LB": {"LB"}, "DB": {"DB", "CB", "S"},
    "IDP_FLEX": {"DL", "DE", "DT", "LB", "DB", "CB", "S"},
}
NON_STARTER_SLOTS = {"BN", "IR", "TAXI"}
PALETTE = px.colors.qualitative.Bold + px.colors.qualitative.Safe + px.colors.qualitative.Vivid

st.set_page_config(page_title="League Dashboard", page_icon="🏈", layout="wide")

# ---------------------------------------------------------------------------
# Sleeper API
# ---------------------------------------------------------------------------
_session = requests.Session()


def get(path):
    for attempt in range(3):
        try:
            r = _session.get(f"{BASE}/{path}", timeout=20)
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(1.5 * (attempt + 1))


def get_many(paths):
    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(get, paths))


@st.cache_data(ttl=86400, show_spinner="Loading the NFL player list…")
def load_players():
    raw = get("players/nfl") or {}
    players = {}
    for pid, p in raw.items():
        name = p.get("full_name") or f"{p.get('first_name', '')} {p.get('last_name', '')}".strip() or pid
        pos = p.get("position") or ""
        fpos = p.get("fantasy_positions") or ([pos] if pos else [])
        players[pid] = {"name": name, "pos": pos, "fpos": fpos}
    return players


@st.cache_data(ttl=3600, show_spinner=False)
def load_nfl_state():
    return get("state/nfl") or {}


@st.cache_data(ttl=86400, show_spinner=False)
def load_league_chain(league_id):
    """This league plus every previous season it was renewed from."""
    chain, seen, lid = [], set(), league_id
    while lid and lid != "0" and lid not in seen:
        seen.add(lid)
        lg = get(f"league/{lid}")
        if not lg:
            break
        chain.append({"league_id": lid, "season": str(lg.get("season")), "name": lg.get("name")})
        lid = lg.get("previous_league_id")
    return chain


def _load_season(league_id):
    league, users, rosters = get_many(
        [f"league/{league_id}", f"league/{league_id}/users", f"league/{league_id}/rosters"])
    weeks = range(1, MAX_WEEK + 1)
    matchups = get_many([f"league/{league_id}/matchups/{w}" for w in weeks])
    transactions = get_many([f"league/{league_id}/transactions/{w}" for w in weeks])
    drafts = get(f"league/{league_id}/drafts") or []
    picks = []
    if drafts:
        main = next((d for d in drafts if d.get("status") == "complete"), drafts[0])
        picks = get(f"draft/{main['draft_id']}/picks") or []
    try:
        bracket = get(f"league/{league_id}/winners_bracket") or []
    except requests.RequestException:
        bracket = []
    return {
        "league": league,
        "users": users or [],
        "rosters": rosters or [],
        "matchups": {w: (m or []) for w, m in zip(weeks, matchups)},
        "transactions": [t for ts in transactions for t in (ts or [])],
        "picks": picks,
        "bracket": bracket,
    }


load_current_season = st.cache_data(ttl=900, show_spinner="Loading league data…")(_load_season)
load_past_season = st.cache_data(ttl=7 * 86400, show_spinner="Loading past seasons…")(_load_season)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def team_labels(users, rosters, mode):
    umap = {u["user_id"]: u for u in users}
    labels = {}
    for r in rosters:
        u = umap.get(r.get("owner_id"))
        if u is None:
            labels[r["roster_id"]] = f"Team {r['roster_id']}"
        elif mode == "Team name":
            labels[r["roster_id"]] = (u.get("metadata") or {}).get("team_name") or u.get("display_name")
        else:
            labels[r["roster_id"]] = u.get("display_name")
    # keep labels unique
    counts = defaultdict(int)
    for rid in sorted(labels):
        counts[labels[rid]] += 1
        if counts[labels[rid]] > 1:
            labels[rid] = f"{labels[rid]} ({rid})"
    return labels


def season_weeks(league, state):
    """Return (regular-season weeks, finished regular-season weeks, live week or None)."""
    s = league.get("settings", {})
    start = int(s.get("start_week") or 1)
    playoff_start = int(s.get("playoff_week_start") or 15)
    regular = list(range(start, playoff_start))
    season = str(league.get("season"))
    if str(state.get("season")) == season and state.get("season_type") == "pre":
        return regular, [], None
    if str(state.get("season")) == season and state.get("season_type") == "regular":
        current = int(state.get("week") or 1)
        finished = [w for w in regular if w < current]
        live = current if current in regular else None
        return regular, finished, live
    return regular, regular, None


def optimal_lineup(players_points, slots, players):
    """Best possible lineup: returns (points, set of player ids)."""
    pool = sorted(players_points.items(), key=lambda kv: -(kv[1] or 0))
    used, total = set(), 0.0
    for slot in sorted(slots, key=lambda s: len(SLOT_ELIGIBILITY.get(s, {s}))):
        eligible = SLOT_ELIGIBILITY.get(slot, {slot})
        for pid, pts in pool:
            if pid in used:
                continue
            if set(players.get(pid, {}).get("fpos") or []) & eligible:
                used.add(pid)
                total += pts or 0
                break
    return total, used


def optimal_points(players_points, slots, players):
    return optimal_lineup(players_points, slots, players)[0]


def build_weekly(data, weeks, players, labels):
    slots = [s for s in data["league"].get("roster_positions", []) if s not in NON_STARTER_SLOTS]
    rows, starter_rows = [], []
    for wk in weeks:
        ms = data["matchups"].get(wk) or []
        pts_by_rid = {m["roster_id"]: float(m.get("custom_points") or m.get("points") or 0) for m in ms}
        groups = defaultdict(list)
        for m in ms:
            if m.get("matchup_id") is not None:
                groups[m["matchup_id"]].append(m["roster_id"])
        for m in ms:
            rid = m["roster_id"]
            pts = pts_by_rid[rid]
            opp = next((o for o in groups.get(m.get("matchup_id"), []) if o != rid), None)
            opp_pts = pts_by_rid.get(opp) if opp is not None else None
            result = None
            if opp_pts is not None:
                result = "W" if pts > opp_pts else "L" if pts < opp_pts else "T"
            best = optimal_points(m.get("players_points") or {}, slots, players)
            rows.append({
                "week": wk, "roster_id": rid, "team": labels.get(rid, str(rid)),
                "points": pts, "optimal": max(best, pts),
                "opp_id": opp, "opponent": labels.get(opp) if opp is not None else None,
                "opp_points": opp_pts, "result": result, "matchup_id": m.get("matchup_id"),
            })
            for pid, spts in zip(m.get("starters") or [], m.get("starters_points") or []):
                if pid and pid != "0":
                    starter_rows.append({
                        "week": wk, "roster_id": rid, "team": labels.get(rid, str(rid)),
                        "player_id": pid, "player": players.get(pid, {}).get("name", pid),
                        "pos": players.get(pid, {}).get("pos", "?"), "points": float(spts or 0),
                    })
    return pd.DataFrame(rows), pd.DataFrame(starter_rows)


def add_median_games(wdf, enabled):
    """Leagues with 'play the league median' get an extra W/L each week."""
    wdf = wdf.copy()
    wdf["median_result"] = None
    if enabled and not wdf.empty:
        med = wdf.groupby("week")["points"].transform("median")
        wdf["median_result"] = np.where(wdf["points"] > med, "W", np.where(wdf["points"] < med, "L", "T"))
    return wdf


def standings(wdf, n_teams):
    if wdf.empty:
        return pd.DataFrame()
    df = wdf.copy()
    df["ap_wins"] = df.groupby("week")["points"].rank(method="average") - 1
    df["bench"] = df["optimal"] - df["points"]
    out = []
    for team, g in df.groupby("team"):
        res = pd.concat([g["result"], g["median_result"]]).dropna()
        w, l, t = (res == "W").sum(), (res == "L").sum(), (res == "T").sum()
        ap_w = g["ap_wins"].sum()
        ap_games = len(g) * (n_teams - 1)
        x_w = g["ap_wins"].sum() / (n_teams - 1)
        if g["median_result"].notna().any():
            x_w *= 2  # median leagues: each week is worth two games
        out.append({
            "Team": team, "W": int(w), "L": int(l), "T": int(t),
            "PF": g["points"].sum(), "PA": g["opp_points"].sum(),
            "All-play W": ap_w, "All-play L": ap_games - ap_w,
            "All-play %": ap_w / ap_games if ap_games else 0,
            "Expected W": x_w, "Luck": w + 0.5 * t - x_w,
            "Max PF": g["optimal"].sum(), "Efficiency": g["points"].sum() / g["optimal"].sum() if g["optimal"].sum() else 0,
            "Bench pts": g["bench"].sum(), "Avg": g["points"].mean(), "Std dev": g["points"].std(ddof=0),
            "High": g["points"].max(), "Low": g["points"].min(),
        })
    out = pd.DataFrame(out).sort_values(["W", "PF"], ascending=False).reset_index(drop=True)
    out.index += 1
    return out


def color_map(teams):
    return {t: PALETTE[i % len(PALETTE)] for i, t in enumerate(sorted(teams))}


def style_fig(fig, height=None):
    fig.update_layout(
        margin=dict(l=10, r=10, t=90, b=10),
        title=dict(yref="container", y=0.98, yanchor="top", x=0, xanchor="left", pad=dict(l=10)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None),
        font=dict(size=13),
    )
    if height:
        fig.update_layout(height=height)
    return fig


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
try:
    chain = load_league_chain(LEAGUE_ID)
    state = load_nfl_state()
    players = load_players()
except requests.RequestException as e:
    st.error(f"Couldn't reach Sleeper ({e}). Check that LEAGUE_ID is correct, then reload the page.")
    st.stop()

if not chain:
    st.error("Sleeper didn't return a league for this ID. Check LEAGUE_ID at the top of app.py.")
    st.stop()

with st.sidebar:
    st.header("Options")
    season_choice = st.selectbox("Season", [c["season"] for c in chain], index=0)
    label_mode = st.radio("Label teams by", ["Team name", "Manager"], horizontal=True)
    if st.button("Refresh data"):
        st.cache_data.clear()
        st.rerun()
    st.caption("Data refreshes automatically every 15 minutes.")

season_meta = next(c for c in chain if c["season"] == season_choice)
loader = load_current_season if season_meta == chain[0] else load_past_season
data = loader(season_meta["league_id"])
league = data["league"]
settings = league.get("settings", {})
labels = team_labels(data["users"], data["rosters"], label_mode)
n_teams = len(data["rosters"])
regular_weeks, finished_weeks, live_week = season_weeks(league, state)
median_league = bool(settings.get("league_average_match"))

all_weekly, all_starters = build_weekly(data, sorted(set(regular_weeks)), players, labels)
all_weekly = add_median_games(all_weekly, median_league)
cmap = color_map(labels.values())

st.title(league.get("name") or "League dashboard")
if not finished_weeks:
    st.info("No regular-season weeks are final yet. Check back after week 1 wraps up.")
    st.stop()

with st.sidebar:
    if len(finished_weeks) > 1:
        wk_range = st.slider("Weeks to include", min(finished_weeks), max(finished_weeks),
                             (min(finished_weeks), max(finished_weeks)))
    else:
        wk_range = (finished_weeks[0], finished_weeks[0])

sel_weeks = [w for w in finished_weeks if wk_range[0] <= w <= wk_range[1]]
weekly = all_weekly[all_weekly["week"].isin(sel_weeks)]
starters = all_starters[all_starters["week"].isin(sel_weeks)] if not all_starters.empty else all_starters
table = standings(weekly, n_teams)

subtitle = f"{season_choice} season, weeks {wk_range[0]}–{wk_range[1]}"
if live_week:
    subtitle += f". Week {live_week} is in progress and not counted yet."
if median_league:
    subtitle += " Records include the weekly median game."
st.caption(subtitle)

tabs = st.tabs(["Weekly awards", "Standings & luck", "Weekly scores", "Managers", "Power rankings",
                "Trades & waivers", "Draft", "Playoff odds", "History"])

# ---------------------------------------------------------------------------
# Weekly awards
# ---------------------------------------------------------------------------
def lineup_swaps(m, slots):
    """Players who should have started vs. the starters they'd have replaced."""
    pp = m.get("players_points") or {}
    _, best = optimal_lineup(pp, slots, players)
    started = {p for p in (m.get("starters") or []) if p and p != "0"}
    should_start = sorted(best - started, key=lambda p: -(pp.get(p) or 0))
    should_sit = sorted(started - best, key=lambda p: pp.get(p) or 0)
    return [(players.get(a, {}).get("name", a), pp.get(a) or 0, players.get(b, {}).get("name", b), pp.get(b) or 0)
            for a, b in zip(should_start, should_sit)]


def week_awards(wk):
    wd = all_weekly[all_weekly["week"] == wk].copy()
    raw = {m["roster_id"]: m for m in data["matchups"].get(wk) or []}
    slots = [s for s in league.get("roster_positions", []) if s not in NON_STARTER_SLOTS]
    wd["bench"] = wd["optimal"] - wd["points"]
    a = {}
    top, low = wd.loc[wd["points"].idxmax()], wd.loc[wd["points"].idxmin()]
    a["top"] = (top["team"], f"{top['points']:.1f} points")
    a["low"] = (low["team"], f"{low['points']:.1f} points")

    winners, losers = wd[wd["result"] == "W"], wd[wd["result"] == "L"]
    if not winners.empty:
        w = winners.loc[winners["points"].idxmin()]
        above = int((wd["points"] > w["points"]).sum())
        a["lucky"] = (w["team"], f"Beat {w['opponent']} with {w['points']:.1f}, "
                                 f"even though {above} team{'s' if above != 1 else ''} scored more")
    if not losers.empty:
        l = losers.loc[losers["points"].idxmax()]
        below = int((wd["points"] < l["points"]).sum())
        a["tough"] = (l["team"], f"Lost to {l['opponent']} with {l['points']:.1f}, "
                                 f"which would have beaten {below} other team{'s' if below != 1 else ''}")

    b = wd.loc[wd["bench"].idxmax()]
    detail = f"Left {b['bench']:.1f} points on the bench"
    swaps = lineup_swaps(raw.get(b["roster_id"], {}), slots)
    if swaps:
        good, gp, bad, bp = swaps[0]
        detail += f". Started {bad} ({bp:.1f}) over {good} ({gp:.1f})"
    a["blunder"] = (b["team"], detail)

    perfect = wd[wd["bench"] < 0.05]["team"].tolist()
    if perfect:
        a["coach"] = (", ".join(perfect), "Started the best possible lineup")
    else:
        c = wd.loc[(wd["points"] / wd["optimal"]).idxmax()]
        a["coach"] = (c["team"], f"Started {c['points'] / c['optimal']:.1%} of their best possible lineup")

    st_wk = all_starters[all_starters["week"] == wk] if not all_starters.empty else all_starters
    if not st_wk.empty:
        p = st_wk.loc[st_wk["points"].idxmax()]
        a["player"] = (p["player"], f"{p['points']:.1f} points for {p['team']}")
    bench_rows = []
    for rid, m in raw.items():
        started = set(m.get("starters") or [])
        for pid, pts in (m.get("players_points") or {}).items():
            if pid not in started:
                bench_rows.append((pts or 0, pid, rid))
    if bench_rows:
        pts, pid, rid = max(bench_rows)
        if pts > 0:
            a["benchmvp"] = (players.get(pid, {}).get("name", pid), f"{pts:.1f} points on {labels.get(rid, rid)}'s bench")

    games = wd.dropna(subset=["opp_points"])
    games = games[games["points"] > games["opp_points"]]
    if not games.empty:
        margin = games["points"] - games["opp_points"]
        g = games.loc[margin.idxmax()]
        a["blowout"] = (g["team"], f"Beat {g['opponent']} by {g['points'] - g['opp_points']:.1f}")
        g = games.loc[margin.idxmin()]
        a["close"] = (g["team"], f"Beat {g['opponent']} by {g['points'] - g['opp_points']:.2f}")
    return a


AWARDS = [
    ("top", "🔥", "Top score"), ("low", "🧊", "Lowest score"),
    ("lucky", "🍀", "Luckiest win"), ("tough", "💔", "Toughest loss"),
    ("blunder", "🪑", "Bench blunder"), ("coach", "📋", "Coach of the week"),
    ("player", "⭐", "Player of the week"), ("benchmvp", "😭", "Bench MVP"),
    ("blowout", "💥", "Blowout"), ("close", "😬", "Nail-biter"),
]

with tabs[0]:
    award_wk = st.selectbox("Week", finished_weeks[::-1], format_func=lambda w: f"Week {w}")
    aw = week_awards(award_wk)
    shown = [x for x in AWARDS if x[0] in aw]
    for row in range(0, len(shown), 3):
        cols = st.columns(3)
        for col, (key, emoji, title) in zip(cols, shown[row:row + 3]):
            who, what = aw[key]
            with col.container(border=True):
                st.markdown(f"{emoji} **{title}**")
                st.markdown(f"### {who}")
                st.caption(what)

    recap = [f"🏈 {league.get('name') or 'League'}: week {award_wk} awards", ""]
    recap += [f"{emoji} {title}: {aw[key][0]}, {aw[key][1][0].lower()}{aw[key][1][1:]}" for key, emoji, title in shown]
    st.markdown("**Copy for the group chat**")
    st.code("\n".join(recap), language=None, wrap_lines=True)

    tally_keys = [("top", "🔥 Top score"), ("low", "🧊 Lowest"), ("lucky", "🍀 Lucky win"),
                  ("tough", "💔 Tough loss"), ("blunder", "🪑 Blunder"), ("coach", "📋 Coach")]
    tally = pd.DataFrame(0, index=sorted(labels.values()), columns=[c for _, c in tally_keys])
    for wk in finished_weeks:
        wa = week_awards(wk)
        for key, colname in tally_keys:
            if key in wa:
                for team in wa[key][0].split(", "):
                    if team in tally.index:
                        tally.loc[team, colname] += 1
    tally = tally.sort_values(list(tally.columns), ascending=False)
    st.markdown("**Season trophy case**")
    st.dataframe(tally, width="stretch")

# ---------------------------------------------------------------------------
# Standings & luck
# ---------------------------------------------------------------------------
with tabs[1]:
    show = table[["Team", "W", "L", "T", "PF", "PA", "All-play W", "All-play L", "Expected W", "Luck"]].copy()
    if not show["T"].any():
        show = show.drop(columns="T")
    st.dataframe(
        show, width="stretch",
        column_config={
            "PF": st.column_config.NumberColumn(format="%.1f"),
            "PA": st.column_config.NumberColumn(format="%.1f"),
            "All-play W": st.column_config.NumberColumn(format="%.1f"),
            "All-play L": st.column_config.NumberColumn(format="%.1f"),
            "Expected W": st.column_config.NumberColumn(format="%.1f", help="Wins you'd expect from your scores alone"),
            "Luck": st.column_config.NumberColumn(format="%+.1f", help="Actual wins minus expected wins"),
        },
    )
    st.caption("All-play: your record if you played every team every week. "
               "Luck: actual wins minus the wins your scores earned on average.")

    c1, c2 = st.columns(2)
    with c1:
        luck = table.sort_values("Luck")
        fig = go.Figure(go.Bar(
            x=luck["Luck"], y=luck["Team"], orientation="h",
            marker_color=np.where(luck["Luck"] >= 0, "#2a9d8f", "#e76f51"),
            hovertemplate="%{y}: %{x:+.1f} wins<extra></extra>"))
        fig.update_layout(title="Luck: wins above or below expected", xaxis_title="Wins")
        st.plotly_chart(style_fig(fig, 420), width="stretch")
    with c2:
        fig = px.scatter(table, x="PF", y="PA", text="Team", color="Team", color_discrete_map=cmap,
                         hover_data={"W": True, "L": True, "Team": False})
        fig.add_vline(x=table["PF"].mean(), line_dash="dot", line_color="gray")
        fig.add_hline(y=table["PA"].mean(), line_dash="dot", line_color="gray")
        fig.update_traces(textposition="top center", marker_size=12)
        x0, x1 = table["PF"].min(), table["PF"].max()
        y0, y1 = table["PA"].min(), table["PA"].max()
        for x, y, txt in [(x1, y0, "Good and lucky"), (x1, y1, "Good, tough schedule"),
                          (x0, y0, "Weak, easy schedule"), (x0, y1, "Weak and unlucky")]:
            fig.add_annotation(x=x, y=y, text=txt, showarrow=False, font=dict(color="gray", size=11))
        fig.update_layout(title="Points for vs. points against", showlegend=False)
        st.plotly_chart(style_fig(fig, 420), width="stretch")

# ---------------------------------------------------------------------------
# Weekly scores
# ---------------------------------------------------------------------------
with tabs[2]:
    games = weekly.dropna(subset=["opp_points"]).copy()
    games = games[games["roster_id"] < games["opp_id"]]
    games["margin"] = (games["points"] - games["opp_points"]).abs()
    top = weekly.loc[weekly["points"].idxmax()]
    low = weekly.loc[weekly["points"].idxmin()]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Highest score", f"{top['points']:.1f}", f"{top['team']}, week {top['week']}", delta_color="off")
    m2.metric("Lowest score", f"{low['points']:.1f}", f"{low['team']}, week {low['week']}", delta_color="off")
    if not games.empty:
        blow = games.loc[games["margin"].idxmax()]
        close = games.loc[games["margin"].idxmin()]
        m3.metric("Biggest blowout", f"{blow['margin']:.1f}", f"{blow['team']} vs {blow['opponent']}, wk {blow['week']}", delta_color="off")
        m4.metric("Closest game", f"{close['margin']:.2f}", f"{close['team']} vs {close['opponent']}, wk {close['week']}", delta_color="off")

    view = st.radio("Show", ["Points", "Weekly rank"], horizontal=True)
    grid = weekly.pivot(index="team", columns="week", values="points")
    grid = grid.loc[grid.mean(axis=1).sort_values(ascending=False).index]
    if view == "Weekly rank":
        grid = grid.rank(ascending=False, method="min").astype(int)
        fig = px.imshow(grid, text_auto=True, aspect="auto", color_continuous_scale="RdYlGn_r")
    else:
        fig = px.imshow(grid.round(1), text_auto=".0f", aspect="auto", color_continuous_scale="RdYlGn")
    fig.update_layout(title=f"{view} by week", xaxis_title="Week", yaxis_title=None, coloraxis_showscale=False)
    fig.update_xaxes(dtick=1)
    st.plotly_chart(style_fig(fig, 60 + 38 * n_teams), width="stretch")

    team_pick = st.selectbox("Team score trend", sorted(labels.values()))
    tdf = weekly[weekly["team"] == team_pick]
    fig = go.Figure()
    fig.add_scatter(x=tdf["week"], y=tdf["points"], mode="lines+markers", name=team_pick,
                    line_color=cmap[team_pick])
    fig.add_scatter(x=tdf["week"], y=tdf["opp_points"], mode="lines+markers", name="Opponent",
                    line=dict(color="gray", dash="dot"), text=tdf["opponent"],
                    hovertemplate="%{text}: %{y:.1f}<extra></extra>")
    league_avg = weekly.groupby("week")["points"].mean()
    fig.add_scatter(x=league_avg.index, y=league_avg.values, mode="lines", name="League average",
                    line=dict(color="lightgray"))
    fig.update_layout(xaxis_title="Week", yaxis_title="Points")
    fig.update_xaxes(dtick=1)
    st.plotly_chart(style_fig(fig, 380), width="stretch")

# ---------------------------------------------------------------------------
# Managers
# ---------------------------------------------------------------------------
with tabs[3]:
    c1, c2 = st.columns(2)
    with c1:
        eff = table.sort_values("Efficiency")
        fig = px.bar(eff, x="Efficiency", y="Team", orientation="h", color="Team", color_discrete_map=cmap,
                     text=eff["Efficiency"].map("{:.1%}".format))
        fig.update_layout(title="Lineup efficiency (points ÷ best possible)", showlegend=False,
                          xaxis=dict(tickformat=".0%", range=[max(0, eff["Efficiency"].min() - 0.1), 1]))
        st.plotly_chart(style_fig(fig, 420), width="stretch")
    with c2:
        bench = table.sort_values("Bench pts")
        fig = px.bar(bench, x="Bench pts", y="Team", orientation="h", color="Team", color_discrete_map=cmap,
                     text=bench["Bench pts"].map("{:.0f}".format))
        fig.update_layout(title="Points left on the bench", showlegend=False)
        st.plotly_chart(style_fig(fig, 420), width="stretch")

    bgrid = (weekly.assign(bench=weekly["optimal"] - weekly["points"])
             .pivot(index="team", columns="week", values="bench"))
    bgrid = bgrid.loc[bgrid.sum(axis=1).sort_values(ascending=False).index]
    fig = px.imshow(bgrid.round(1), text_auto=".0f", aspect="auto", color_continuous_scale="Reds")
    fig.update_layout(title="Bench points left behind, by week", xaxis_title="Week", yaxis_title=None,
                      coloraxis_showscale=False)
    fig.update_xaxes(dtick=1)
    st.plotly_chart(style_fig(fig, 60 + 38 * n_teams), width="stretch")

    order = table.sort_values("Avg", ascending=False)["Team"].tolist()
    fig = px.box(weekly, x="team", y="points", color="team", color_discrete_map=cmap, points="all",
                 category_orders={"team": order}, hover_data=["week"])
    fig.update_layout(title="Weekly score range (tighter box = more consistent)", showlegend=False,
                      xaxis_title=None, yaxis_title="Points")
    st.plotly_chart(style_fig(fig, 440), width="stretch")

    if not starters.empty:
        share = st.radio("Positional scoring", ["Total points", "Share of team points"], horizontal=True)
        pos = starters.groupby(["team", "pos"])["points"].sum().reset_index()
        if share == "Share of team points":
            pos["points"] = pos["points"] / pos.groupby("team")["points"].transform("sum")
        fig = px.bar(pos, x="team", y="points", color="pos", category_orders={"team": order},
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(title="Starter points by position", xaxis_title=None,
                          yaxis_title="Share" if share != "Total points" else "Points",
                          yaxis_tickformat=".0%" if share != "Total points" else None)
        st.plotly_chart(style_fig(fig, 440), width="stretch")

# ---------------------------------------------------------------------------
# Power rankings
# ---------------------------------------------------------------------------
with tabs[4]:
    st.caption("Score = 40% all-play win rate + 30% season scoring average + 30% last-3-week average, "
               "each scaled across the league.")
    hist = []
    full = all_weekly[all_weekly["week"].isin(finished_weeks)].copy()
    full["ap"] = full.groupby("week")["points"].rank(method="average") - 1
    for wk in finished_weeks:
        sub = full[full["week"] <= wk]
        g = sub.groupby("team")
        ap = g["ap"].sum() / (g.size() * (n_teams - 1))
        avg = g["points"].mean()
        recent = sub[sub["week"] > wk - 3].groupby("team")["points"].mean()

        def scale(s):
            return (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else s * 0 + 0.5

        score = 0.4 * ap + 0.3 * scale(avg) + 0.3 * scale(recent)
        rank = score.rank(ascending=False, method="first").astype(int)
        for team in score.index:
            hist.append({"week": wk, "team": team, "score": score[team], "rank": rank[team]})
    ph = pd.DataFrame(hist)
    last_wk = finished_weeks[-1]
    now = ph[ph["week"] == last_wk].set_index("team")
    prev = ph[ph["week"] == finished_weeks[-2]].set_index("team") if len(finished_weeks) > 1 else now
    pr = now.assign(change=prev["rank"] - now["rank"]).sort_values("rank").reset_index()
    pr["Move"] = pr["change"].map(lambda c: f"▲ {c}" if c > 0 else f"▼ {-c}" if c < 0 else "–")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.dataframe(pr[["rank", "team", "score", "Move"]].rename(columns={"rank": "Rank", "team": "Team", "score": "Score"}),
                     hide_index=True, width="stretch",
                     column_config={"Score": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})
    with c2:
        all_teams = pr["team"].tolist()
        highlight = st.multiselect("Highlight teams", all_teams, default=all_teams[:3])
        fig = go.Figure()
        # gray teams first so highlighted lines draw on top
        for team in sorted(all_teams, key=lambda t: t in highlight):
            t = ph[ph["team"] == team].sort_values("week")
            on = team in highlight
            color = cmap[team] if on else "rgba(150,150,150,0.3)"
            fig.add_scatter(x=t["week"], y=t["rank"], mode="lines+markers" if on else "lines",
                            line=dict(color=color, width=3.5 if on else 1.5), marker=dict(size=8),
                            showlegend=False, hovertemplate=f"{team}<br>Week %{{x}}: #%{{y}}<extra></extra>")
            last = t.iloc[-1]
            fig.add_annotation(x=last["week"], y=last["rank"], text=f"{int(last['rank'])}. {team}",
                               xanchor="left", xshift=10, showarrow=False,
                               font=dict(size=12, color=cmap[team] if on else "rgba(150,150,150,0.9)"))
        fig.update_yaxes(autorange="reversed", dtick=1, title="Rank", showgrid=False)
        fig.update_xaxes(dtick=1, title="Week")
        fig.update_layout(title="Power rank over the season")
        style_fig(fig, 480).update_layout(margin=dict(r=200))
        st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------------------
# Trades & waivers
# ---------------------------------------------------------------------------
with tabs[5]:
    txns = [t for t in data["transactions"] if t.get("status") == "complete"]
    started = (all_starters[all_starters["week"].isin(finished_weeks)]
               if not all_starters.empty else all_starters)

    def points_for_team_since(pid, rid, week):
        if started.empty:
            return 0.0
        m = (started["player_id"] == pid) & (started["roster_id"] == rid) & (started["week"] >= week)
        return float(started.loc[m, "points"].sum())

    st.subheader("Trades")
    trades = sorted([t for t in txns if t.get("type") == "trade"], key=lambda t: t.get("created", 0))
    if not trades:
        st.write("No trades yet this season.")
    trade_rows = []
    for i, t in enumerate(trades, 1):
        wk = t.get("leg") or 1
        sides = t.get("roster_ids") or []
        cols = st.columns(len(sides) or 1)
        for col, rid in zip(cols, sides):
            got = [pid for pid, to in (t.get("adds") or {}).items() if to == rid]
            pts = sum(points_for_team_since(pid, rid, wk) for pid in got)
            picks = [f"{p['season']} round {p['round']} pick" for p in (t.get("draft_picks") or [])
                     if p.get("owner_id") == rid]
            faab = [f"${w['amount']} FAAB" for w in (t.get("waiver_budget") or []) if w.get("receiver") == rid]
            with col:
                st.markdown(f"**{labels.get(rid, rid)}** received")
                lines = [f"- {players.get(p, {}).get('name', p)}: {points_for_team_since(p, rid, wk):.1f} pts started"
                         for p in got] + [f"- {x}" for x in picks + faab]
                st.markdown("\n".join(lines) or "- nothing")
            trade_rows.append({"trade": f"Trade {i} (wk {wk})", "team": labels.get(rid, rid), "points": pts})
        st.caption(f"Trade {i}, week {wk}. Points count only weeks the player started for his new team.")
        st.divider()
    if trade_rows:
        fig = px.bar(pd.DataFrame(trade_rows), x="trade", y="points", color="team", barmode="group",
                     color_discrete_map=cmap)
        fig.update_layout(title="Starter points produced by what each side received", xaxis_title=None,
                          yaxis_title="Points")
        st.plotly_chart(style_fig(fig, 400), width="stretch")

    st.subheader("Waivers and free agents")
    pickups = []
    for t in txns:
        if t.get("type") not in ("waiver", "free_agent"):
            continue
        wk = t.get("leg") or 1
        bid = (t.get("settings") or {}).get("waiver_bid") or 0
        for pid, rid in (t.get("adds") or {}).items():
            pickups.append({"Team": labels.get(rid, rid), "Player": players.get(pid, {}).get("name", pid),
                            "Week": wk, "Type": "Waiver" if t["type"] == "waiver" else "Free agent",
                            "FAAB": bid, "Pts started": points_for_team_since(pid, rid, wk)})
    if pickups:
        pdf = pd.DataFrame(pickups)
        summary = (pdf.groupby("Team").agg(Moves=("Player", "count"), FAAB=("FAAB", "sum"),
                                           Points=("Pts started", "sum")).reset_index())
        summary["Pts per $"] = np.where(summary["FAAB"] > 0, summary["Points"] / summary["FAAB"].replace(0, np.nan), np.nan)
        c1, c2 = st.columns([3, 2])
        with c1:
            fig = px.scatter(summary, x="Moves", y="Points", size=summary["FAAB"].clip(lower=1), text="Team",
                             color="Team", color_discrete_map=cmap)
            fig.update_traces(textposition="top center")
            fig.update_layout(title="Pickup activity vs. points started (bubble = FAAB spent)", showlegend=False,
                              yaxis_title="Points started from pickups")
            st.plotly_chart(style_fig(fig, 420), width="stretch")
        with c2:
            st.markdown("**Best pickups**")
            st.dataframe(pdf.sort_values("Pts started", ascending=False).head(10), hide_index=True,
                         width="stretch",
                         column_config={"Pts started": st.column_config.NumberColumn(format="%.1f"),
                                        "FAAB": st.column_config.NumberColumn(format="$%d")})
        st.dataframe(summary.sort_values("Points", ascending=False), hide_index=True, width="stretch",
                     column_config={"Points": st.column_config.NumberColumn(format="%.1f"),
                                    "FAAB": st.column_config.NumberColumn(format="$%d"),
                                    "Pts per $": st.column_config.NumberColumn(format="%.2f")})
    else:
        st.write("No waiver or free-agent pickups yet.")

# ---------------------------------------------------------------------------
# Draft
# ---------------------------------------------------------------------------
with tabs[6]:
    picks = data["picks"]
    if not picks:
        st.write("No draft found for this season.")
    else:
        season_pts = defaultdict(float)
        for wk in finished_weeks:
            for m in data["matchups"].get(wk) or []:
                for pid, p in (m.get("players_points") or {}).items():
                    season_pts[pid] += p or 0
        rows = []
        for p in picks:
            pid = p.get("player_id")
            meta = p.get("metadata") or {}
            rows.append({
                "Pick": p.get("pick_no"), "Round": p.get("round"),
                "Team": labels.get(p.get("roster_id"), "?"),
                "Player": players.get(pid, {}).get("name") or f"{meta.get('first_name', '')} {meta.get('last_name', '')}",
                "Pos": players.get(pid, {}).get("pos") or meta.get("position", ""),
                "Points": season_pts.get(pid, 0.0),
            })
        ddf = pd.DataFrame(rows).sort_values("Pick")
        if len(ddf) > 3:
            coef = np.polyfit(np.log(ddf["Pick"]), ddf["Points"], 1)
            ddf["Expected"] = np.polyval(coef, np.log(ddf["Pick"])).clip(min=0)
        else:
            ddf["Expected"] = ddf["Points"].mean()
        ddf["Value"] = ddf["Points"] - ddf["Expected"]

        fig = px.scatter(ddf, x="Pick", y="Points", color="Team", color_discrete_map=cmap,
                         hover_data=["Player", "Pos", "Round"], symbol="Pos")
        fig.add_scatter(x=ddf["Pick"], y=ddf["Expected"], mode="lines", name="Expected for that pick",
                        line=dict(color="gray", dash="dot"))
        fig.update_layout(title="Draft pick vs. fantasy points scored while rostered", yaxis_title="Points")
        st.plotly_chart(style_fig(fig, 460), width="stretch")

        grades = ddf.groupby("Team").agg(Points=("Points", "sum"), Value=("Value", "sum")).reset_index()
        grades = grades.sort_values("Value", ascending=False)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Draft value by team**")
            st.dataframe(grades, hide_index=True, width="stretch",
                         column_config={"Points": st.column_config.NumberColumn(format="%.0f"),
                                        "Value": st.column_config.NumberColumn(format="%+.0f", help="Points above what those draft slots usually produce")})
        with c2:
            st.markdown("**Best values**")
            st.dataframe(ddf.sort_values("Value", ascending=False).head(8)[["Player", "Team", "Round", "Value"]],
                         hide_index=True, width="stretch",
                         column_config={"Value": st.column_config.NumberColumn(format="%+.0f")})
        with c3:
            st.markdown("**Biggest busts**")
            st.dataframe(ddf.sort_values("Value").head(8)[["Player", "Team", "Round", "Value"]],
                         hide_index=True, width="stretch",
                         column_config={"Value": st.column_config.NumberColumn(format="%+.0f")})

# ---------------------------------------------------------------------------
# Playoff odds
# ---------------------------------------------------------------------------
with tabs[7]:
    remaining = [w for w in regular_weeks if w not in finished_weeks]
    playoff_teams = int(settings.get("playoff_teams") or 6)
    if not remaining:
        st.write("The regular season is over. Final standings are on the first tab.")
    else:
        final = all_weekly[all_weekly["week"].isin(finished_weeks)]
        rids = sorted(labels)
        idx = {r: i for i, r in enumerate(rids)}
        lg_mu, lg_sd = final["points"].mean(), final["points"].std()
        g = final.groupby("roster_id")["points"]
        n = g.size().reindex(rids).fillna(0).values
        w_shrink = n / (n + 4)
        mu = w_shrink * g.mean().reindex(rids).fillna(lg_mu).values + (1 - w_shrink) * lg_mu
        sd = np.sqrt(w_shrink * g.std().reindex(rids).fillna(lg_sd).values ** 2 + (1 - w_shrink) * lg_sd ** 2)

        base = standings(final, n_teams).set_index("Team")
        wins0 = np.array([base.loc[labels[r], "W"] + 0.5 * base.loc[labels[r], "T"] for r in rids], dtype=float)
        pf0 = np.array([base.loc[labels[r], "PF"] for r in rids], dtype=float)

        rng = np.random.default_rng(7)
        wins = np.tile(wins0, (SIMULATIONS, 1))
        pf = np.tile(pf0, (SIMULATIONS, 1))
        for wk in remaining:
            ms = data["matchups"].get(wk) or []
            pairs = defaultdict(list)
            for m in ms:
                if m.get("matchup_id") is not None and m["roster_id"] in idx:
                    pairs[m["matchup_id"]].append(idx[m["roster_id"]])
            scores = rng.normal(mu, sd, size=(SIMULATIONS, len(rids)))
            pf += scores
            for pair in pairs.values():
                if len(pair) == 2:
                    a, b = pair
                    wins[:, a] += scores[:, a] > scores[:, b]
                    wins[:, b] += scores[:, b] > scores[:, a]
            if median_league:
                med = np.median(scores, axis=1, keepdims=True)
                wins += scores > med
        key = wins * 1e5 + pf
        seed = np.argsort(np.argsort(-key, axis=1), axis=1) + 1
        odds = pd.DataFrame({
            "Team": [labels[r] for r in rids],
            "Current W": wins0,
            "Projected W": wins.mean(axis=0),
            "Make playoffs": (seed <= playoff_teams).mean(axis=0),
            "Top-2 seed": (seed <= 2).mean(axis=0),
            "#1 seed": (seed == 1).mean(axis=0),
            "Last place": (seed == len(rids)).mean(axis=0),
        }).sort_values(["Make playoffs", "Projected W"], ascending=False)
        st.caption(f"{SIMULATIONS:,} simulations of the {len(remaining)} remaining regular-season weeks. "
                   f"Top {playoff_teams} make it; ties broken by points for. Each team's scoring is modeled "
                   "from its results so far, pulled toward the league average early in the season.")
        if settings.get("divisions"):
            st.caption("Your league has divisions. These odds ignore division-winner spots.")
        c1, c2 = st.columns([3, 2])
        with c1:
            pct = st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent")
            st.dataframe(odds, hide_index=True, width="stretch",
                         column_config={"Make playoffs": pct, "Top-2 seed": pct, "#1 seed": pct, "Last place": pct,
                                        "Current W": st.column_config.NumberColumn(format="%.1f"),
                                        "Projected W": st.column_config.NumberColumn(format="%.1f")})
        with c2:
            o = odds.sort_values("Make playoffs")
            fig = px.bar(o, x="Make playoffs", y="Team", orientation="h", color="Team", color_discrete_map=cmap,
                         text=o["Make playoffs"].map("{:.0%}".format))
            fig.update_layout(title="Playoff odds", showlegend=False, xaxis=dict(tickformat=".0%", range=[0, 1.05]))
            st.plotly_chart(style_fig(fig, 60 + 36 * n_teams), width="stretch")

# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------
with tabs[8]:
    st.caption("Head-to-head records count regular-season games across every season of this league on Sleeper.")
    names, h2h, alltime, titles = {}, defaultdict(lambda: [0, 0, 0]), defaultdict(lambda: [0, 0, 0, 0.0]), defaultdict(int)
    for i, c in enumerate(chain):
        d = load_current_season(c["league_id"]) if i == 0 else load_past_season(c["league_id"])
        for u in d["users"]:
            names.setdefault(u["user_id"], u.get("display_name"))
        owner = {r["roster_id"]: r.get("owner_id") for r in d["rosters"]}
        reg, fin, _ = season_weeks(d["league"], state)
        for wk in fin:
            ms = d["matchups"].get(wk) or []
            pts = {m["roster_id"]: float(m.get("custom_points") or m.get("points") or 0) for m in ms}
            groups = defaultdict(list)
            for m in ms:
                if m.get("matchup_id") is not None:
                    groups[m["matchup_id"]].append(m["roster_id"])
            for pair in groups.values():
                if len(pair) != 2:
                    continue
                a, b = pair
                ua, ub = owner.get(a), owner.get(b)
                if not ua or not ub:
                    continue
                for x, y, ux, uy in [(a, b, ua, ub), (b, a, ub, ua)]:
                    res = 0 if pts[x] > pts[y] else 1 if pts[x] < pts[y] else 2
                    h2h[(ux, uy)][res] += 1
                    alltime[ux][res] += 1
                    alltime[ux][3] += pts[x]
        champ = next((m.get("w") for m in d["bracket"] if m.get("p") == 1 and m.get("w")), None)
        if champ and owner.get(champ):
            titles[owner[champ]] += 1

    if not alltime:
        st.write("No completed games yet.")
    else:
        at = pd.DataFrame([{"Manager": names.get(u, u), "W": v[0], "L": v[1], "T": v[2],
                            "Win %": (v[0] + 0.5 * v[2]) / max(1, sum(v[:3])), "PF": v[3],
                            "Titles": titles.get(u, 0)} for u, v in alltime.items()])
        at = at.sort_values(["Titles", "Win %"], ascending=False)
        if not at["T"].any():
            at = at.drop(columns="T")
        st.markdown(f"**All-time standings** ({len(chain)} season{'s' if len(chain) > 1 else ''})")
        st.dataframe(at, hide_index=True, width="stretch",
                     column_config={"Win %": st.column_config.NumberColumn(format="%.3f"),
                                    "PF": st.column_config.NumberColumn(format="%.0f"),
                                    "Titles": st.column_config.NumberColumn(format="%d 🏆")})

        mgrs = at["Manager"].tolist()
        uid = {names.get(u, u): u for u in alltime}
        pct = pd.DataFrame(np.nan, index=mgrs, columns=mgrs)
        txt = pd.DataFrame("", index=mgrs, columns=mgrs)
        for a in mgrs:
            for b in mgrs:
                w, l, t = h2h.get((uid[a], uid[b]), [0, 0, 0])
                if w + l + t:
                    pct.loc[a, b] = (w + 0.5 * t) / (w + l + t)
                    txt.loc[a, b] = f"{w}-{l}" + (f"-{t}" if t else "")
        fig = go.Figure(go.Heatmap(z=pct.values, x=mgrs, y=mgrs, text=txt.values, texttemplate="%{text}",
                                   colorscale="RdYlGn", zmin=0, zmax=1, showscale=False,
                                   hovertemplate="%{y} vs %{x}: %{text}<extra></extra>"))
        fig.update_layout(title="Head-to-head (row manager's record against column manager)")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(style_fig(fig, 80 + 42 * len(mgrs)), width="stretch")
