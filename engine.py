"""
engine.py - Resilience Rewired (Tangerine track)

Official definitions (from the track brief, using monthly_summary.csv):
    I = median(money_in)
    F = median(household_bills + insurance + loan_payments)   (display only)
    S = median(money_out)
    B = mean(avg_daily_balance) over the account's last 6 rows sorted by month
    H = median(household_bills)
    P = loans.monthly_payment if the loan is still running at end of 2025 else 0

    buffer_left = B - one_off_costs
    gap         = spending_after_shock - income_after_shock
    buffer_left <= 0 -> runway 0 ; gap <= 0 -> 120 ("no shortfall") ; else min(buffer_left/gap, 120)
    rounded to 1 decimal.

Quick start:
    ms, loans, accounts, districts = load_data("data")
    profiles = build_profiles(ms, loans)
    p = get_profile(profiles, 1234)
    r = apply_shock(p, PRESETS["EI job loss (55%, 6 mo)"])
    print(r["runway_shock"], goal_plan(p), backup_plan(p, PRESETS["EI job loss (55%, 6 mo)"]))
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

RUNWAY_CAP = 120.0
END_OF_DATA = pd.Period("2025-12", freq="M")

# Spending categories a customer could plausibly trim (from monthly_summary.csv).
FLEX_COLS = ["card_withdrawals", "cash_withdrawals_genuine", "transfers_out_other", "bank_fees"]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_data(data_dir: str = "data"):
    """Load the small files. (transactions.csv is NOT needed for the core engine.)"""
    d = Path(data_dir)
    ms = pd.read_csv(d / "monthly_summary.csv")
    loans = pd.read_csv(d / "loans.csv", parse_dates=["loan_start"])
    accounts = pd.read_csv(d / "accounts.csv")
    districts = pd.read_csv(d / "districts.csv")
    return ms, loans, accounts, districts


def load_transactions(data_dir: str = "data", usecols=None) -> pd.DataFrame:
    """For optional extra insight only. partner_bank must be read as text."""
    return pd.read_csv(
        Path(data_dir) / "transactions.csv",
        usecols=usecols,
        dtype={"partner_bank": str, "partner_account": str},
        keep_default_na=False,
        parse_dates=["date"] if (usecols is None or "date" in usecols) else None,
    )


# --------------------------------------------------------------------------- #
# Profiles (vectorised for all accounts; every live lookup is then instant)
# --------------------------------------------------------------------------- #
def loan_payment_table(loans: pd.DataFrame, rule: str = "status") -> pd.DataFrame:
    """
    P per account. Two ways to decide "still running at end of 2025":
      rule="status": status in {C, D}               (bank's own label)   <- ASSUMPTION default
      rule="date"  : loan_start + duration > 2025-12
    Both are always computed so mismatches can be inspected / explained.
    """
    l = loans.copy()
    l["loan_start"] = pd.to_datetime(l["loan_start"])
    end_period = l["loan_start"].dt.to_period("M") + l["duration_months"].astype(int)
    l["running_by_date"] = end_period > END_OF_DATA
    l["running_by_status"] = l["status"].isin(["C", "D"])
    l["loan_rule_mismatch"] = l["running_by_date"] != l["running_by_status"]
    running = l["running_by_status"] if rule == "status" else l["running_by_date"]
    l["P"] = np.where(running, l["monthly_payment"], 0.0)
    return l[["account_id", "P", "running_by_date", "running_by_status",
              "loan_rule_mismatch", "defaulted", "status", "amount",
              "duration_months", "monthly_payment"]].set_index("account_id")


def build_profiles(ms: pd.DataFrame, loans: Optional[pd.DataFrame] = None,
                   loan_rule: str = "status") -> pd.DataFrame:
    """One row per account with I, F, S, B, H, P, base runway, plus extras for ML/plans."""
    ms = ms.sort_values(["account_id", "month"]).copy()
    ms["fixed"] = ms["household_bills"] + ms["insurance"] + ms["loan_payments"]
    g = ms.groupby("account_id")

    prof = pd.DataFrame({
        "I": g["money_in"].median(),
        "F": g["fixed"].median(),
        "S": g["money_out"].median(),
        "H": g["household_bills"].median(),
        "n_months": g.size(),
    })
    # Buffer: mean of avg_daily_balance over the last 6 rows (fewer if the account is shorter).
    prof["B"] = ms.groupby("account_id").tail(6).groupby("account_id")["avg_daily_balance"].mean()

    # Flexible-category medians (for plans) and volatility features (for discovery / ML).
    for c in FLEX_COLS:
        prof[f"flex_{c}"] = g[c].median()
    prof["flex_total"] = prof[[f"flex_{c}" for c in FLEX_COLS]].sum(axis=1)
    prof["income_cv"] = (g["money_in"].std(ddof=0) / g["money_in"].mean().replace(0, np.nan)).fillna(0.0)
    prof["spend_cv"] = (g["money_out"].std(ddof=0) / g["money_out"].mean().replace(0, np.nan)).fillna(0.0)
    prof["pct_months_below_zero"] = g["days_below_zero"].apply(lambda s: (s > 0).mean())
    prof["avg_days_below_zero"] = g["days_below_zero"].mean()
    prof["min_balance_ever"] = g["min_balance"].min()

    # Loans
    if loans is not None:
        lt = loan_payment_table(loans, loan_rule)
        prof = prof.join(lt[["P", "running_by_date", "running_by_status", "loan_rule_mismatch",
                             "defaulted", "status"]], how="left")
        prof["P"] = prof["P"].fillna(0.0)
        prof["has_loan"] = prof["status"].notna()
    else:
        prof["P"] = 0.0
        prof["has_loan"] = False

    # Base runway (no shock): official formula with gap = S - I, no one-off.
    prof["runway_base"] = [runway(b, 0.0, s, i) for b, s, i in zip(prof["B"], prof["S"], prof["I"])]
    prof["surplus"] = prof["I"] - prof["S"]
    prof.index.name = "account_id"
    return prof


def get_profile(profiles: pd.DataFrame, account_id: int) -> dict:
    """Plain dict for one account (what the UI, shocks and plans consume)."""
    if account_id not in profiles.index:
        raise KeyError(f"account_id {account_id} not found in monthly_summary")
    row = profiles.loc[account_id]
    d = {k: (None if (isinstance(v, float) and math.isnan(v)) else v) for k, v in row.to_dict().items()}
    d["account_id"] = int(account_id)
    for k in ("n_months",):
        d[k] = int(d[k])
    return d


# --------------------------------------------------------------------------- #
# Official runway formula
# --------------------------------------------------------------------------- #
def runway(B: float, one_off: float, spending_after: float, income_after: float) -> float:
    """Official runway in months, rounded to 1 decimal. 120.0 means 'no shortfall (120+)'."""
    buffer_left = B - one_off
    gap = spending_after - income_after
    if buffer_left <= 0:
        return 0.0
    if gap <= 0:
        return RUNWAY_CAP
    return round(min(buffer_left / gap, RUNWAY_CAP), 1)


def runway_label(r: float) -> str:
    return "no shortfall (120+)" if r >= RUNWAY_CAP else f"{r:.1f} months"


# --------------------------------------------------------------------------- #
# Shocks
# --------------------------------------------------------------------------- #
@dataclass
class Shock:
    """
    All fields are neutral by default. A shock changes income and/or spending:
      income_after   = income_mult * I
      spending_after = S + bills_pct * H + spend_add - (P if loan_relief)
      one_off_total  = one_off + one_off_spend_mult * S
    `months` = how long the income/spending shock lasts (e.g. 6 for EI). None = permanent.
    NOTE: the official runway formula ignores `months`; it is used for the
    survives/run-out/projection outputs only.
    """
    income_mult: float = 1.0        # 0.55 = EI at 55%, 0 = no income
    bills_pct: float = 0.0          # 0.15 = household bills up 15% (applied to H)
    spend_add: float = 0.0          # extra $/month of spending
    one_off: float = 0.0            # one-off cost in dollars
    one_off_spend_mult: float = 0.0  # one-off as multiple of typical monthly spending S (furnace = 1.0)
    loan_relief: bool = False       # pause the loan payment P
    months: Optional[int] = None    # duration of the shock; None = permanent
    name: str = "Custom shock"

    @classmethod
    def from_dict(cls, d: dict) -> "Shock":
        """Build from Gemini's parsed JSON. Unknown keys are ignored; values are sanity-clamped."""
        allowed = {f for f in cls.__dataclass_fields__}
        s = cls(**{k: v for k, v in d.items() if k in allowed and v is not None})
        s.income_mult = max(0.0, float(s.income_mult))
        s.months = int(s.months) if s.months else None
        return s

    def to_dict(self) -> dict:
        return asdict(self)


PRESETS = {
    "EI job loss (55%, 6 mo)": Shock(income_mult=0.55, months=6, name="EI job loss (55%, 6 mo)"),
    "Furnace dies (1 month of spending)": Shock(one_off_spend_mult=1.0, name="Furnace dies (1 month of spending)"),
    "Job loss + bills +15%": Shock(income_mult=0.55, bills_pct=0.15, months=6, name="Job loss + bills +15%"),
    "Rent/bills +15%": Shock(bills_pct=0.15, name="Rent/bills +15%"),
}


def apply_shock(p: dict, shock: Shock | None = None) -> dict:
    """Recalculate runway under a shock. Returns every intermediate number so the UI can show its work."""
    shock = shock or Shock()
    I, S, H, B, P = p["I"], p["S"], p["H"], p["B"], p["P"]

    income_after = shock.income_mult * I
    spending_after = S + shock.bills_pct * H + shock.spend_add
    if shock.loan_relief:
        spending_after -= P
    one_off_total = shock.one_off + shock.one_off_spend_mult * S
    buffer_left = B - one_off_total
    gap = spending_after - income_after
    r_shock = runway(B, one_off_total, spending_after, income_after)

    months = shock.months
    out = {
        "shock": shock.to_dict(),
        "I": I, "S": S, "H": H, "B": B, "P": P,
        "income_after": income_after,
        "spending_after": spending_after,
        "one_off_total": one_off_total,
        "buffer_left": buffer_left,
        "gap": gap,
        "runway_base": p.get("runway_base", runway(B, 0, S, I)),
        "runway_shock": r_shock,
        "runway_label": runway_label(r_shock),
        "months": months,
    }
    # Does the customer make it through the shock period?
    if months:
        out["survives_shock_period"] = bool(r_shock >= months)
        out["runout_month"] = None if r_shock >= months else (int(math.floor(r_shock)) + 1)
    else:
        out["survives_shock_period"] = bool(r_shock >= RUNWAY_CAP)
        out["runout_month"] = None if r_shock >= RUNWAY_CAP else int(math.floor(r_shock)) + 1
    out["shortfall_over_period"] = (
        max(0.0, gap * months - buffer_left) if months and gap > 0 else 0.0
    )
    return out


def project_balance(p: dict, shock: Shock | None = None, horizon: int = 12) -> pd.DataFrame:
    """
    Month-by-month balance. Month 0 = B - one_off. During the shock (first `months`
    months, or the whole horizon if permanent) flow = income_after - spending_after;
    afterwards flow reverts to the customer's normal I - S.
    The balance is allowed to go negative so the UI can show where it crosses zero.
    """
    shock = shock or Shock()
    r = apply_shock(p, shock)
    shock_len = shock.months or horizon
    bal = r["buffer_left"]
    rows = [{"month": 0, "balance": bal, "in_shock": False}]
    for t in range(1, horizon + 1):
        in_shock = t <= shock_len
        flow = (r["income_after"] - r["spending_after"]) if in_shock else (p["I"] - p["S"])
        bal += flow
        rows.append({"month": t, "balance": bal, "in_shock": in_shock})
    df = pd.DataFrame(rows)
    df["baseline"] = p["B"] + df["month"] * (p["I"] - p["S"])
    return df


# --------------------------------------------------------------------------- #
# Plans
# --------------------------------------------------------------------------- #
def goal_plan(p: dict, target_months: int = 3, horizon_months: int = 12) -> dict:
    """
    Emergency-fund plan: target = target_months * S.
    If surplus (I - S) > 0 -> time to goal at current surplus + suggested savings rate.
    If surplus <= 0 or too slow -> which of the customer's own flexible categories to trim.
    """
    I, S, B = p["I"], p["S"], p["B"]
    target = target_months * S
    shortfall = max(0.0, target - B)
    surplus = I - S
    plan = {
        "target_months": target_months, "target_fund": target, "current_buffer": B,
        "gap_to_target": shortfall, "monthly_surplus": surplus,
        "already_there": shortfall == 0,
        "savings_rate_now": (surplus / I) if I > 0 else None,
    }
    if shortfall == 0:
        plan.update(months_to_goal=0, required_monthly_saving=0.0, required_cut=0.0, feasible=True,
                    trim_suggestions=[], note="Buffer already covers the target.")
        return plan

    plan["months_to_goal_at_current_surplus"] = (shortfall / surplus) if surplus > 0 else None
    req_saving = shortfall / horizon_months          # needed to hit the goal within the horizon
    req_cut = max(0.0, req_saving - surplus)         # extra $/month that must come from spending
    plan.update(horizon_months=horizon_months, required_monthly_saving=req_saving,
                required_cut=req_cut,
                required_saving_rate=(req_saving / I) if I > 0 else None)

    flex = {c: max(0.0, p.get(f"flex_{c}") or 0.0) for c in FLEX_COLS}
    plan["trim_suggestions"] = _trim_suggestions(flex, req_cut)
    pool = sum(flex.values())
    plan["flex_pool"] = pool
    plan["feasible"] = bool(req_cut <= pool)
    plan["note"] = ("Reachable by trimming flexible spending." if plan["feasible"] and req_cut > 0
                    else "Reachable from current surplus." if req_cut == 0
                    else "Flexible spending alone can't close the gap; extend the horizon or raise income.")
    return plan


def _trim_suggestions(flex: dict, needed: float) -> list[dict]:
    """Spread a needed monthly cut across flexible categories, proportional to size (capped at 50% each)."""
    if needed <= 0:
        return []
    total = sum(flex.values())
    if total <= 0:
        return []
    out, remaining = [], needed
    for cat, amt in sorted(flex.items(), key=lambda kv: -kv[1]):
        cut = min(amt * 0.5, needed * amt / total)
        cut = min(cut, remaining)
        if cut > 0.5:
            out.append({"category": cat, "typical_monthly": amt, "suggested_cut": round(cut, 2)})
            remaining -= cut
    return out


def backup_plan(p: dict, shock: Shock, target_months: Optional[int] = None) -> dict:
    """
    Plan for when the shock hits. Duration defaults to shock.months, else 6.
      sustainable_gap = buffer_left / duration   (max monthly deficit the buffer can fund)
      required_cut    = gap - sustainable_gap    (monthly cut needed to last the full duration)
    Then lists options: own flexible spending, loan relief, household-bill cut, bridge amount.
    """
    r = apply_shock(p, shock)
    duration = target_months or shock.months or 6
    buffer_left, gap = r["buffer_left"], r["gap"]

    plan = {"shock_result": r, "duration_months": duration}
    if gap <= 0 and buffer_left > 0:
        plan.update(needs_action=False, required_cut=0.0, options=[],
                    summary="Income still covers spending under this shock; no cuts needed.")
        return plan

    sustainable = max(0.0, buffer_left) / duration
    required_cut = max(0.0, gap - sustainable)
    flex = {c: max(0.0, p.get(f"flex_{c}") or 0.0) for c in FLEX_COLS}
    flex_pool = sum(flex.values()) * 0.5            # assume at most half of flexible spend is cuttable
    loan_relief_amt = 0.0 if shock.loan_relief else p["P"]
    bills_cut = 0.10 * p["H"]                       # ASSUMPTION: a 10% household-bill reduction is realistic

    options = [
        {"option": "Trim own flexible spending (up to 50%)", "monthly_saving": round(flex_pool, 2),
         "detail": _trim_suggestions(flex, required_cut)},
        {"option": "Loan payment relief / deferral", "monthly_saving": round(loan_relief_amt, 2)},
        {"option": "Cut household bills ~10%", "monthly_saving": round(bills_cut, 2)},
    ]
    total_available = flex_pool + loan_relief_amt + bills_cut
    remaining_need = max(0.0, required_cut - total_available)
    plan.update(
        needs_action=True,
        monthly_gap=gap, sustainable_gap=sustainable, required_cut=required_cut,
        options=options, total_cuts_available=total_available,
        covered_by_cuts=bool(required_cut <= total_available),
        bridge_needed_total=round(remaining_need * duration, 2),   # $ still missing over the period
        runout_month_without_action=r["runout_month"],
    )
    plan["summary"] = (
        f"Without action the buffer lasts {r['runway_label']}"
        + (f" and runs out in month {r['runout_month']}" if r["runout_month"] else "")
        + f". To last {duration} months, cut ${required_cut:,.0f}/month"
        + (" (achievable with the options listed)." if plan["covered_by_cuts"]
           else f"; even all options leave about ${plan['bridge_needed_total']:,.0f} to bridge.")
    )
    return plan


# --------------------------------------------------------------------------- #
# Population helpers (for the discovery tab / Colab)
# --------------------------------------------------------------------------- #
def population_runways(profiles: pd.DataFrame, shock: Shock) -> pd.Series:
    """Runway under a shock for every account, using the same official formula."""
    one_off = shock.one_off + shock.one_off_spend_mult * profiles["S"]
    s_after = profiles["S"] + shock.bills_pct * profiles["H"] + shock.spend_add
    if shock.loan_relief:
        s_after = s_after - profiles["P"]
    i_after = shock.income_mult * profiles["I"]
    return pd.Series(
        [runway(b, o, s, i) for b, o, s, i in zip(profiles["B"], one_off, s_after, i_after)],
        index=profiles.index, name=f"runway|{shock.name}",
    )


def fragility_table(profiles: pd.DataFrame, presets: dict | None = None, threshold: float = 3.0) -> pd.DataFrame:
    """Share of accounts with runway below `threshold` months under each preset shock."""
    presets = presets or PRESETS
    rows = [{"scenario": "No shock", "pct_below": (profiles["runway_base"] < threshold).mean() * 100}]
    for name, sh in presets.items():
        rows.append({"scenario": name, "pct_below": (population_runways(profiles, sh) < threshold).mean() * 100})
    return pd.DataFrame(rows).rename(columns={"pct_below": f"% with runway < {threshold:g} mo"})


# --------------------------------------------------------------------------- #
# Self-test with synthetic data:  python engine.py
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Hand-checkable case: I=1000, S=900, H=300, B constant 2000 -> official example twists.
    months = pd.period_range("2025-01", "2025-12", freq="M").astype(str)
    ms = pd.DataFrame({
        "account_id": 1, "month": months, "money_in": 1000.0, "money_out": 900.0,
        "household_bills": 300.0, "insurance": 50.0, "loan_payments": 100.0,
        "card_withdrawals": 200.0, "cash_withdrawals_genuine": 100.0,
        "transfers_out_other": 100.0, "bank_fees": 10.0, "avg_daily_balance": 2000.0,
        "min_balance": 1500.0, "days_below_zero": 0,
    })
    loans = pd.DataFrame({"loan_id": [1], "account_id": [1], "loan_start": pd.to_datetime(["2025-06-01"]),
                          "amount": [2400], "duration_months": [24], "monthly_payment": [100],
                          "status": ["C"], "defaulted": [0]})
    prof = build_profiles(ms, loans)
    p = get_profile(prof, 1)
    assert (p["I"], p["S"], p["H"], p["B"], p["P"], p["F"]) == (1000, 900, 300, 2000, 100, 450)
    assert p["runway_base"] == 120.0                                   # I > S -> no shortfall

    ei = apply_shock(p, PRESETS["EI job loss (55%, 6 mo)"])
    assert ei["runway_shock"] == round(2000 / (900 - 550), 1) == 5.7   # gap=350
    assert ei["survives_shock_period"] is False and ei["runout_month"] == 6

    furnace = apply_shock(p, PRESETS["Furnace dies (1 month of spending)"])
    assert furnace["runway_shock"] == 120.0                            # buffer_left=1100, gap=-100

    combo = apply_shock(p, PRESETS["Job loss + bills +15%"])
    assert combo["runway_shock"] == round(2000 / (945 - 550), 1) == 5.1  # spending=900+45

    assert runway(-5, 0, 900, 1000) == 0.0 and runway(100, 200, 900, 500) == 0.0
    print("backup:", backup_plan(p, PRESETS["EI job loss (55%, 6 mo)"])["summary"])
    print("goal  :", goal_plan(p, 6)["note"], "| target", goal_plan(p, 6)["target_fund"])
    print("projection tail:\n", project_balance(p, PRESETS["EI job loss (55%, 6 mo)"], 8).round(0).to_string(index=False))
    print("fragility:\n", fragility_table(prof).to_string(index=False))
    print("All self-tests passed.")
