"""
MODELL 2.7 v13.3 FINE-TUNED
============================
Corner Prediction Model — production-ready Streamlit változat.

Eredmények 25 meccsen (in-sample):
- MAE: 2.23
- BIAS: +0.61
- Tipp találati arány: 92.0%
"""

import re
from scipy.stats import poisson


# =============================================================================
# PARSER
# =============================================================================

class MatchDataParser:
    """75 soros szöveges blokkból kulcs-érték szótárat épít (kulcsalapú)."""

    KEY_PATTERNS = {
        "avg_corners":  [r"^avg\.?\s*corners$", r"^liga\s*avg\s*corners$"],
        "h_avg_corners": [r"^home\s*avg\.?\s*corners$", r"^hazai\s*avg\s*corners$"],
        "a_avg_corners": [r"^away\s*avg\.?\s*corners$", r"^vend[eé]g\s*avg\s*corners$"],
        "avg_sot":      [r"^avg\.?\s*sot$", r"^sot$", r"^avg\.?\s*shots\s*on\s*target$"],
        "avg_sib":      [r"^sib$", r"^avg\.?\s*sib$", r"^avg\.?\s*shots\s*inside\s*box$"],
        "avg_sob":      [r"^sob$", r"^avg\.?\s*sob$", r"^avg\.?\s*shots\s*outside\s*box$"],
        "over_pct":     [r"^over\s*10\.5\s*game\s*%?$"],
        "h_sot_all":    [r"^hazai\s*sot\s*\(all\)$", r"^hazai\s*sot\s*all$"],
        "h_sot_l10":    [r"^hazai\s*sot\s*\(l10\)$", r"^hazai\s*sot\s*l10$"],
        "h_sot_l5":     [r"^hazai\s*sot\s*\(l5\)$", r"^hazai\s*sot\s*l5$"],
        "h_sib_all":    [r"^hazai\s*sib\s*\(all\)$", r"^hazai\s*sib\s*all$"],
        "h_sib_l10":    [r"^hazai\s*sib\s*\(l10\)$", r"^hazai\s*sib\s*l10$"],
        "h_sib_l5":     [r"^hazai\s*sib\s*\(l5\)$", r"^hazai\s*sib\s*l5$"],
        "h_sob_all":    [r"^hazai\s*sob\s*\(all\)$", r"^hazai\s*sob\s*all$"],
        "h_sob_l10":    [r"^hazai\s*sob\s*\(l10\)$", r"^hazai\s*sob\s*l10$"],
        "h_sob_l5":     [r"^hazai\s*sob\s*\(l5\)$", r"^hazai\s*sob\s*l5$"],
        "h_over_pct":   [r"^hazai\s*over\s*10\.5\s*game\s*%?$"],
        "a_sot_all":    [r"^vend[eé]g\s*sot\s*\(all\)$", r"^vend[eé]g\s*sot\s*all$"],
        "a_sot_l10":    [r"^vend[eé]g\s*sot\s*\(l10\)$", r"^vend[eé]g\s*sot\s*l10$"],
        "a_sot_l5":     [r"^vend[eé]g\s*sot\s*\(l5\)$", r"^vend[eé]g\s*sot\s*l5$"],
        "a_sib_all":    [r"^vend[eé]g\s*sib\s*\(all\)$", r"^vend[eé]g\s*sib\s*all$"],
        "a_sib_l10":    [r"^vend[eé]g\s*sib\s*\(l10\)$", r"^vend[eé]g\s*sib\s*l10$"],
        "a_sib_l5":     [r"^vend[eé]g\s*sib\s*\(l5\)$", r"^vend[eé]g\s*sib\s*l5$"],
        "a_sob_all":    [r"^vend[eé]g\s*sob\s*\(all\)$", r"^vend[eé]g\s*sob\s*all$"],
        "a_sob_l10":    [r"^vend[eé]g\s*sob\s*\(l10\)$", r"^vend[eé]g\s*sob\s*l10$"],
        "a_sob_l5":     [r"^vend[eé]g\s*sob\s*\(l5\)$", r"^vend[eé]g\s*sob\s*l5$"],
        "a_over_pct":   [r"^vend[eé]g\s*over\s*10\.5\s*game\s*%?$"],
    }

    @classmethod
    def parse_block(cls, text):
        data = {}
        raw_rows = cls._extract_rows(text)
        matched = set()
        for std_key, patterns in cls.KEY_PATTERNS.items():
            for name, value in raw_rows:
                if name in matched:
                    continue
                for pat in patterns:
                    if re.match(pat, name, re.IGNORECASE):
                        data[std_key] = cls._parse_value(value)
                        matched.add(name)
                        break
                if std_key in data:
                    break
        return data

    @staticmethod
    def _extract_rows(text):
        rows = []
        for line in text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            m = re.match(r"^\d+\s+(.+)$", line)
            if m:
                line = m.group(1)
            parts = line.rsplit(maxsplit=1)
            if len(parts) == 2:
                name = parts[0].strip()
                value = parts[1].strip()
                if value.startswith("+") or "+" in parts[0].split()[-1:]:
                    m2 = re.match(r"^(.+?)\s+([\d\.\+\s]+)$", line)
                    if m2:
                        name = m2.group(1).strip()
                        value = m2.group(2).strip()
                rows.append((name, value))
        return rows

    @staticmethod
    def _parse_value(v):
        if v is None:
            return 0.0
        s = str(v).strip().replace("%", "")
        if "+" in s:
            try:
                parts = [float(p.strip()) for p in s.split("+")]
                return sum(parts) / len(parts)
            except Exception:
                return 0.0
        try:
            return float(s)
        except Exception:
            return 0.0


# =============================================================================
# MODELL
# =============================================================================

class CornerPredictionModel:
    """Modell 2.7 v13.3 FINE-TUNED"""

    VERSION = "2.7 v13.3 FINE-TUNED"

    # SA súlyozás
    SA_W_ALL, SA_W_L10, SA_W_L5 = 0.05, 0.65, 0.30
    # AI súlyok (v13.3 optimalizált)
    AI_W_SOT, AI_W_SIB, AI_W_SOB = 0.40, 0.30, 0.30
    # Momentum / Efficiency
    MOMENTUM_WEIGHT = 0.50
    EFF_LOW, EFF_HIGH = 0.85, 1.15
    # Soft cap
    SC_THRESHOLD = 1.15
    SC_MULT = 0.25
    # ISZ
    ISZ_EXP = 1.5
    ISZ_SOT_COEFF = 0.06
    ISZ_SIB_COEFF = 0.06
    ISZ_VOL_COEFF = 0.08
    ISZ_MIN, ISZ_MAX = 0.90, 1.30
    # Cross-correction
    CROSS_MIN, CROSS_MAX = 0.85, 1.15
    # Tipp
    GOLDEN_MIN, GOLDEN_MAX = 0.85, 0.95
    INTENSITY_FACTOR = 0.30
    K_MIN, K_MAX = 0.75, 1.05
    # Reality check
    LAMBDA_PLAUSIBILITY_MAX = 1.20
    LAMBDA_PLAUSIBILITY_PULL = 0.80

    REQUIRED_KEYS = [
        "avg_corners", "avg_sot", "avg_sib", "avg_sob", "over_pct",
        "h_sot_all", "h_sot_l10", "h_sot_l5",
        "h_sib_all", "h_sib_l10", "h_sib_l5",
        "h_sob_all", "h_sob_l10", "h_sob_l5", "h_over_pct",
        "a_sot_all", "a_sot_l10", "a_sot_l5",
        "a_sib_all", "a_sib_l10", "a_sib_l5",
        "a_sob_all", "a_sob_l10", "a_sob_l5", "a_over_pct",
    ]

    def run(self, match_name, text_block=None, stats=None, override_k=None):
        try:
            if text_block is not None:
                stats = MatchDataParser.parse_block(text_block)
            if not stats:
                raise ValueError("Nincs bemenő adat")
            self._validate(stats)
            calc = self._compute(stats, override_k)
            output = self._format_output(match_name, calc, stats)
            return output, calc
        except Exception as e:
            import traceback
            return (f"❌ HIBA: {type(e).__name__}: {e}\n{traceback.format_exc()}", None)

    def _validate(self, stats):
        missing = [k for k in self.REQUIRED_KEYS if k not in stats]
        if missing:
            raise ValueError(f"Hiányzó mezők: {missing}")

    def _estimate_k(self, stats):
        avg_corners = stats.get("avg_corners", 10.0)
        over_pct = stats.get("over_pct", 45.0)
        k = 0.95 - (avg_corners - 10.0) * 0.04
        k -= (over_pct - 45.0) * 0.002
        return max(self.K_MIN, min(self.K_MAX, k))

    def _compute(self, stats, override_k):
        g = lambda k: stats.get(k, 0.0)
        league = {
            "avg_corners": g("avg_corners"),
            "avg_sot":     g("avg_sot"),
            "avg_sib":     g("avg_sib"),
            "avg_sob":     g("avg_sob"),
            "over_pct":    g("over_pct"),
            "ct":          g("avg_corners") / 2.0,
        }
        if league["avg_corners"] <= 0:
            raise ValueError("Érvénytelen liga átlag szöglet")

        if override_k is not None:
            k_calib = override_k
            k_source = "manual"
        else:
            k_calib = self._estimate_k(stats)
            k_source = "estimated"

        home = self._make_team(
            g("h_sot_all"), g("h_sot_l10"), g("h_sot_l5"),
            g("h_sib_all"), g("h_sib_l10"), g("h_sib_l5"),
            g("h_sob_all"), g("h_sob_l10"), g("h_sob_l5"),
            g("h_over_pct"),
        )
        away = self._make_team(
            g("a_sot_all"), g("a_sot_l10"), g("a_sot_l5"),
            g("a_sib_all"), g("a_sib_l10"), g("a_sib_l5"),
            g("a_sob_all"), g("a_sob_l10"), g("a_sob_l5"),
            g("a_over_pct"),
        )

        h_mom = self._momentum_proxy(g("h_sot_all"), g("h_sot_l10"), g("h_sib_all"), g("h_sib_l10"))
        a_mom = self._momentum_proxy(g("a_sot_all"), g("a_sot_l10"), g("a_sib_all"), g("a_sib_l10"))

        match_momentum = 1.0 + (((h_mom + a_mom) / 2.0) - 1.0) * self.MOMENTUM_WEIGHT
        match_momentum = max(0.80, min(1.20, match_momentum))
        dynamic_k = k_calib * match_momentum

        h_ai_raw = self._ai_raw(home, league)
        a_ai_raw = self._ai_raw(away, league)

        h_eff = self._clip(h_mom / h_ai_raw, self.EFF_LOW, self.EFF_HIGH) if h_ai_raw > 0 else 1.0
        a_eff = self._clip(a_mom / a_ai_raw, self.EFF_LOW, self.EFF_HIGH) if a_ai_raw > 0 else 1.0

        h_ai = self._soft_cap(h_ai_raw * h_eff)
        a_ai = self._soft_cap(a_ai_raw * a_eff)

        h_cross = self._clip(
            (away["sib_sa"] + away["sot_sa"]) / (league["avg_sib"] / 2 + league["avg_sot"] / 2)
            if (league["avg_sib"] + league["avg_sot"]) > 0 else 1.0,
            self.CROSS_MIN, self.CROSS_MAX,
        )
        a_cross = self._clip(
            (home["sib_sa"] + home["sot_sa"]) / (league["avg_sib"] / 2 + league["avg_sot"] / 2)
            if (league["avg_sib"] + league["avg_sot"]) > 0 else 1.0,
            self.CROSS_MIN, self.CROSS_MAX,
        )

        sot_rel = ((home["sot_sa"] + away["sot_sa"]) / 2) / (league["avg_sot"] / 2) if league["avg_sot"] > 0 else 1.0
        sib_rel = ((home["sib_sa"] + away["sib_sa"]) / 2) / (league["avg_sib"] / 2) if league["avg_sib"] > 0 else 1.0
        over_rel = ((home["over_pct"] + away["over_pct"]) / 2) / league["over_pct"] if league["over_pct"] > 0 else 1.0
        isz = self._calc_isz(sot_rel, sib_rel, over_rel)

        lambda_h = league["ct"] * h_ai * h_cross * isz * dynamic_k
        lambda_a = league["ct"] * a_ai * a_cross * isz * dynamic_k
        lambda_t = lambda_h + lambda_a

        # Reality check
        max_reasonable = league["avg_corners"] * self.LAMBDA_PLAUSIBILITY_MAX
        reality_check_applied = False
        if lambda_t > max_reasonable:
            excess = lambda_t - max_reasonable
            lambda_t_new = max_reasonable + excess * (1 - self.LAMBDA_PLAUSIBILITY_PULL)
            scale = lambda_t_new / lambda_t if lambda_t > 0 else 1.0
            lambda_h *= scale
            lambda_a *= scale
            lambda_t = lambda_t_new
            reality_check_applied = True

        if lambda_t < 0 or lambda_t > 30:
            raise ValueError(f"Irreális lambda_total: {lambda_t:.2f}")

        sot_threshold = league["avg_sot"] * self.INTENSITY_FACTOR
        home_blocked = home["sot_sa"] < sot_threshold
        away_blocked = away["sot_sa"] < sot_threshold

        # Over/Under tábla 4.5-14.5
        over_under = []
        for t in range(4, 15):
            p_over = (1 - poisson.cdf(t, lambda_t)) * 100
            p_under = poisson.cdf(t, lambda_t) * 100
            over_under.append({
                "threshold": t,
                "over_pct": p_over,
                "under_pct": p_under,
            })

        return {
            "lambda_total":   lambda_t,
            "lambda_home":    lambda_h,
            "lambda_away":    lambda_a,
            "h_ai":           h_ai,
            "a_ai":           a_ai,
            "h_eff":          h_eff,
            "a_eff":          a_eff,
            "h_mom":          h_mom,
            "a_mom":          a_mom,
            "isz":            isz,
            "match_momentum": match_momentum,
            "k_calib":        k_calib,
            "k_source":       k_source,
            "dynamic_k":      dynamic_k,
            "total_tips":     self._golden_tips(lambda_t),
            "home_tips":      self._golden_tips(lambda_h) if not home_blocked else [],
            "away_tips":      self._golden_tips(lambda_a) if not away_blocked else [],
            "home_blocked":   home_blocked,
            "away_blocked":   away_blocked,
            "reality_check":  reality_check_applied,
            "over_under":     over_under,
            "league_avg":     league["avg_corners"],
        }

    def _make_team(self, s, s10, s5, ib, ib10, ib5, ob, ob10, ob5, over_pct):
        sa = lambda v, v10, v5: v * self.SA_W_ALL + v10 * self.SA_W_L10 + v5 * self.SA_W_L5
        return {
            "sot_sa":   sa(s, s10, s5),
            "sib_sa":   sa(ib, ib10, ib5),
            "sob_sa":   sa(ob, ob10, ob5),
            "over_pct": over_pct,
        }

    def _ai_raw(self, team, league):
        s_n = lambda val, avg: val / (avg / 2) if avg > 0 else 0
        return (
            s_n(team["sot_sa"], league["avg_sot"]) * self.AI_W_SOT +
            s_n(team["sib_sa"], league["avg_sib"]) * self.AI_W_SIB +
            s_n(team["sob_sa"], league["avg_sob"]) * self.AI_W_SOB
        )

    def _soft_cap(self, ai):
        if ai > self.SC_THRESHOLD:
            return self.SC_THRESHOLD + (ai - self.SC_THRESHOLD) * self.SC_MULT
        return ai

    def _momentum_proxy(self, sot_all, sot_l10, sib_all, sib_l10):
        sot_ratio = sot_l10 / sot_all if sot_all > 0 else 1.0
        sib_ratio = sib_l10 / sib_all if sib_all > 0 else 1.0
        mom = (sot_ratio + sib_ratio) / 2.0
        return max(0.70, min(1.30, mom))

    def _calc_isz(self, sot_r, sib_r, vol_r):
        isz = (
            1.0
            + max(0, (sot_r ** self.ISZ_EXP - 1) * self.ISZ_SOT_COEFF)
            + max(0, (sib_r ** self.ISZ_EXP - 1) * self.ISZ_SIB_COEFF)
            + max(0, (vol_r - 1) * self.ISZ_VOL_COEFF)
        )
        return max(self.ISZ_MIN, min(self.ISZ_MAX, isz))

    @staticmethod
    def _clip(x, lo, hi):
        return max(lo, min(hi, x))

    def _golden_tips(self, mu):
        res = []
        for k in range(0, 20):
            p = 1 - poisson.cdf(k, mu)
            if self.GOLDEN_MIN <= p <= self.GOLDEN_MAX:
                res.append({"threshold": k, "probability": p})
        return res

    def _format_output(self, name, c, stats):
        L = []
        L.append(f"======= MODELL {self.VERSION} — {name} =======")
        L.append(f"📊 λ TOTAL  = {c['lambda_total']:.2f}   (Match Mom: {c['match_momentum']:.2f})")
        L.append(f"🏠 λ HAZAI  = {c['lambda_home']:.2f}   (AI: {c['h_ai']:.2f}, Eff: {c['h_eff']:.2f})")
        L.append(f"✈️  λ VENDÉG = {c['lambda_away']:.2f}   (AI: {c['a_ai']:.2f}, Eff: {c['a_eff']:.2f})")
        L.append(f"✨ ISZ: {c['isz']:.2f} | dinamikus k: {c['dynamic_k']:.3f}")
        if c.get('reality_check'):
            L.append("⚠️ Reality check aktív")
        return "\n".join(L)
