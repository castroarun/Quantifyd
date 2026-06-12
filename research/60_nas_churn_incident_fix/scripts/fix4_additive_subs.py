"""FIX #4 (2026-06-12): additive option-leg subscriptions. Same-strike legs across
variants share ONE instrument token, so a variant re-subscribe must never unsubscribe
a token a sibling still needs. The old exclusion sets were triangular (OTM excluded
nothing; ATM excluded OTM; ATM2 excluded OTM+ATM; only ATM4 excluded all), so an
earlier variant could drop a later variant's live leg -> tick gap -> SL skip (#1).
Replace every exclusion with one helper that excludes ALL sibling variant maps."""
import ast, shutil
P = '/home/arun/quantifyd/services/nas_ticker.py'
s = open(P, encoding='utf-8').read()
if '_tokens_in_use_by_others' in s:
    print('ALREADY PATCHED'); raise SystemExit

HELPER = '''    def _tokens_in_use_by_others(self, own_map):
        """Union of instrument tokens referenced by every variant leg-map EXCEPT
        own_map. Same-strike legs across variants share ONE instrument token, so a
        variant re-subscribe must never unsubscribe a token a sibling still needs
        (fix 2026-06-12). Caller holds self._lock."""
        in_use = set()
        for m in (self._option_tokens, self._atm_option_tokens,
                  self._atm2_option_tokens, self._atm4_option_tokens):
            if m is own_map:
                continue
            in_use |= set(m.keys())
        return in_use

    def subscribe_option_legs(self, positions: List[dict]):'''

edits = [
    # insert helper just before subscribe_option_legs
    ("    def subscribe_option_legs(self, positions: List[dict]):", HELPER),

    # --- OTM ---
    ("            # Unsubscribe tokens no longer needed\n            tokens_to_remove = old_tokens - set(new_tokens)\n",
     "            # Unsubscribe tokens no longer needed (keep any token a sibling variant still uses)\n            tokens_to_remove = old_tokens - set(new_tokens) - self._tokens_in_use_by_others(self._option_tokens)\n"),

    # --- ATM ---
    ("                    # Only unsubscribe tokens not used by OTM\n                    tokens_to_unsub = old_tokens - set(self._option_tokens.keys())",
     "                    # Only unsubscribe tokens not used by any sibling variant (shared strikes)\n                    tokens_to_unsub = old_tokens - self._tokens_in_use_by_others(self._atm_option_tokens)"),
    ("            # Subscribe new tokens (avoid duplicating OTM subscriptions)\n            all_existing = set(self._option_tokens.keys()) | old_tokens",
     "            # Subscribe new tokens (avoid duplicating any sibling variant's subscriptions)\n            all_existing = self._tokens_in_use_by_others(self._atm_option_tokens) | old_tokens"),
    ("            # Unsubscribe old ATM tokens no longer needed (and not used by OTM)\n            tokens_to_remove = old_tokens - set(new_tokens) - set(self._option_tokens.keys())",
     "            # Unsubscribe old ATM tokens no longer needed (and not used by any sibling variant)\n            tokens_to_remove = old_tokens - set(new_tokens) - self._tokens_in_use_by_others(self._atm_option_tokens)"),

    # --- ATM2 ---
    ("                    # Only unsubscribe tokens not used by OTM or ATM\n                    tokens_to_unsub = old_tokens - set(self._option_tokens.keys()) - set(self._atm_option_tokens.keys())",
     "                    # Only unsubscribe tokens not used by any sibling variant (shared strikes)\n                    tokens_to_unsub = old_tokens - self._tokens_in_use_by_others(self._atm2_option_tokens)"),
    ("            all_existing = set(self._option_tokens.keys()) | set(self._atm_option_tokens.keys()) | old_tokens",
     "            all_existing = self._tokens_in_use_by_others(self._atm2_option_tokens) | old_tokens"),
    ("            # Unsubscribe old ATM2 tokens no longer needed (and not used by OTM/ATM)\n            tokens_to_remove = old_tokens - set(new_tokens) - set(self._option_tokens.keys()) - set(self._atm_option_tokens.keys())",
     "            # Unsubscribe old ATM2 tokens no longer needed (and not used by any sibling variant)\n            tokens_to_remove = old_tokens - set(new_tokens) - self._tokens_in_use_by_others(self._atm2_option_tokens)"),

    # --- ATM4 (already complete, convert to helper for symmetry) ---
    ("                    tokens_to_unsub = old_tokens - set(self._option_tokens.keys()) - set(self._atm_option_tokens.keys()) - set(self._atm2_option_tokens.keys())",
     "                    tokens_to_unsub = old_tokens - self._tokens_in_use_by_others(self._atm4_option_tokens)"),
    ("            all_existing = set(self._option_tokens.keys()) | set(self._atm_option_tokens.keys()) | set(self._atm2_option_tokens.keys()) | old_tokens",
     "            all_existing = self._tokens_in_use_by_others(self._atm4_option_tokens) | old_tokens"),
    ("            tokens_to_remove = old_tokens - set(new_tokens) - set(self._option_tokens.keys()) - set(self._atm_option_tokens.keys()) - set(self._atm2_option_tokens.keys())",
     "            tokens_to_remove = old_tokens - set(new_tokens) - self._tokens_in_use_by_others(self._atm4_option_tokens)"),
]

for i, (old, new) in enumerate(edits, 1):
    assert s.count(old) == 1, 'edit %d count=%d' % (i, s.count(old))
    s = s.replace(old, new, 1)
ast.parse(s)
shutil.copy(P, P + '.bak_fix4')
open(P, 'w', encoding='utf-8').write(s)
print('PATCHED #4: subscriptions now additive across all variants (helper-based)')
