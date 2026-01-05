import re
import random
import math
from math import comb

# =========================
# CONFIG
# =========================
MAX_N = 60
PICK = 6

# Quantos candidatos aleatórios testar (quanto maior, melhor a busca)
N_SAMPLES = 200_000

# Quantos "melhores" mostrar no final
TOP_K = 10

# Pesos do modelo (você pode ajustar)
W_ODD   = 1.0   # peso pares/ímpares
W_FAIXA = 1.0   # peso faixas
W_SOMA  = 0.7   # peso soma (aprox)
W_OVLP  = 0.8   # peso overlap com o último round

# Regras/limites opcionais (podem ser None pra não restringir)
MAX_OVERLAP_WITH_LAST = 2     # ex.: não deixar repetir mais que 2 dezenas do último round
FORBID_ONE_BUCKET = True      # proibir 6 dezenas na mesma faixa
FORBID_SEQ3 = False           # proibir sequência de 3 consecutivos (escadinha)

# =========================
# COLE AQUI SEUS ROUNDS (formato seguro com :)
# =========================
ROUNDS_TEXT = """
Round 1: 4,5,30,33,41,52
Round 2: 9,37,39,41,43,49
Round 3: 10,11,29,30,36,47
Round 4: 1,5,6,27,42,59
Round 5: 1,2,6,16,19,46
Round 6: 7,13,19,22,40,47
Round 7: 3,5,20,21,38,56
Round 8: 4,17,37,38,47,53
Round 9: 8,43,54,55,56,60
Round 10: 4,18,21,25,38,57
Round 11: 15,25,37,38,58,59
""".strip()

# =========================
# PARSE + VALIDAÇÃO
# =========================
def parse_rounds(text: str):
    round_ids, rounds = [], []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"Faltou ':' na linha: {line}")

        left, right = line.split(":", 1)
        rid_nums = re.findall(r"\d+", left)
        if not rid_nums:
            raise ValueError(f"Não achei o número do round antes de ':': {line}")
        rid = int(rid_nums[0])

        vals = list(map(int, re.findall(r"\d+", right)))
        if len(vals) != PICK:
            raise ValueError(f"Round {rid}: esperado {PICK} dezenas após ':', veio {len(vals)} | {line}")

        round_ids.append(rid)
        rounds.append(sorted(vals))
    return round_ids, rounds

def validate_rounds(round_ids, rounds):
    for rid, r in zip(round_ids, rounds):
        if len(r) != PICK:
            raise ValueError(f"Round {rid}: não tem {PICK} dezenas: {r}")
        if len(set(r)) != PICK:
            raise ValueError(f"Round {rid}: tem repetição: {r}")
        if any(x < 1 or x > MAX_N for x in r):
            raise ValueError(f"Round {rid}: fora de 1..{MAX_N}: {r}")

# =========================
# FUNÇÕES DE "LÓGICA"
# =========================
def bucket(n: int) -> int:
    if 1 <= n <= 20: return 1
    if 21 <= n <= 40: return 2
    return 3

def faixa_comp(nums):
    c1 = sum(1 for x in nums if 1 <= x <= 20)
    c2 = sum(1 for x in nums if 21 <= x <= 40)
    c3 = PICK - c1 - c2
    return c1, c2, c3

def has_seq3(nums):
    nums = sorted(nums)
    for i in range(PICK - 2):
        if nums[i] + 1 == nums[i+1] and nums[i+1] + 1 == nums[i+2]:
            return True
    return False

# =========================
# PROBABILIDADES TEÓRICAS (as “lógicas” que você citou)
# =========================
TOTAL_COMB = comb(MAX_N, PICK)

def p_overlap_k(k: int) -> float:
    # P(k comuns) = C(6,k)*C(54,6-k)/C(60,6)
    return comb(PICK, k) * comb(MAX_N - PICK, PICK - k) / TOTAL_COMB

def p_odds(o: int) -> float:
    # 30 ímpares e 30 pares
    return comb(30, o) * comb(30, PICK - o) / TOTAL_COMB

def p_faixa(c1: int, c2: int, c3: int) -> float:
    # 3 faixas de 20
    return comb(20, c1) * comb(20, c2) * comb(20, c3) / TOTAL_COMB

# Soma: usamos média/variância teóricas do somatório sem reposição
# média de 1..60 = 30.5 -> soma média = 6*30.5 = 183
# variância populacional 1..N: (N^2 - 1)/12
# variância do somatório sem reposição: n*(N-n)/(N-1) * var_pop
N = MAX_N
n = PICK
mu_sum = n * (N + 1) / 2
var_pop = (N**2 - 1) / 12
var_sum = n * (N - n) / (N - 1) * var_pop
sd_sum = math.sqrt(var_sum)

def p_sum_two_sided(s: int) -> float:
    # p-value bicaudal aprox normal: erfc(|z|/sqrt(2))
    z = (s - mu_sum) / sd_sum
    return math.erfc(abs(z) / math.sqrt(2))

def log_safe(p: float) -> float:
    return math.log(max(p, 1e-300))

# =========================
# "ADIVINHAR" (na prática: escolher o mais provável pelo modelo)
# =========================
def score_candidate(nums, last_nums_set):
    nums = sorted(nums)

    # Restrições opcionais
    if FORBID_ONE_BUCKET:
        bs = {bucket(x) for x in nums}
        if len(bs) == 1:
            return None  # rejeita

    if FORBID_SEQ3 and has_seq3(nums):
        return None

    ov = len(set(nums) & last_nums_set)
    if MAX_OVERLAP_WITH_LAST is not None and ov > MAX_OVERLAP_WITH_LAST:
        return None

    o = sum(x % 2 for x in nums)
    c1, c2, c3 = faixa_comp(nums)
    s = sum(nums)

    # Modelo: soma de log-probabilidades (quanto maior, mais “provável” no modelo)
    L = 0.0
    L += W_ODD   * log_safe(p_odds(o))
    L += W_FAIXA * log_safe(p_faixa(c1, c2, c3))
    L += W_SOMA  * log_safe(p_sum_two_sided(s))
    L += W_OVLP  * log_safe(p_overlap_k(ov))

    details = {
        "nums": nums,
        "sum": s,
        "odds": o,
        "faixa": (c1, c2, c3),
        "overlap_last": ov,
        "score": L
    }
    return details

def suggest_next_round(round_ids, rounds, n_samples=N_SAMPLES, top_k=TOP_K, seed=123):
    random.seed(seed)

    existing = {tuple(r) for r in rounds}  # evitar repetir round idêntico
    last = rounds[-1]
    last_set = set(last)

    best = []  # lista de dicts, vamos manter top_k maiores por score

    for _ in range(n_samples):
        cand = sorted(random.sample(range(1, MAX_N + 1), PICK))
        if tuple(cand) in existing:
            continue

        d = score_candidate(cand, last_set)
        if d is None:
            continue

        # mantém top_k
        if len(best) < top_k:
            best.append(d)
            best.sort(key=lambda x: x["score"], reverse=True)
        else:
            if d["score"] > best[-1]["score"]:
                best[-1] = d
                best.sort(key=lambda x: x["score"], reverse=True)

    return best

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    round_ids, rounds = parse_rounds(ROUNDS_TEXT)
    validate_rounds(round_ids, rounds)

    print(f"Rounds carregados: {len(rounds)} | Último round: {round_ids[-1]} -> {rounds[-1]}")
    print("\nTeórico (overlap k):")
    for k in range(7):
        pct = p_overlap_k(k) * 100
        if k == 6:
            print(f"  k={k}: {pct:.10f}%  (~1 em {TOTAL_COMB:,})")
        else:
            print(f"  k={k}: {pct:.5f}%")

    best = suggest_next_round(round_ids, rounds)

    next_id = max(round_ids) + 1
    print("\n" + "="*70)
    print(f"SUGESTÕES DE PRÓXIMO ROUND (Round {next_id}) — 'mais provável' pelo MODELO")
    print("="*70)

    for i, d in enumerate(best, start=1):
        print(f"{i:>2}) Round {next_id}: {d['nums']} | soma={d['sum']} | ímpares={d['odds']} "
              f"| faixas={d['faixa']} | overlap c/último={d['overlap_last']} | score={d['score']:.3f}")

