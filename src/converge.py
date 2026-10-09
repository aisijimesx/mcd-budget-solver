"""分阶段收敛：把试算次数从120 压到 20 以内。

问题背景
--------
实测每calculate-price 调用约 0.34 秒（MCP 单次往返），120 次串行请求
意味着 40 秒以上的等待。Skill 的用户不会等这么久，一次查询超时
就等于产品不可用。

核心思路：**先本地剪枝，再花钱试算。**

优惠券是「指定商品专用」的，这是极强的结构信息。实测每张券的
`applicable_codes` 只含 1~2 个商品编码，也就是说：

  - 一张券只能挂在它指定的商品上
  - 因此含券的组合必然包含那个商品
  - 不含该商品的组合，无论叠多少券都是同一个价

所以真正需要试算的组合数量，比朴素枚举少一个数量级。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from budget_solver import (
    Combo,
    Coupon,
    MenuItem,
    PriceQuote,
    RateLimiter,
    score_combo,
)


@dataclass
class ConvergenceResult:
    """分阶段求解的结果。"""

    ok: bool
    best: Combo | None = None
    quote: PriceQuote | None = None
    baseline_quote: PriceQuote | None = None
    evaluated: int = 0
    phase1_evaluated: int = 0
    phase2_evaluated: int = 0
    reason: str = ""
    runners_up: list[tuple[Combo, PriceQuote]] | None = None


def seed_items_for_coupons(
    menu: Sequence[MenuItem],
    coupons: Sequence[Coupon],
) -> set[str]:
    """收集所有券指定的商品编码 —— 求解必须以这些商品为种子。

    没有这些商品，任何券都挂不上去，券组合就是无效的。
    """
    seeds: set[str] = set()
    for coupon in coupons:
        seeds.update(coupon.applicable_codes)
    return seeds & {m.code for m in menu}


def build_phase1(menu: Sequence[MenuItem], budget: float) -> list[Combo]:
    """第一阶段：无券组合。

    目标是找到「不靠券时预算内能吃最好的方案」，作为基准线。
    只需枚举少量高价值组合，不必铺开。
    """
    staples = [m for m in menu if m.is_combo] or list(menu)
    staples = [m for m in staples if m.price <= budget]
    staples.sort(key=lambda m: m.price)

    combos: list[Combo] = []
    # 预算区间内各取几个代表：最贴近预算的、略低的、最便宜的
    for item in staples[-6:]:
        combos.append(Combo(items=(item,)))
    for item in staples[:3]:
        combos.append(Combo(items=(item,)))
    # 两个单品凑一份
    cheap = [m for m in menu if m.price <= budget * 0.6]
    cheap.sort(key=lambda m: m.price)
    if len(cheap) >= 2:
        combos.append(Combo(items=tuple(cheap[:2])))
    return combos


def build_phase2(
    menu: Sequence[MenuItem],
    coupons: Sequence[Coupon],
    budget: float,
    baseline: tuple[Combo, PriceQuote] | None,
) -> list[Combo]:
    """第二阶段：含券组合。

    围绕每张券的适用商品构造候选，只保留**券后可能进入预算**的组合。
    关键剪枝：券前原价超过预算 1.3 倍的组合直接丢弃，因为券最多省几十元。
    """
    by_code = {m.code: m for m in menu}
    seeds = seed_items_for_coupons(menu, coupons)
    if not seeds:
        return []

    combos: list[Combo] = []
    fillers = sorted(
        (m for m in menu if m.code not in seeds),
        key=lambda m: m.price,
    )

    for coupon in coupons:
        if not coupon.valid:
            continue
        for seed_code in coupon.applicable_codes:
            seed = by_code.get(seed_code)
            if seed is None or seed.price > budget:
                continue

            # 券 + 种子商品
            combo = Combo(items=(seed,), coupons=(coupon,))
            if combo.list_price <= budget * 1.3:
                combos.append(combo)

            # 券 + 种子 + 1 件配菜（凑单让券更划算的常见形态）
            for filler in fillers[:5]:
                if seed.price + filler.price > budget * 1.3:
                    break
                combo = Combo(items=(seed, filler), coupons=(coupon,))
                if combo.list_price <= budget * 1.3:
                    combos.append(combo)

            # 券 + 种子 + 2 件配菜
            if len(fillers) >= 2:
                for i, f1 in enumerate(fillers[:5]):
                    for f2 in fillers[i + 1:6]:
                        if seed.price + f1.price + f2.price > budget * 1.3:
                            break
                        combo = Combo(items=(seed, f1, f2), coupons=(coupon,))
                        combos.append(combo)

    # 多券组合：只有当多张券能挂在同一组商品上才尝试
    if len(coupons) >= 2:
        for c1 in coupons:
            for c2 in coupons:
                if c1 is c2:
                    continue
                shared = set(c1.applicable_codes) & set(c2.applicable_codes)
                for code in shared:
                    seed = by_code.get(code)
                    if seed is None:
                        continue
                    combo = Combo(items=(seed,), coupons=(c1, c2))
                    if combo.list_price <= budget * 1.3:
                        combos.append(combo)

    return combos


def solve_converged(
    menu: Sequence[MenuItem],
    coupons: Sequence[Coupon],
    budget: float,
    servings: int,
    calculate_price: Callable[[Combo], PriceQuote],
    max_evaluations: int = 24,
    limiter: RateLimiter | None = None,
) -> ConvergenceResult:
    """分两阶段求解，把试算次数控制在可接受范围内。

    阶段一：无券组合，找到基准线。
    阶段二：含券组合，只保留可能进预算的候选。
    两阶段结果合并，按效用分取胜出者。
    """
    limiter = limiter or RateLimiter()
    cache: dict[tuple, PriceQuote] = {}
    evaluated = 0

    def evaluate(combo: Combo) -> PriceQuote | None:
        nonlocal evaluated
        key = (tuple(i.code for i in combo.items),
               tuple(c.coupon_id for c in combo.coupons))
        if key in cache:
            return cache[key]
        if evaluated >= max_evaluations:
            return None
        limiter.acquire()
        try:
            quote = calculate_price(combo)
        except Exception:  # noqa: BLE001
            return None
        cache[key] = quote
        evaluated += 1
        return quote

    # --- 阶段一：无券基准 -------------------------------------------------
    phase1 = build_phase1(menu, budget)
    p1_feasible: list[tuple[float, float, Combo, PriceQuote]] = []
    baseline: tuple[Combo, PriceQuote] | None = None

    for combo in phase1:
        quote = evaluate(combo)
        if quote is None:
            continue
        if baseline is None or quote.effective_payable < baseline[1].effective_payable:
            baseline = (combo, quote)
        if quote.effective_payable <= budget + 0.01:
            score = score_combo(combo, _req(budget, servings), quote.effective_payable)
            p1_feasible.append((score, quote.effective_payable, combo, quote))
    p1_count = evaluated

    # --- 阶段二：含券组合 -------------------------------------------------
    phase2 = build_phase2(menu, coupons, budget, baseline)
    p2_feasible: list[tuple[float, float, Combo, PriceQuote]] = []
    for combo in phase2:
        quote = evaluate(combo)
        if quote is None:
            continue
        if quote.effective_payable <= budget + 0.01:
            score = score_combo(combo, _req(budget, servings), quote.effective_payable)
            p2_feasible.append((score, quote.effective_payable, combo, quote))

    all_feasible = p1_feasible + p2_feasible
    if not all_feasible:
        return ConvergenceResult(
            ok=False,
            reason=f"预算 {budget:.0f} 元内没有可行组合，已试算 {evaluated} 组。"
                   f"建议提高预算或放宽忌口。",
            evaluated=evaluated,
            phase1_evaluated=p1_count,
            phase2_evaluated=evaluated - p1_count,
        )

    all_feasible.sort(key=lambda t: (-t[0], t[1]))
    _, _, best_combo, best_quote = all_feasible[0]
    runners = [(c, q) for _, _, c, q in all_feasible[1:4]]

    return ConvergenceResult(
        ok=True,
        best=best_combo,
        quote=best_quote,
        baseline_quote=baseline[1] if baseline else None,
        evaluated=evaluated,
        phase1_evaluated=p1_count,
        phase2_evaluated=evaluated - p1_count,
        reason=f"试算 {evaluated} 组，其中 {len(p2_feasible)} 组用券后进入预算。",
        runners_up=runners,
    )


def _req(budget: float, servings: int):
    """构造轻量请求对象供 score_combo 使用。"""
    from budget_solver import BudgetRequest
    return BudgetRequest(budget=budget, servings=servings)


if __name__ == "__main__":
    print(__doc__)
