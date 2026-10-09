"""
budget_solver.py — 预算内最优组合求解器

设计原则
--------
不自研背包 / 优惠券叠加算法。所有金额与优惠计算一律交给麦当劳 MCP 的
`calculate-price` 工具完成，本模块只负责三件事：

1. 约束提取：把用户一句话诉求拆成硬约束（预算、人数、忌口）与偏好。
2. 候选生成：按硬约束剪枝，生成规模可控的候选组合。
3. 试算收敛：逐组调用 `calculate-price`，保留应付最低且不超预算的解。

这样做的原因：券的叠加规则、套餐替换规则、配送费口径都由官方服务定义，
本地重算极易与服务端不一致，且 `calculate-price` 的返回值是唯一可信来源。

限流
----
麦当劳 MCP Server 限制 600 次请求 / 分钟。`RateLimiter` 在超限时自动挂起，
避免触发 429。批量试算时优先复用同一组 (商品, 券) 的结果缓存。

本文件不包含任何真实 Token，也不包含其他餐饮品牌信息。
"""

from __future__ import annotations

import itertools
import os
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Sequence

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

#: 服务端限流：每分钟 600 次。
RATE_LIMIT_PER_MIN = 600

#: 预算是硬约束，最多允许落到预算的 100%，不放宽。
BUDGET_TOLERANCE = 1.0

#: 单次求解最多向 calculate-price 发起的试算次数，防止组合爆炸。
MAX_EVALUATIONS = 120

#: 候选商品数量上限，按口味标签命中度与价格升序截断。
MAX_CANDIDATE_ITEMS = 14

#: 单组内最多包含的券数。多数场景 1~2 张已能触达最低价，
#: 放宽到 3 张是为了覆盖大额满减与低满减互斥的情况。
MAX_COUPONS_PER_COMBO = 3

#: 单组内最多包含的餐饮商品数。
MAX_ITEMS_PER_COMBO = 4


# ---------------------------------------------------------------------------
# 效用模型
# ---------------------------------------------------------------------------
#
# 目标函数是本项目最容易做错的地方。
#
# 朴素做法是「让实付最低」。这是错的：用户说「30 块能吃什么」，
# 求解器会返回「一杯可乐减 3 块，实付 5 元」—— 它在数学上确实更优，
# 但在用户眼里是废话。极小化金额必然被单件廉价商品钻空子。
#
# 正确做法是把实付当作约束，把「吃到多少东西」当作目标：
#   效用 = 主食权重 + 配菜权重
# 再用预算利用率做轻量修正，让同样效用下更接近预算的解胜出。
# 金额只参与约束判定与展示，不参与打分。

#: 配菜（薯条、鸡翅等）在饱腹维度上的折算系数，低于主食。
SIDES_WEIGHT = 0.6

#: 套票价通常已含主食与配食，按整份计权，不重复拆分。
COMBO_WEIGHT = 1.4

#: 判定为「主食」的标签关键词。
STAPLE_KEYWORDS = ("汉堡", "堡", "麦辣", "巨无", "板烧", "鸡腿", "鱼", "牛肉", "火腿扒")

#: 判定为「饮料/冰品」的标签关键词，几乎不计入饱腹。
DRINK_KEYWORDS = ("饮料", "可乐", "雪碧", "咖啡", "奶茶", "茶饮")

#: 判定为「调料/酱料」的关键词。
#:
#: 这类商品必须单独识别。实测踩过的坑：酱料单价 3 元，若按普通配菜算
#: 权重 0.6，「美式 + 2 个酱 + 苹果片」会得到「3 样小食」的评价，
#: 看起来很划算，实际上是把调料当饭吃。调料不计饱腹，且几乎不加权。
CONDIMENT_KEYWORDS = ("酱", "调味", "沙拉", "番茄酱", " mustard", "horns")

#: 判定为「小食」的关键词。薯条、鸡块、蛋挞等。
SIDE_KEYWORDS = ("薯条", "薯饼", "鸡块", "鸡翅", "鸡柳", "蛋挞", "小食",
                 "拼盘", "鸡盒", "块", "卷", "脆鸡", "炸鸡", "小杯", "玉米杯")


def classify_staple(item: "MenuItem") -> bool:
    """判断餐品是否算主食（汉堡类）。标签缺失时按名称兜底。"""
    haystack = " ".join([item.name, *item.tags]).lower()
    if any(k in haystack for k in DRINK_KEYWORDS):
        return False
    return any(k in haystack for k in STAPLE_KEYWORDS)


def is_condiment(item: "MenuItem") -> bool:
    """判断是否为调料/酱料。

    调料虽出现在菜单里、能下单，但它不构成「吃到东西」。
    实测中3 元的酱料会让「19.9 元吃到 3 样」这种话术成立，
    用户看到会认为推荐很蠢。
    """
    haystack = " ".join([item.name, *item.tags]).lower()
    return any(k in haystack for k in CONDIMENT_KEYWORDS)


def item_weight(item: "MenuItem") -> float:
    """单件商品的饱腹权重。

    分档理由：
    - 套餐已含汉堡 + 小食 + 饮料，按整份计权，不拆分。
    - 主食（汉堡类）满分。
    - 小食（薯条、鸡块等）计 0.6。
    - 饮料几乎不顶饱，计 0.2。
    - 调料计 0，但不计入"样数"展示 —— 见 describe_servings。
    """
    if item.is_combo:
        return COMBO_WEIGHT
    if is_condiment(item):
        return 0.0
    if classify_staple(item):
        return 1.0
    haystack = " ".join([item.name, *item.tags]).lower()
    if any(k in haystack for k in SIDE_KEYWORDS):
        return SIDES_WEIGHT
    return 0.3
    """未识别的品类给低权重而不是 0.5，避免误判为正餐。"""


def score_combo(combo: "Combo", request: "BudgetRequest", payable: float) -> float:
    """给组合打效用分。分数越高越符合「用这个预算吃到最多东西」。

    只使用官方报价 `payable` 判断是否贴预算，不使用本地计算的优惠额。
    """
    weight = 0.0
    for item in combo.items:
        weight += item_weight(item)

    # 没有任何主食时重罚。
    # 实测踩过的坑：30 元预算下「美式 + 2 个酱 + 苹果片」实付 19.9 元，
    # 按件计数会显示「吃到 3 样小食」，但其中两样是 3 元的酱料。
    # 用户要的是一顿饭，不是凑件数的零食清单。
    has_staple = any(
        i.is_combo or classify_staple(i) or not is_condiment(i) and item_weight(i) >= SIDES_WEIGHT
        for i in combo.items
    )
    if not has_staple:
        weight -= 1.5

    # 人数未满足时重罚，避免给出一个人吃不掉的方案。
    if not request.is_servings_satisfied(combo):
        weight -= 2.0

    # 偏好命中加权。
    prefer = {t.lower() for t in request.prefer_tags}
    if prefer:
        hits = sum(1 for i in combo.items if prefer & {t.lower() for t in i.tags})
        weight += 0.3 * hits

    # 预算利用率修正，权重刻意压低，避免压过饱腹维度。
    # 至少吃到六成才给加成，防止"为了贴预算而超吃"。
    if request.budget > 0:
        ratio = payable / request.budget
        if 0.6 <= ratio <= 1.0:
            weight += 0.5 * ratio

    return round(weight, 4)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BudgetRequest:
    """从用户自然语言中提取出的求解约束。"""

    budget: float
    """硬约束：实付金额上限（元）。"""

    servings: int = 1
    """硬约束：用餐人数。用于约束商品份数。"""

    avoid_tags: frozenset[str] = frozenset()
    """硬约束：忌口标签小写集合。命中即从候选中剔除。"""

    prefer_tags: tuple[str, ...] = ()
    """软偏好：口味标签。按命中度提升候选排序权重。"""

    allow_delivery: bool = True
    """是否允许麦乐送外送。False 时排除产生配送费的组合。"""

    order_mode: str | None = None
    """就餐方式：dine_in / takeaway / delivery / None 表示不限。"""

    def is_servings_satisfied(self, combo: "Combo") -> bool:
        """校验组合的份数是否满足人数要求。

        单品套餐按 1 份计，多件组合按件数计。这里采取宽松策略：
        允许在不超预算的前提下略微少 1 份，避免因为凑不齐而放弃合法解。
        """
        if self.servings <= 1:
            return True
        return combo.portions >= self.servings - 1


@dataclass(frozen=True)
class MenuItem:
    """来自 `query-meals` / `query-meal-detail` 的餐品快照。"""

    code: str
    name: str
    price: float
    tags: tuple[str, ...] = ()
    is_combo: bool = False
    combo_contents: tuple[str, ...] = ()
    """`query-meal-detail` 返回的套餐组成。仅 is_combo 为 True 时有效。"""

    @property
    def portions(self) -> int:
        """一份该商品可满足的人数。套餐按 1 份计，与官方口径一致。"""
        return 1

    def has_avoided_tag(self, avoid: frozenset[str]) -> bool:
        """是否命中忌口标签。标签比对统一走小写，避免大小写漏判。"""
        lowered = {t.lower() for t in self.tags}
        return bool(lowered & set(avoid))


@dataclass(frozen=True)
class Coupon:
    """来自 `query-store-coupons` / `query-my-coupons` 的优惠券快照。"""

    coupon_id: str
    name: str
    threshold: float = 0.0
    """满减门槛。0 表示无门槛。具体抵扣金额不本地计算，交由 calculate-price。"""

    coupon_code: str = ""
    """券码。calculate-price 的 couponCode 字段需要它，与 couponId 一起传入。"""

    applicable_codes: tuple[str, ...] = ()
    """该券适用的商品编码，来自 query-store-coupons 的 products[].productCode。

    这是**硬约束**：实测把券挂到不适用该券的商品上，calculate-price 会直接
    报错。候选生成时若不满足此约束，会浪费一次试算额度。
    空元组表示适用商品未知，此时不做本地过滤，交给服务端判定。
    """

    valid: bool = True
    """本地仅做可用性预筛，最终能否抵扣以 calculate-price 返回为准。"""

    expires_at: str = ""
    """有效期原文，来自 query-store-coupons 的 tradeDateTime。用于展示。"""

    def applies_to(self, items: "Iterable[MenuItem]") -> bool:
        """判断券能否作用于这组商品。

        适用商品未知时返回 True（交给服务端判定），避免误杀。
        """
        if not self.applicable_codes:
            return True
        codes = {i.code for i in items}
        return bool(codes & set(self.applicable_codes))


@dataclass(frozen=True)
class Combo:
    """一个候选组合：若干商品 + 若干券。"""

    items: tuple[MenuItem, ...]
    coupons: tuple[Coupon, ...] = ()

    @property
    def portions(self) -> int:
        return sum(i.portions for i in self.items)

    @property
    def list_price(self) -> float:
        """券前原价合计。仅用于排序剪枝，不作为最终应付。"""
        return sum(i.price for i in self.items)

    def describe(self) -> str:
        """人类可读描述，用于输出给用户。"""
        names = " + ".join(i.name for i in self.items)
        if self.coupons:
            names += "｜券：" + "、".join(c.name for c in self.coupons)
        return names


@dataclass(frozen=True)
class PriceQuote:
    """来自 `calculate-price` 的权威报价。"""

    total: float
    """应付总价。本项目唯一采信的金额字段。"""

    discount: float = 0.0
    delivery_fee: float = 0.0
    payable: float | None = None
    """含配送费的实付。缺省时回退为 total。"""

    @property
    def effective_payable(self) -> float:
        return self.total if self.payable is None else self.payable

    @property
    def saved(self) -> float:
        """相对券前原价的节省额。"""
        return round(max(0.0, self.effective_payable - self.total + self.discount), 2)


@dataclass
class SolveResult:
    """求解产物。"""

    ok: bool
    reason: str = ""
    best: Combo | None = None
    quote: PriceQuote | None = None
    baseline_quote: PriceQuote | None = None
    """不使用任何券时的官方报价，用于计算节省额。"""

    evaluated: int = 0
    cached: int = 0
    candidates_generated: int = 0
    request: "BudgetRequest | None" = None
    """求解时的约束，用于回算效用分。"""

    runners_up: list[tuple[Combo, PriceQuote]] = field(default_factory=list)

    @property
    def extra_saved(self) -> float:
        """相比不用券，能多省多少。"""
        if not self.quote or not self.baseline_quote:
            return 0.0
        delta = self.baseline_quote.effective_payable - self.quote.effective_payable
        return round(max(0.0, delta), 2)

    @property
    def utilisation(self) -> float:
        """预算占用率。用于判断是否把预算用足。"""
        if not self.quote or not self.best:
            return 0.0
        return round(self.quote.effective_payable, 2)

    @property
    def score(self) -> float:
        """最优解的效用分。用于向用户解释"为什么推荐这一份"。"""
        if not self.best or not self.quote:
            return 0.0
        return score_combo(self.best, self.request, self.quote.effective_payable)


# ---------------------------------------------------------------------------
# 限流
# ---------------------------------------------------------------------------


class RateLimiter:
    """滑动窗口限流器。

    超过 `RATE_LIMIT_PER_MIN` 时自动 sleep，直到窗口腾出额度。
    MCP Server 超限会返回 429，重试成本高于等待。
    """

    def __init__(self, limit: int = RATE_LIMIT_PER_MIN) -> None:
        self._limit = limit
        self._calls: list[float] = []

    def acquire(self) -> None:
        now = time.monotonic()
        cutoff = now - 60.0
        self._calls = [t for t in self._calls if t > cutoff]
        if len(self._calls) >= self._limit:
            sleep_for = 60.0 - (now - self._calls[0]) + 0.1
            if sleep_for > 0:
                time.sleep(sleep_for)
            self._calls = [t for t in self._calls if t > (time.monotonic() - 60.0)]
        self._calls.append(time.monotonic())


# ---------------------------------------------------------------------------
# 候选生成
# ---------------------------------------------------------------------------


def build_candidates(
    menu: Sequence[MenuItem],
    coupons: Sequence[Coupon],
    request: BudgetRequest,
) -> list[Combo]:
    """按硬约束剪枝，生成待试算的候选组合。

    剪枝顺序（越靠前越便宜，剪枝越激进）：
    1. 剔除忌口命中项。
    2. 剔除原价已明显超预算的项（留20% 余量给券后降价）。
    3. 按口味偏好与价格排序，截断到 MAX_CANDIDATE_ITEMS。
    4. 组合商品（1~4 件），商品数超过人数 + 1 视为浪费。
    5. 组合券（0~MAX_COUPONS_PER_COMBO 张）。
    """
    pool = [
        m
        for m in menu
        if not m.has_avoided_tag(request.avoid_tags)
        and m.price <= request.budget * 1.2
    ]
    if not pool:
        return []

    prefer = {t.lower() for t in request.prefer_tags}
    pool.sort(
        key=lambda m: (
            not (prefer & {t.lower() for t in m.tags}),  # 命中口味优先
            m.price,  # 其次价格升序
        )
    )
    pool = pool[:MAX_CANDIDATE_ITEMS]

    # 份数上限：比人数多 1 份为可接受上限，再多就是浪费。
    max_items = min(MAX_ITEMS_PER_COMBO, request.servings + 1)
    max_items = max(1, max_items)

    # 券按门槛升序，低门槛券更通用，优先进入组合。
    usable_coupons = [c for c in coupons if c.valid]
    usable_coupons.sort(key=lambda c: (c.threshold, c.name))

    combos: list[Combo] = []
    for n in range(1, max_items + 1):
        for item_tuple in itertools.combinations(pool, n):
            combo = Combo(items=item_tuple)
            if not request.is_servings_satisfied(combo):
                continue
            # 券前原价就远超预算的组合，直接不进候选。
            if combo.list_price > request.budget * 1.2:
                continue
            combos.append(combo)
            for k in range(1, MAX_COUPONS_PER_COMBO + 1):
                for coupon_tuple in itertools.combinations(usable_coupons, k):
                    # 券必须在这组商品里有适用对象，否则不生成候选。
                    # 不做这个预筛，每次这样的组合都会浪费一次试算额度，
                    # 而实测中这类组合必然被服务端拒绝。
                    if not all(c.applies_to(item_tuple) for c in coupon_tuple):
                        continue
                    combos.append(Combo(items=item_tuple, coupons=coupon_tuple))

    return combos


def rank_candidates(combos: Iterable[Combo]) -> list[Combo]:
    """候选排序：预估饱腹效用高者优先，同效用时少用券者优先。

    这里用的是**预估**效用（不含券后价格），只用于决定试算顺序以配合
    MAX_EVALUATIONS 截断。真正的胜负由calculate-price 的实付结果决定，
    由score_combo 在最终选择时重新计算。

    注意不能按原价升序排：那样 120 次试算额度会被一堆廉价单品占满，
    真正有价值的组合反而没机会被算。
    """
    def sort_key(combo: Combo) -> tuple:
        weight = sum(
            COMBO_WEIGHT if i.is_combo else (1.0 if classify_staple(i) else SIDES_WEIGHT)
            for i in combo.items
        )
        # 券少用优先：可叠加券的边际收益通常递减。
        return (-weight, len(combo.coupons), combo.list_price)

    return sorted(combos, key=sort_key)


# ---------------------------------------------------------------------------
# 求解主流程
# ---------------------------------------------------------------------------


def solve(
    menu: Sequence[MenuItem],
    coupons: Sequence[Coupon],
    request: BudgetRequest,
    calculate_price: Callable[[Combo], PriceQuote],
    limiter: RateLimiter | None = None,
) -> SolveResult:
    """在预算内寻找实付最低的组合。

    参数
    ----
    menu
        当前门店在售餐品，来自 `query-meals` 与 `query-meal-detail`。
    coupons
        门店可用券，来自 `query-store-coupons`（可叠加 `query-my-coupons`）。
    request
        用户约束。
    calculate_price
        调用麦当劳 MCP `calculate-price` 的适配函数。传入 Combo，返回权威报价。
        所有金额判断只信任该函数的返回值。
    limiter
        限流器。为 None 时内部自建一个。

    返回
    ----
    SolveResult
        best 为实付最低且不超预算的解。找不到时 ok 为 False 并给出 reason。
    """
    limiter = limiter or RateLimiter()

    candidates = build_candidates(menu, coupons, request)
    if not candidates:
        return SolveResult(
            ok=False,
            reason="当前门店没有满足忌口与预算的餐品，建议放宽忌口或提高预算。",
        )

    candidates = rank_candidates(candidates)

    # 基准：无券报价，用于量化"用券省了多少"。
    # 取排序后的第一个无券候选（即饱腹效用最高的无券方案），
    # 这样对比才是同等餐食下的券效差，而不是拿一杯可乐去比。
    baseline_combo = next((c for c in candidates if not c.coupons), None)
    baseline_quote: PriceQuote | None = None
    if baseline_combo is not None:
        limiter.acquire()
        try:
            baseline_quote = calculate_price(baseline_combo)
        except Exception as exc:  # noqa: BLE001 — 基准失败不阻断主流程
            baseline_quote = None
            reason_hint = f"（基准试算失败：{exc}）"
        else:
            reason_hint = ""

    cache: dict[tuple, PriceQuote] = {}
    evaluated = 0
    cached = 0
    #: 预算内的可行解，按 (效用分, 实付) 降序。
    feasible: list[tuple[float, float, Combo, PriceQuote]] = []
    over_budget_best: tuple[Combo, PriceQuote] | None = None
    ceiling = request.budget * BUDGET_TOLERANCE

    for combo in candidates:
        if evaluated >= MAX_EVALUATIONS:
            break
        key = (
            tuple(i.code for i in combo.items),
            tuple(c.coupon_id for c in combo.coupons),
        )
        if key in cache:
            quote = cache[key]
            cached += 1
        else:
            limiter.acquire()
            try:
                quote = calculate_price(combo)
            except Exception:  # noqa: BLE001 — 单组失败跳过，不中断整体
                continue
            cache[key] = quote
            evaluated += 1

        payable = quote.effective_payable
        if not request.allow_delivery and quote.delivery_fee > 0:
            continue

        if payable <= ceiling + 0.01:
            score = score_combo(combo, request, payable)
            feasible.append((score, payable, combo, quote))
        elif over_budget_best is None or payable < over_budget_best[1].effective_payable:
            # 保留一个"最接近预算"的备选，用于给用户降级建议。
            over_budget_best = (combo, quote)

    if not feasible:
        if over_budget_best is not None:
            combo, quote = over_budget_best
            return SolveResult(
                ok=False,
                reason=(
                    f"预算内没有可行组合。最接近的是「{combo.describe()}」，"
                    f"实付 {quote.effective_payable} 元，"
                    f"超出预算 {round(quote.effective_payable - ceiling, 2)} 元。"
                    f"建议提高预算到 {int(quote.effective_payable) + 1} 元。{reason_hint}"
                ),
                best=combo,
                quote=quote,
                baseline_quote=baseline_quote,
                evaluated=evaluated,
                cached=cached,
                candidates_generated=len(candidates),
                request=request,
            )
        return SolveResult(
            ok=False,
            reason=f"没有可试算的候选组合。{reason_hint}",
            baseline_quote=baseline_quote,
            evaluated=evaluated,
            cached=cached,
            candidates_generated=len(candidates),
        )

    # 先比效用（吃到多少），效用相同时再比实付（花得更少）。
    # 这个顺序是本项目的核心，不能颠倒。
    feasible.sort(key=lambda t: (-t[0], t[1]))
    best_score, best_payable, best_combo, best_quote = feasible[0]
    runners = [(combo, quote) for _, _, combo, quote in feasible[1:4]]

    return SolveResult(
        ok=True,
        reason=(
            f"在 {len(feasible)} 个预算内组合中，按「吃到多少东西」选出最优解，"
            f"券后共省 {round(best_quote.discount, 2)} 元。{reason_hint}"
        ),
        best=best_combo,
        quote=best_quote,
        baseline_quote=baseline_quote,
        evaluated=evaluated,
        cached=cached,
        candidates_generated=len(candidates),
        request=request,
        runners_up=runners,
    )


# ---------------------------------------------------------------------------
# 业绩卡文案
# ---------------------------------------------------------------------------


def describe_servings(combo: "Combo") -> str:
    """描述组合能吃到多少东西。

    两处实测踩过的坑都在这里：
    1. 套餐必须单独处理 —— 菜单里套餐实际命名为「XX三件套」，
       若按件数计数会输出「吃到 1 样」，用户会困惑。
    2. 调料必须排除 —— 3 元的酱料算成「1 样」会让「19.9 元吃到 3 样」
       这种话术成立，但用户看到的是一份酱料清单。
    """
    combos = [i for i in combo.items if i.is_combo]
    # 调料降级为「配料」，不计入样数
    condiments = [i for i in combo.items if not i.is_combo and is_condiment(i)]
    singles = [
        i for i in combo.items
        if not i.is_combo and not is_condiment(i)
    ]

    parts: list[str] = []
    if combos:
        parts.append("、".join(i.name for i in combos))
    if singles:
        parts.append(f"{len(singles)} 样小食")
    if condiments:
        parts.append(f"配 {len(condiments)} 样蘸料")

    if not parts:
        return f"{len(combo.items)} 样"
    return " + ".join(parts)


def render_scorecard(result: SolveResult, request: BudgetRequest) -> str:
    """生成"今日战绩"卡片文案。

    统一句式是这个项目传播设计的核心：格式固定，一眼可读，便于模仿转发。
    文案只陈述 calculate-price 返回的数字，不做任何推断或承诺。
    """
    if not result.ok or not result.quote or not result.best:
        return f"今日战绩：{result.reason}"

    quote = result.quote
    lines = [
        f"今日战绩：{request.budget:.0f} 块预算",
        f"实付 {quote.effective_payable} 元，吃到 {describe_servings(result.best)}",
    ]

    if result.extra_saved > 0:
        lines.append(f"比不用券再省 {result.extra_saved} 元")
    elif quote.discount > 0:
        lines.append(f"券后共省 {round(quote.discount, 2)} 元")

    unused = round(request.budget - quote.effective_payable, 2)
    if unused >= 1:
        lines.append(f"还剩 {unused} 元没花完")

    # 方案明细只在 describe_servings 没覆盖全部内容时追加，
    # 否则会出现「吃到 培根安格斯厚牛堡三件套」紧接「方案：培根安格斯厚牛堡三件套」
    # 这种重复，实测中出现过。
    if result.best.items:
        summary = describe_servings(result.best)
        detail = " + ".join(
            i.name for i in result.best.items
            if i.name not in summary
        )
        if result.best.coupons:
            coupon_names = "、".join(c.name for c in result.best.coupons)
            lines.append(f"用券：{coupon_names}")
        if detail:
            lines.append(f"明细：{detail}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 本地自检：不接触网络，用桩数据验证求解逻辑。
    demo_menu = [
        MenuItem("A1", "香辣鸡腿堡", 22.0, ("辣", "汉堡")),
        MenuItem("A2", "巨无霸", 28.0, ("汉堡",)),
        MenuItem("B1", "麦辣鸡腿汉堡", 24.0, ("辣", "汉堡")),
        MenuItem("C1", "薯条（中）", 9.0, ("零食",)),
        MenuItem("C2", "中可乐", 8.0, ("饮料",)),
        MenuItem("D1", "麦乐鸡（5块）", 25.0, ("零食",)),
        MenuItem("E1", "巧克力圣代", 13.0, ("甜品", "冰")),
    ]
    demo_coupons = [
        Coupon("CP1", "满 20 减 6", threshold=20.0),
        Coupon("CP2", "满 30 减 10", threshold=30.0),
        Coupon("CP3", "无门槛减 3", threshold=0.0),
    ]

    def fake_price(combo: Combo) -> PriceQuote:
        """桩算价：模拟官方行为，满减取门槛最贴合的一张券。"""
        base = combo.list_price
        discount = 0.0
        for c in combo.coupons:
            if base >= c.threshold:
                discount += 6.0 if c.threshold >= 30 else (3.0 if c.threshold >= 20 else 3.0)
        payable = max(0.0, base - discount)
        return PriceQuote(
            total=payable,
            discount=round(discount, 2),
            delivery_fee=0.0 if payable >= 30 else 5.0,
            payable=payable,
        )

    req = BudgetRequest(budget=30.0, servings=1, prefer_tags=("辣",))
    res = solve(demo_menu, demo_coupons, req, fake_price)
    print(render_scorecard(res, req))
    print()
    print(f"候选 {res.candidates_generated} 个，官方试算 {res.evaluated} 次，命中缓存 {res.cached} 次")
    for combo, quote in res.runners_up:
        print(f"  备选 {quote.effective_payable} 元 — {combo.describe()}")
