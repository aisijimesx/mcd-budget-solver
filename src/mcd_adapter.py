"""mcd_adapter.py — 麦当劳 MCP Server 的真实数据适配层。

本文件把 MCP 的真实返回结构映射为求解器用的领域模型。**所有字段名与
单位换算都以实际联调结果为准，不要凭文档推断。**

联调确认的关键事实（2026-10-09 实测）：

1. **金额单位是分。** `calculate-price` 返回 `price: 2350` 表示 23.5 元。
   直接当元用会导致结果差 100 倍。

2. **`items` 的字段是 `productCode`，不是 `code`。**
   传 `code` 会报「缺少商品code或者商品数量quantity不正确」。

3. **`query-nearby-stores` 必须传 `searchType=2` 且同时给 city 与 keyword。**
   `searchType=1` 走的是「收藏餐厅」，未收藏过任何门店时必定失败。

4. **券必须挂在适用的商品上才生效。**
   `couponId` 挂在不适用该券的商品上，`calculate-price` 会直接报错。
   每张券的适用商品编码在 `query-store-coupons` 的 `products[].productCode` 中。

5. **券只抵扣它对应的商品，不影响同单其他商品。**
   实测：巨无霸 26.5 元 + 冰美式 19 元（原价）用9.1 元券，
   结果为 26.5 + 9.9 = 36.4 元，抵扣精确落在冰美式上。

6. **门店券因店而异，必须遍历门店。**
   实测广州塔餐厅券数为 0，同城其他门店各有 3 张。
   找门店时应优先选择有券的门店。

7. **`query-my-coupons` 只返回 markdown 文本，没有结构化字段。**
   返回中不含 couponId / couponCode，只有券名、价格、有效期与图片 URL，
   因此账户券无法程序化用于算价。**券枚举只能依赖 `query-store-coupons`。**

8. **`auto-bind-coupons` 对已领取的券不会重复返回。**
   首次调用成功领到 3 张，二次调用返回空列表。

另有文档与实际不一致之处：实测工具数为 35 个（文档写 33 个），
且存在 `query-promotions`、`query-survey-coupon` 等未在文档中列出的工具。

本模块只做只读查询与算价，不含下单、抽奖等写操作。
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Sequence

from budget_solver import Combo, Coupon, MenuItem, PriceQuote

MCP_URL = "https://mcp.mcd.cn"

# 就餐方式与订单类型。beType 与 orderType 必须成对传，组合非法。
BE_TYPE_DINE_IN = 1        # 到店自提
BE_TYPE_DELIVERY = 2       # 麦乐送到家
BE_TYPE_DRIVE_THRU = 5     # 得来速车道
BE_TYPE_GROUP = 6          # 企业团餐

ORDER_TYPE_DINE_IN = 1     # 到店（含自提与得来速）
ORDER_TYPE_DELIVERY = 2    # 外送（含麦乐送与团餐）

#: MCP 限流：每分钟 600 次。
RATE_LIMIT_PER_MIN = 600


class McdMcpError(RuntimeError):
    """MCP 调用失败。"""


@dataclass
class Quote:
    """calculate-price 的结构化结果，已完成分转元。"""

    price: float
    discount: float
    original_price: float
    product_list: list[dict]

    @classmethod
    def from_yuan(cls, d: dict) -> "Quote":
        return cls(
            price=d.get("price", 0) / 100.0,
            discount=d.get("discount", 0) / 100.0,
            original_price=d.get("originalPrice", 0) / 100.0,
            product_list=d.get("productList") or [],
        )

    def to_quote(self) -> PriceQuote:
        return PriceQuote(
            total=self.price,
            discount=self.discount,
            delivery_fee=0.0,
            payable=self.price,
        )


class McdClient:
    """MCP Server 的轻量客户端。

    只依赖标准库，便于放进Skill 目录直接使用。
    Token 从环境变量 MCD_MCP_TOKEN 读取，不落盘。
    """

    def __init__(self, token: str | None = None, url: str = MCP_URL) -> None:
        self._token = token or os.environ.get("MCD_MCP_TOKEN", "")
        if not self._token:
            raise McdMcpError("未设置 MCD_MCP_TOKEN")
        self._url = url
        self._session: str | None = None
        self._calls: list[float] = []
        self._tools: list[dict] | None = None
        self._handshake()

    # -- 底层协议 --------------------------------------------------------

    def _rpc(self, method: str, params: dict | None = None) -> Any:
        body: dict[str, Any] = {"jsonrpc": "2.0", "id": 1, "method": method}
        if params is not None:
            body["params"] = params
        req = urllib.request.Request(
            self._url, method="POST", data=json.dumps(body).encode()
        )
        req.add_header("Content-Type", "application/json")
        req.add_header("Accept", "application/json, text/event-stream")
        req.add_header("Authorization", f"Bearer {self._token}")
        if self._session:
            req.add_header("Mcp-Session-Id", self._session)

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                sid = resp.headers.get("Mcp-Session-Id")
                if sid:
                    self._session = sid
                raw = resp.read().decode()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode()[:300]
            if exc.code == 401:
                raise McdMcpError("MCP Token 无效或已过期") from exc
            if exc.code == 429:
                raise McdMcpError("触发限流，超过 600 次/分钟") from exc
            raise McdMcpError(f"HTTP {exc.code}: {detail}") from exc
        except Exception as exc:  # noqa: BLE001
            raise McdMcpError(f"{type(exc).__name__}: {exc}") from exc

        # Streamable HTTP 可能返回 SSE 格式
        if raw.startswith(("event:", "data:")):
            for line in raw.splitlines():
                if line.startswith("data:"):
                    raw = line[5:].strip()
                    break
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise McdMcpError(f"响应解析失败: {raw[:200]}") from exc
        if "error" in payload:
            raise McdMcpError(str(payload["error"]))
        return payload.get("result")

    def _throttle(self) -> None:
        """滑动窗口限流，避免触发 429。"""
        now = time.monotonic()
        self._calls = [t for t in self._calls if t > now - 60.0]
        if len(self._calls) >= RATE_LIMIT_PER_MIN:
            wait = 60.0 - (now - self._calls[0]) + 0.1
            if wait > 0:
                time.sleep(wait)
            self._calls = [t for t in self._calls if t > time.monotonic() - 60.0]
        self._calls.append(time.monotonic())

    def _handshake(self) -> None:
        self._rpc("initialize", {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "mcd-budget-solver", "version": "1.0.0"},
        })
        try:
            self._rpc("notifications/initialized", {})
        except McdMcpError:
            # 通知类调用无返回体，失败不影响后续
            pass

    # -- 公共接口 --------------------------------------------------------

    def call(self, name: str, args: dict | None = None) -> str:
        """调用一个 Tool，返回其文本内容。"""
        self._throttle()
        result = self._rpc("tools/call", {"name": name, "arguments": args or {}})
        if not result:
            raise McdMcpError(f"{name} 无返回")
        for block in result.get("content", []):
            if block.get("type") == "text":
                return block["text"]
        raise McdMcpError(f"{name} 返回中无文本内容")

    def call_json(self, name: str, args: dict | None = None) -> dict:
        """调用 Tool 并提取末尾的 JSON 数据块。

        MCP 返回是「说明文档 + 原始JSON」的markdown 混排，真正的数据在
        最后一个以 {"success": 开头的 JSON 块里。
        """
        text = self.call(name, args)
        start = text.rfind('{"success"')
        if start < 0:
            raise McdMcpError(f"{name} 返回中未找到数据块")
        depth = 0
        for idx in range(start, len(text)):
            if text[idx] == "{":
                depth += 1
            elif text[idx] == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[start:idx + 1])
        raise McdMcpError(f"{name} 的JSON 数据块不完整")

    def try_call_json(self, name: str, args: dict | None = None) -> dict | None:
        """调用失败时返回 None 而不抛异常。用于降级路径。"""
        try:
            payload = self.call_json(name, args)
        except McdMcpError:
            return None
        if not payload.get("success"):
            return None
        return payload

    def tools(self) -> list[dict]:
        if self._tools is None:
            result = self._rpc("tools/list")
            self._tools = (result or {}).get("tools", [])
        return self._tools


# ---------------------------------------------------------------------------
# 业务封装
# ---------------------------------------------------------------------------


class McdStore:
    """把 MCP 数据映射为求解器的领域模型。所有方法只读。"""

    def __init__(self, client: McdClient, *, be_type: int = BE_TYPE_DINE_IN,
                 order_type: int = ORDER_TYPE_DINE_IN) -> None:
        self.client = client
        self.be_type = be_type
        self.order_type = order_type

    def find_store(self, city: str, keyword: str = "麦当劳") -> list[dict]:
        """查找门店。

        必须传 `searchType=2` 且同时给 city 与 keyword —— `searchType=1`
        走的是收藏列表，未收藏门店时必定返回错误。
        """
        payload = self.client.try_call_json(
            "query-nearby-stores",
            {"beType": self.be_type, "searchType": 2, "city": city, "keyword": keyword},
        )
        return (payload or {}).get("data") or []

    def get_menu(self, store_code: str) -> list[MenuItem]:
        """拉取门店在售菜单并映射为 MenuItem。

        `query-meals` 的返回是三层结构：
          data.categories[].meals[]只给 code 与 tags，
          真正的名称与价格在 data.meals[code] 这个字典里。
        两者需要按 code 合并才能得到完整信息。
        """
        payload = self.client.try_call_json("query-meals", {
            "storeCode": store_code,
            "beType": self.be_type,
            "orderType": self.order_type,
        })
        if not payload:
            return []
        data = payload.get("data") or {}
        details = data.get("meals") or {}

        items: list[MenuItem] = []
        seen: set[str] = set()
        for category in data.get("categories") or []:
            cat_name = (category.get("name") or "").replace("\n", "/")
            for stub in category.get("meals") or []:
                code = stub.get("code")
                if not code or code in seen:
                    continue
                seen.add(code)
                detail = details.get(code) or {}
                name = detail.get("name")
                price = detail.get("currentPrice")
                if not name or price is None:
                    continue
                name = str(name)
                # 分类名并入标签，使「鸡腿堡」「三件套」等词可用于忌口与套餐判定
                tags = tuple(stub.get("tags") or ()) + (cat_name,)
                is_combo = self._looks_like_combo(name, cat_name)
                items.append(MenuItem(
                    code=code,
                    name=name,
                    price=float(price),
                    tags=tags,
                    is_combo=is_combo,
                ))
        return items

    @staticmethod
    def _looks_like_combo(name: str, category: str) -> bool:
        """判定是否套餐。

        菜单里套餐的实际命名规则是「XX三件套」「XX四件套」，
        而 category 叫「精选单人餐」时未必是套餐，所以以名称特征为准。
        """
        return any(k in name for k in ("三件套", "四件套", "套餐", "桶"))

    def get_store_coupons(self, store_code: str) -> list[Coupon]:
        """取当前门店可用券。

        实测该 Tool 返回结构化 JSON，字段完整可用：
        `couponId` / `couponCode` / `title` / `tradeDateTime` /
        `products[].productCode`（适用商品）。

        **券数因店而异。** 实测广州塔餐厅返回 0 张，同城其他门店各有 3 张，
        所以选店策略应当优先挑有券的门店。
        """
        payload = self.client.try_call_json("query-store-coupons", {
            "storeCode": store_code,
            "beType": self.be_type,
            "orderType": self.order_type,
        })
        return self._map_coupons((payload or {}).get("data") or [])

    def _map_coupons(self, rows: Sequence[dict]) -> list[Coupon]:
        """映射 query-store-coupons 的返回，保留适用商品编码。

        `applicable_codes` 是券能否生效的硬约束：券只抵扣它指定的商品，
        挂到其他商品上`calculate-price` 会直接报错。
        """
        coupons: list[Coupon] = []
        for row in rows:
            cid = row.get("couponId")
            code = row.get("couponCode")
            if not cid or not code:
                continue
            applicable = tuple(
                p.get("productCode")
                for p in (row.get("products") or [])
                if p.get("productCode")
            )
            coupons.append(Coupon(
                coupon_id=cid,
                name=row.get("title") or "优惠券",
                threshold=0.0,
                coupon_code=code,
                applicable_codes=applicable,
                expires_at=row.get("tradeDateTime") or "",
            ))
        return coupons

    def get_my_coupons_raw(self, page: int = 1, page_size: int = 20) -> str:
        """取账户级券的原始文本。

        **这个 Tool 无法用于算价。** 实测其返回只有 markdown 文本，
        不含 couponId 与 couponCode 字段，只有券名、价格、有效期和图片 URL。
        缺了这两个标识符就无法传给 `calculate-price`。

        保留此方法是为了向用户展示账户持券情况（例如提示"你还有 3 张券
        今天到期"），**不要**把它的输出喂给求解器。
        """
        return self.client.call("query-my-coupons", {"page": page, "pageSize": page_size})

    def pick_stores_with_coupons(self, city: str, limit: int = 5,
                                 keyword: str = "麦当劳") -> list[tuple[dict, list[Coupon]]]:
        """找出有可用券的门店。

        这是选店的关键：由于券是门店级发放，直接取第一家门店很可能一张券都没有。
        实测广州第一家门店券数为 0，第二家起各有 3 张。
        """
        results: list[tuple[dict, list[Coupon]]] = []
        for store in self.find_store(city, keyword)[:limit]:
            coupons = self.get_store_coupons(store["storeCode"])
            if coupons:
                results.append((store, coupons))
        return results

    def bind_coupons(self) -> str:
        """一键领券。会修改用户账户状态，调用前必须取得用户确认。"""
        return self.client.call("auto-bind-coupons", {})

    def calculate(self, store_code: str, combo: Combo) -> PriceQuote:
        """调用 `calculate-price` 试算一个组合。

        三处易错点，都是实测踩过的：
        - items 的字段是 `productCode` 而非 `code`。
        - 券必须挂到它适用的商品上，否则服务端直接报错。
        - 返回金额单位是分，Quote.from_yuan 已做换算。
        """
        return self.calculate_raw(store_code, combo).to_quote()

    def calculate_raw(self, store_code: str, combo: Combo) -> Quote:
        """同 calculate，但返回带商品明细的 Quote，供调试与展示使用。"""
        items = self.build_items(combo)
        if items is None:
            raise McdMcpError("组合中的券找不到适用商品，无法试算")
        payload = self.client.try_call_json("calculate-price", {
            "storeCode": store_code,
            "beType": self.be_type,
            "orderType": self.order_type,
            "items": items,
        })
        if not payload:
            raise McdMcpError("calculate-price 调用失败")
        return Quote.from_yuan(payload.get("data") or {})

    @staticmethod
    def build_items(combo: Combo) -> list[dict] | None:
        """构造 calculate-price 的 items 数组。

        每张券挂到它适用的第一件商品上。实测服务端只对挂载正确的券抵扣，
        挂错会直接返回错误。一张券若在组合里找不到适用商品，返回 None
        表示这次试算无法进行。
        """
        by_code = {item.code: item for item in combo.items}
        items: list[dict] = [
            {"productCode": item.code, "quantity": 1} for item in combo.items
        ]
        index = {item["productCode"]: i for i, item in enumerate(items)}

        for coupon in combo.coupons:
            target = None
            for code in coupon.applicable_codes:
                if code in by_code:
                    target = code
                    break
            if target is None:
                # 该券在这组商品里无处可挂，试算无意义
                return None
            slot = items[index[target]]
            # 同商品挂多张券时用数组承载，实测服务端支持
            if "couponId" in slot:
                slot.setdefault("couponIds", []).append(coupon.coupon_id)
            else:
                slot["couponId"] = coupon.coupon_id
                slot["couponCode"] = coupon.coupon_code
        return items


# ---------------------------------------------------------------------------
# 联调自检
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    city = sys.argv[1] if len(sys.argv) > 1 else "广州"
    client = McdClient()
    print(f"工具数: {len(client.tools())}")

    store = McdStore(client)
    stores = store.find_store(city)
    if not stores:
        print(f"在{city} 未找到门店，尝试用其他城市名")
        raise SystemExit(1)

    first = stores[0]
    code = first["storeCode"]
    print(f"门店: {first.get('storeName')}  code={code}  distance={first.get('distance')}")

    menu = store.get_menu(code)
    print(f"菜单: {len(menu)} 个餐品")
    for item in menu[:8]:
        flag = "套餐" if item.is_combo else "单品"
        print(f"  {item.code:<14}{item.name[:22]:<24}{item.price:>7}元  {flag}")

    store_coupons = store.get_store_coupons(code)
    my_coupons = store.get_my_coupons()
    print(f"门店券 {len(store_coupons)} 张，账户券 {len(my_coupons)} 张")

    if menu:
        probe = Combo(items=menu[:1])
        quote = store.calculate_raw(code, probe)
        print(f"试算 {menu[0].name}: 原价 {quote.original_price} 元 "
              f"→ 实付 {quote.price} 元")
