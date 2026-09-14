def render(i18n):
    return [
        i18n.get("menu-main-title"),
        i18n.get("only-in-second", who="me"),
        i18n.balance.info(amount=1, currency="X", _path="wallet/balance.ftl"),
        i18n.get("star", count=2),
    ]
