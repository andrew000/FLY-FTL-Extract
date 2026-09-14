from aiogram import Router
from aiogram.types import Message
from aiogram_i18n import I18nContext, L, LazyFilter

router = Router()


@router.message()
async def start(message: Message, i18n: I18nContext, extra: dict) -> None:
    """Docstring mentioning i18n.get("not-a-call-in-docstring")."""
    await message.answer(i18n.get("menu-main-title"))
    await message.answer(i18n.get("hello-user", name=message.from_user.full_name))
    await message.answer(i18n.balance.info(currency="UAH", amount=10, _path="wallet/balance.ftl"))
    await message.answer(i18n.some.key_1(_path="wallet"))
    await message.answer(i18n.get("empty-path", _path=""))
    await message.answer(i18n.get("outer", inner=i18n.get("inner", n=1)))
    await message.answer(i18n.get(f"dynamic-{message.text}"))
    await message.answer(i18n.get("concat-" + "key"))
    await message.answer(i18n.get("zeta-order", zeta=1, alpha=2, mid=3))
    await message.answer(i18n.get("hello-user", name="again"))
    await message.answer(i18n.get("star", **extra))
    await message.answer(i18n.get("star", count=1))
    await message.answer(i18n.get(key="not-positional"))
    await message.answer(i18n.get())
    await message.answer(i18n.get(*["not-literal"]))
    await message.answer(i18n("call-name"))
    await message.answer(L("lazy-l"))
    await message.answer(LazyFilter("lazy-filter", x=1))
    await message.answer(i18n_utils.get("not-a-key"))
    await message.answer(other.i18n.get("prefix-without-p"))
    await message.answer(i18n.get("bytes-key", b=b"raw"))
    await message.answer(
        i18n.get(
            "multi-line-call",
            first=1,
            second=2,
        )
    )
    await message.answer(i18n.deep.attr.chain.here())
    await message.answer(i18n.get("dotted.key.name"))
    await message.answer(i18n.get("hello-user", name=x).upper())
    await message.answer(i18n.get("hello-user", name=x)["x"])
    await message.answer(i18n.only_attr())
    await message.answer(i18n.get("dup-in-lambda", a=(lambda: i18n.get("in-lambda"))()))
    await message.answer(i18n.get("unicode-ключ", тест=1))
    await message.answer(i18n.get("with-comment"))  # i18n.get("comment-not-a-key")
