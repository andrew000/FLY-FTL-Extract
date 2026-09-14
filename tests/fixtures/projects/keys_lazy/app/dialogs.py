from aiogram_dialog import Dialog, Window
from aiogram_dialog.widgets.text import Const
from aiogram_i18n import LazyProxy
from aiogram_i18n.utils.aiogram_dialog import I18nFormat
from utils import LF

window = Window(
    I18nFormat("dialog-title", when=lambda *_: True, name="x"),
    I18nFormat("dialog-plain"),
    Const("not a key"),
)

LF("lf-key")
LazyProxy("lazy-proxy", user=1)
LazyProxy("lazy-proxy", user=2, when="ignored-here-too")
LF.get("lf-get")
LF.attr.chain(k=1)
i18n.get("default-key-not-extracted")
L("l-not-extracted")
LazyFilter("lazy-filter-not-extracted")
