class Handler:
    def __init__(self, i18n):
        self.i18n = i18n
        self.other = i18n

    def run(self):
        self.i18n.get("self-get")
        self.i18n.self_attr.key(x=1)
        self.i18n("self-call")
        self.other.get("not-a-key")
        self.i18n.set_locale("uk")
        obj.self.i18n.get("deep-root-is-obj")
        self.get("self-is-not-a-key-itself")

    @classmethod
    def make(cls):
        cls.i18n.cls_key()
        cls.i18n.get("cls-get", a=1)
        cls.L("cls-l-name-call")
