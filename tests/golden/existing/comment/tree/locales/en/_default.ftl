### Resource comment at the top of the file


## Group comment


# Standalone comment (blank line below, so it is not attached)

hello-user = Hello, { $name }!

# unused-key = This one is not used anymore


# kwargs-changed = Old { $old }


# moved-key = I live in _default but code says wallet

multi =
    First line
    Second { $a } line
select-msg =
    { $count ->
        [one] One item
       *[other] { $count } items
    }
with-attr = Value
    .title = Tooltip
-brand = FlyBot
uses-term = Powered by { -brand }
ref-parent = See { ref-child }
ref-child = I am referenced only

# # Attached comment on a stale key
# stale-with-comment = has comment

same-kwargs-different-order = { $b } and { $a }
func-call = { NUMBER($n) } items
nested-placeable = {{ $inner }}
new-key = new-key{ $who }
kwargs-changed = kwargs-changed{ $new }
