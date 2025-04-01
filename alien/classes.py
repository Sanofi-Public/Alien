from functools import wraps
from inspect import currentframe

try:
    from typing import final as weak_final
except ImportError:

    def weak_final(fn):
        return fn


def parent_locals():
    return currentframe().f_back.f_back.f_locals


def final(fn):
    """
    Decorates a method, to prohibit subclasses from overriding it (unless the
    overriding method is decorated with `@override`). This prohibition is
    actually enforced (during subclass creation), unlike with python's
    `typing.final` decorator, which is only used by external type checkers.
    Decorating with this version of `final` does also apply `typing.final`,
    so no need to use both.
    """
    exec(injected_code, parent_locals(), globals())
    fn.__final__ = True
    return weak_final(fn)


def override(fn):
    """
    Decorates a method to allow it to override `@final` methods.
    """
    fn.__override__ = True
    return fn


def init_subclass(subclass, superclass, **kwargs):  # NOSONAR
    for f_super in superclass.__dict__.values():
        if groups := getattr(f_super, "__abstract_group__", False):
            for group in groups:
                if not hasattr(subclass, "__abstract_req__"):
                    subclass.__abstract_req__ = {group: abstract_req[group]}
                    old_init = subclass.__init__

                    @wraps(subclass.__init__)
                    def abstract_init(self, *args, **kwargs):
                        old_init(self, *args, **kwargs)
                        for group, n in subclass.__abstract_req__.items():
                            k = n
                            for meth_name in abstract_groups[group]:
                                if getattr(self, meth_name) != getattr(superclass, meth_name).__get__(self):
                                    k -= 1
                            if k > 0:
                                raise TypeError(
                                    f"Subclasses of {superclass.__name__} must implement at least {n} of "
                                    + ", ".join(abstract_groups[group])
                                )

                    subclass.__init__ = abstract_init

                else:
                    subclass.__abstract_req__[group] = abstract_req[group]

        if getattr(f_super, "__final__", False):
            for cls in subclass.__mro__:
                if cls == superclass:
                    raise TypeError(
                        f"Class `{subclass.__qualname__}` cannot override @final "
                        f"method `{name}` (defined in class `{superclass.__name__}`). Use @override "
                        "decorator if you must."
                    )
                elif getattr(getattr(cls, name, None), "__override__", False):
                    break


def abstract_group(id, n=1):
    """
    Decorates a group of methods so that at least `n` (typically 1)
    of them must be overriden by a subclass. Each group is determined
    by its `id`, which must be a hashable key.
    """

    def abstract_decorator(fn):
        if "__abstract_group__" not in fn.__dict__:
            fn.__abstract_group__ = set()
        fn.__abstract_group__.add(id)

        if id not in abstract_groups:
            abstract_groups[id] = set()
        abstract_groups[id].add(fn.__name__)
        abstract_req[id] = n

        exec(injected_code, parent_locals(), globals())

        return fn

    return abstract_decorator


abstract_groups = {}
abstract_req = {}


injected_code = """
if "__alien_abstract_class__" not in locals():
    __alien_abstract_class__ = True
    old_init_name = "__init_subclass_old_abstract_group__"

    old_init = locals().get("__init_subclass__", lambda *a, **k : None)
    locals()[old_init_name] = old_init
    def __init_subclass__(cls, old_init=old_init, **kwargs):
        old_init(cls, **kwargs)
        init_subclass(cls, superclass=__class__, **kwargs)
"""
