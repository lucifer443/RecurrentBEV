import datetime
import warnings

from mmengine import DefaultScope


def register_all_modules(init_default_scope: bool = True) -> None:
    """Register all modules in bev_perception into the registries.

    Args:
        init_default_scope (bool): Whether initialize the default scope.
            When `init_default_scope=True`, the global default scope will be
            set to `bev`, and all registries will build modules from mmdet3d's
            registry node. To understand more about the registry, please refer
            to https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/registry.md
            Defaults to True.
    """  # noqa
    import bev.datasets
    import bev.evaluation
    import bev.models
    if init_default_scope:
        never_created = DefaultScope.get_current_instance() is None \
                        or not DefaultScope.check_instance_created('bev')
        if never_created:
            DefaultScope.get_instance('bev', scope_name='bev')
            return
        current_scope = DefaultScope.get_current_instance()
        if current_scope.scope_name != 'bev':
            warnings.warn('The current default scope '
                          f'"{current_scope.scope_name}" is not "bev", '
                          '`register_all_modules` will force the current'
                          'default scope to be "bev". If this is not '
                          'expected, please set `init_default_scope=False`.')
            # avoid name conflict
            new_instance_name = f'bev-{datetime.datetime.now()}'
            DefaultScope.get_instance(new_instance_name, scope_name='bev')
