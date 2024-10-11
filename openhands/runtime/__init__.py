from openhands.runtime.impl.e2b.sandbox import E2BBox


def get_runtime_cls(name: str):
    # Local imports to avoid circular imports
    if name == 'eventstream':
        from openhands.runtime.impl.client.runtime import EventStreamRuntime

        return EventStreamRuntime
    elif name == 'e2b':
        from openhands.runtime.impl.e2b.runtime import E2BRuntime

        return E2BRuntime
    elif name == 'remote':
        from openhands.runtime.impl.remote.runtime import RemoteRuntime

        return RemoteRuntime
    else:
        raise ValueError(f'Runtime {name} not supported')


__all__ = [
    'E2BBox',
    'get_runtime_cls',
]
