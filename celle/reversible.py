import torch.nn as nn

# for routing arguments into the functions of the reversible layer
def route_args(router, args, depth):
    routed_args = [(dict(), dict()) for _ in range(depth)]
    matched_keys = [key for key in args.keys() if key in router]

    for key in matched_keys:
        val = args[key]
        for depth, ((f_args, g_args), routes) in enumerate(
            zip(routed_args, router[key])
        ):
            new_f_args, new_g_args = map(
                lambda route: ({key: val} if route else {}), routes
            )
            routed_args[depth] = ({**f_args, **new_f_args}, {**g_args, **new_g_args})
    return routed_args

class SequentialSequence(nn.Module):
    def __init__(self, layers, args_route={}, layer_dropout=0.0):
        super().__init__()
        assert all(
            len(route) == len(layers) for route in args_route.values()
        ), "each argument route map must have the same depth as the number of sequential layers"
        self.layers = layers
        self.args_route = args_route
        self.layer_dropout = layer_dropout

    def forward(self, x, **kwargs):
        args = route_args(self.args_route, kwargs, len(self.layers))
        layers_and_args = list(zip(self.layers, args))

        for (f, g), (f_args, g_args) in layers_and_args:
            x = x + f(x, **f_args)
            x = x + g(x, **g_args)
        return x