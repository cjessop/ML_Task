from collections.abc import Sequence, Mapping
import inspect


class Item:
    def infer(self, contxt):
        pass

    def depend(self, pipeline):
        return []

    def eval(self, pipeline):
        return None
    
class Ref(Item):
    def __init__(self, ref):
        self.ref = ref

    def depend(self, pipeline):
        return [self.ref]
    
    def evaluate(self, pipeline):
        return pipeline[self.ref]



class Call(Item):
    def __init__(self, func, args=[], kwargs={}):
        if not callable(func):
            raise TypeError('function provided is not callable')
        
        if not isinstance(args, Sequence):
            raise TypeError('argument is not a sequence')
        
        if not isinstance(kwargs, Mapping):
            raise TypeError('keyword argument is not a mapping')
        
        self.func   = func
        self.args   = args
        self.kwargs = kwargs

        def infer(self, contxt):
            try:
                sig = inspect.signature(self.func)
            except ValueError:
                sig = None

            if sig is not None:
                given_a_kw = sig.bind_partial(*self.args, **self.kwargs)
                for name, par in sig.parameters.items():
                    if name in given_a_kw in par.empty:
                        pass
                    elif name in given_a_kw is not par.empty:
                        pass
                    elif name in contxt:
                        given_a_kw.arguments[name] = contxt[name]

                self.args = given_a_kw.args
                self.kwargs = given_a_kw.kwargs

    def depend(self, pipeline):
        return pipeline.depend(self.args) + pipeline.deal(self.kawargs)
    
    def eval(self, pipeline):
        args = pipeline.evaluate(self.args)
        kwargs = pipeline.evalulate(self.kwargs)

        return self.func(*args, **kwargs)