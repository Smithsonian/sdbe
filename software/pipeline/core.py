
class Args(object):

    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


class Steps(object):

    def __init__(self, *subs):
        self.subs = subs

    def gen_wrap(self, func, _input):
        for val in _input:
            yield func(val)

    def evaluate(self, _input):
        self.sub_steps = [_input, ]
        for sub in self.subs:
            if isinstance(sub, Steps):
                sub.evaluate(self.sub_steps[-1])
                sub_step = sub
            else:
                sub_step = self.gen_wrap(sub, self.sub_steps[-1])
            self.sub_steps.append(sub_step)

    def __iter__(self):
        return self

    def __next__(self):
        return self.sub_steps[-1].next()

    def next(self):
        return self.__next__()
