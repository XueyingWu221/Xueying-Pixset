class father(object):
    def __init__(self):
        self.str = "world"
    def myprint(self):
        print("hello", self.str)


class child(father):
    # def __init__(self):
    #     super(child, self).__init__()
    pass


# child.myprint()
dad = father()
dad.myprint()
son = child()
son.myprint()
