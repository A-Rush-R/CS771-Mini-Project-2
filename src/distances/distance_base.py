class Distance:

    def update(self):
        raise NotImplementedError("Subclasses must implement this method")

    def distance(self, x, y):
        raise NotImplementedError("Subclasses must implement this method")