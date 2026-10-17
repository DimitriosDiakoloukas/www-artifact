from training_oracle import Query
from oracle import relevant, occurrence

class ThreeQuery(Query):

    def __init__(self, graph, pair):
        super().__init__(graph, pair)
        self.gate = False
        self.clean[6] = 0
        self.owners = relevant(graph, pair, 3)
        self.O = 2 * occurrence(graph, pair, 3)[self.owners].astype(float) / self.den
        self.O[:, 6] = 0
