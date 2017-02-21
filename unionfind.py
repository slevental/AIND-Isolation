class UnionFind:
    """
    Simple implementation of Union-Find algorithm
    """
    def __init__(self, n):
        self._obj_to_id = {}
        self._next_id = 0
        self._ids = list(range(n))
        self._components = [1] * n

    def _to_id(self, obj):
        if obj not in self._obj_to_id:
            self._obj_to_id[obj] = self._next_id
            self._next_id += 1
        return self._obj_to_id[obj]

    def _get_root(self, i):
        i = self._to_id(i)
        tmp = i
        while tmp != self._ids[tmp]:
            self._ids[tmp] = self._ids[self._ids[tmp]]
            tmp = self._ids[tmp]
        return tmp

    def components(self, x):
        return self._components[self._get_root(x)]

    def union(self, x, y):
        i = self._get_root(x)
        j = self._get_root(y)
        if i == j:
            return
        if self._components[i] < self._components[j]:
            self._ids[i] = j
            self._components[j] += self._components[i]
        else:
            self._ids[j] = i
            self._components[i] += self._components[j]
