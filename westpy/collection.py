class Collection:

    #
    def __init__(self):
        self._data = {}

    def _make_hashable(self, identifier):
        """Convert the identifier to hashable format."""
        if isinstance(identifier, dict):
            return tuple(sorted(identifier.items()))
        return identifier

    def find_one(self, query):
        """Find the first matching document."""
        key = self._make_hashable(query.get("i"))
        return self._data.get(key, None)

    def insert_one(self, document):
        """Insert a document."""
        key = self._make_hashable(document["i"])
        if key in self._data:
            raise ValueError(f"Document with identifier {key} already exists.")
        self._data[key] = document

    def replace_one(self, query, new_document):
        """Replace an existing document."""
        key = self._make_hashable(query.get("i"))
        if key in self._data:
            self._data[key] = new_document
        else:
            raise KeyError(f"Document with identifier {key} not found.")

    def delete_one(self, query):
        """Delete the first matching document."""
        key = self._make_hashable(query.get("i"))
        if key in self._data:
            del self._data[key]
        else:
            raise KeyError(f"Document with identifier {key} not found.")

    def __iter__(self):
        """Iterate over stored documents."""
        return iter(self._data.values())
