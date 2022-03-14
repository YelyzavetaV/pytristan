class ObjectManager:
    """Class to store and manage existing instances of objects."""

    def ids(self):
        """Get a list of ids of all stored instances."""

        ids = [
            [int(s) for s in key.split(sep="-") if s.isdigit()]
            for key in vars(self).keys()
        ]

        return [id[0] if len(id) == 1 else id for id in ids]
