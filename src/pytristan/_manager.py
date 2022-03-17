class ObjectManager:
    """Class to store and manage existing instances of objects."""

    def nums(self):
        """Get a list of ids of all stored instances."""

        nums = [
            [int(s) for s in key.split(sep="-") if s.isdigit()]
            for key in vars(self).keys()
        ]

        return [num[0] if len(num) == 1 else num for num in nums]
