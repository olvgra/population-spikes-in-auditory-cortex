class Stimuli:
    def __init__(self, columns):
        """
        Initialises an empty stimulus dictionary.
        Keys are column indices, and values are lists of stimulus dictionaries.
        """
        self.stimuli = [[] for _ in range(columns)]

    def add_stimulus(self, col, A, start, stop):
        """
        Adds a stimulus to a specific column.

        Parameters:
            col: Column index.
            A: Amplitude of the stimulus.
            start: Start time of the stimulus.
            stop: Stop time of the stimulus.
        """
        self.stimuli[col-1].append({"amplitude": A, "begin": start, "end": stop})

    def get_stimuli(self, column):
        """
        Returns the stored stimuli list for the specified column.
        """
        return self.stimuli[column]
