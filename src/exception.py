"""Custom exception that records where an error happened."""
import traceback


class CustomException(Exception):
    def __init__(self, error: Exception):
        # Point at the innermost frame of the original traceback
        frames = traceback.extract_tb(error.__traceback__)
        location = f"{frames[-1].filename}, line {frames[-1].lineno}" if frames else "unknown location"
        self.message = f"{type(error).__name__} in {location}: {error}"
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message
