import datetime as dt

class Timer():
    """A simple Timer class to measure execution time"""

    def __init__(self):
        self.start_dt = None

    def start(self):
        """Start the timer by recording the current time."""
        self.start_dt = dt.datetime.now()

    def stop(self):
        """Stop the timer and print the time difference."""
        if self.start_dt is None:
            print("Timer was not started. Please call start() first.")
            return
        
        end_dt = dt.datetime.now()
        elapsed_time = end_dt - self.start_dt
        print(f"Time taken: {elapsed_time}")

        # Optionally, display in seconds or milliseconds
        elapsed_seconds = elapsed_time.total_seconds()
        print(f"Time taken in seconds: {elapsed_seconds:.2f}s")
        print(f"Time taken in milliseconds: {elapsed_seconds * 1000:.2f}ms")

# Example usage:
# timer = Timer()
# timer.start()
# <some code to time>
# timer.stop()
