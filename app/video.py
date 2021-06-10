import time
from uwsgidecorators import thread

running = {}


def split_video(session_id: str, filename: str):
    """
    TODO: split video into proper files.
    Original video file is in "uploads/{session_id}/{filename}"
    Read the video file and divide it into frames and blocks, again.

    Save these to "processing/{session_id}/{filename}/{block_number}/{frame_number}/000.png
    Note that our deep learning model uses frame-major order dataset, so those would saved as that convention.
    """


@thread
def process(session_id: str, filename: str):
    if session_id not in running:
        running[session_id] = []

    if filename in running[session_id]:
        return False

    running[session_id].append(filename)
    """
    TODO: Process it!
    Using result of split_video, run tensorflow model.
    Merge and save these result into "processed/{session_id}/{filename}"
    """
    # Start
    time.sleep(10)
    # End
    running[session_id].remove(filename)

