import logging

logger = logging.getLogger(__name__)

class ProgressLogger:

    def __init__(self, logger, total, step):
        self.logger = logger
        self.total = total
        self.lastp = -1
        self.step = step

    def set_step(self, i):
        percentage = int(round(((i+1) / (self.total)) * 100))
        if percentage % self.step == 0:
            if self.lastp == percentage:
                return None
            self.lastp = percentage
            self.logger.info(f'Processing raster: {percentage:3d} %')





