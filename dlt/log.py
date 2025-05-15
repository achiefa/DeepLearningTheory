import logging

import blessings

t = blessings.Terminal()


class MyHandler(logging.StreamHandler):
    colors = {
        logging.DEBUG: {"[%(levelname)s]:": t.bold_blue},
        logging.INFO: {"[%(levelname)s]:": t.bold_green},
        logging.WARNING: {"[%(levelname)s]:": t.bold_yellow},
        logging.ERROR: {"[%(levelname)s]:": t.bold_red, "%(message)s": t.bold},
        logging.CRITICAL: {
            "[%(levelname)s]:": t.bold_white_on_red,
            "%(message)s": t.bold,
        },
    }

    _fmt = "[%(levelname)s]: %(message)s"

    def format(self, record):
        levelcolors = self.colors[record.levelno]
        fmt = self._fmt
        for s, subs in levelcolors.items():
            fmt = fmt.replace(s, subs(s))
        return logging.Formatter(fmt).format(record)
