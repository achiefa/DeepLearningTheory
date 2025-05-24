import logging
import sys

import blessings

t = blessings.Terminal()


def is_in_notebook():
    """Check if the script is running in a Jupyter notebook."""
    try:
        from IPython import get_ipython

        if get_ipython() is not None and "IPKernelApp" in get_ipython().config:
            return True
        else:
            return False
    except ImportError:
        return False


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

    def __init__(self, stream=None):
        super().__init__(stream)
        self.in_notebook = is_in_notebook()
        self.colors = self.__build_colors_dict()

    def __build_colors_dict(self):
        if not self.in_notebook:
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
        else:
            colors = {
                logging.DEBUG: {
                    "[%(levelname)s]: ": {"color": "blue", "bold": True},
                    "%(message)s": {"bold": False},
                },
                logging.INFO: {
                    "[%(levelname)s]: ": {"color": "green", "bold": True},
                    "%(message)s": {"bold": False},
                },
                logging.WARNING: {
                    "[%(levelname)s]: ": {"color": "yellow", "bold": True},
                    "%(message)s": {"bold": False},
                },
                logging.ERROR: {
                    "[%(levelname)s]: ": {"color": "red", "bold": True},
                    "%(message)s": {"bold": True},
                },
                logging.CRITICAL: {
                    "[%(levelname)s]: ": {"color": "darkred", "bold": True},
                    "%(message)s": {"bold": True},
                },
            }
        return colors

    def __create_html_string(self, msg, color=None, is_bold=False):
        """Create a string with HTML formatting"""

        # Use CSS variables to adapt to the theme when color is not specified
        if color is None:
            # This uses the currentColor value which inherits from the parent
            # It will be white in dark theme and black in light theme
            style = "color: var(currentColor, white)"
        else:
            style = f"color: {color}"

        if is_bold:
            style += "; font-weight: bold"

        html = f'<span style="{style}">{msg}</span>'
        return html

    def format(self, record):
        levelcolors = self.colors[record.levelno]

        # If running in a Jupyter notebook, use HTML formatting
        if self.in_notebook:
            html = f"<div>"
            for s, subs in levelcolors.items():
                html += f'{self.__create_html_string(s, subs.get("color", None), subs.get("bold", False))}'
            html += "</div>"

            return logging.Formatter(html).format(record)
        else:
            fmt = self._fmt
            for s, subs in levelcolors.items():
                fmt = fmt.replace(s, subs(s))
            return logging.Formatter(fmt).format(record)

    def emit(self, record):
        try:
            msg = self.format(record)
            if self.in_notebook:
                try:
                    from IPython.display import HTML, display

                    display(HTML(msg))
                except Exception as e:
                    stream = self.stream
                    stream.write(
                        msg.replace("<div>", "")
                        .replace("</div>", "\n")
                        .replace(
                            '<span style="color:'
                            + self.colors.get(record.levelno, {"color": "black"})[
                                "color"
                            ]
                            + ';font-weight:bold;">',
                            "",
                        )
                        .replace("</span>", "")
                    )
            else:
                stream = self.stream
                stream.write(msg + self.terminator)
                self.flush()
        except Exception:
            self.handleError(record)


# Add a setup function that can be called to configure the logger
def setup_logger(name=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Remove all existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Add our notebook-aware handler
    handler = MyHandler(sys.stdout)
    logger.addHandler(handler)

    return logger
