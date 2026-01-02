import sys


def error_message_detail(error, error_detail: sys):
    try:
        _, _, exc_tb = error_detail.exc_info()
        if exc_tb is not None:
            file_name = exc_tb.tb_frame.f_code.co_filename
            line_number = exc_tb.tb_lineno
        else:
            file_name = "<unknown>"
            line_number = 0
    except Exception:
        file_name = "<unknown>"
        line_number = 0

    error_message = f"Error occurred in script: {file_name} at line: {line_number} | error message: {str(error)}"
    return error_message


class SignException(Exception):
    def __init__(self, error_message, error_detail):
        """
        :param error_message: error message in string format
        """
        super().__init__(error_message)

        self.error_message = error_message_detail(
            error_message, error_detail=error_detail
        )

    def __str__(self):
        return self.error_message

