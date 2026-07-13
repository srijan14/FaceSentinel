class KYCAutomationExceptions(BaseException):
    """ General KYC Automation Face Matching exception occurred """

    def __init__(self, message: str = None):
        self.message = message
        self.error_code = 777

    def __str__(self):
        """
        gets a string representation of this exception
        :return: string representation of exception
        """
        return self.message


class InvalidInputException(KYCAutomationExceptions):
    """

    """
    def __init__(self,message:str="Bad Input Data"):

        self.message = message
        self.error_code = 400


class ImageEnhancementException(KYCAutomationExceptions):
    """

    """
    def __init__(self, message: str = "Bad Input Data"):
        self.message = message
        self.error_code = 400
        
        
class InvalidOpcoException(KYCAutomationExceptions):
    """

    """
    def __init__(self, message: str = "Provide Valid Config Type"):
        self.message = message
        self.error_code = 400