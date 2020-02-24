class Character():
    def __init__(self, char:str):
        self.character = char
        self.images = []

    def addImage(self, image):
        self.images.append(image)